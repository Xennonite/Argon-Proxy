from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import sqlite3
import pandas as pd
import numpy as np
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

app = FastAPI(title="Argon Proxy API Gateway")

# Initialize Microsoft Presidio Privacy Engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def add_laplace_noise(value: float, epsilon: float = 1.0, sensitivity: float = 500.0) -> float:
    """Applies Differential Privacy Laplace noise and clamps illogical outliers."""
    if pd.isnull(value):
        return value
    
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale)
    noisy_value = round(float(value) + noise, 2)
    
    # ML Data Constraint: Revenue cannot be mathematically negative
    return max(0.0, noisy_value)

def apply_presidio_mask(text: str) -> str:
    """Uses NLP to detect and redact entities, with strict fallbacks for tabular data."""
    if not isinstance(text, str) or text.strip() == "":
        return text
    
    # 1. Analyze the text for sensitive entities
    results = analyzer.analyze(text=text, entities=["PERSON", "EMAIL_ADDRESS"], language="en")
    
    # 2. Hackathon Failsafe: If the Western NLP model completely misses the Indian name, force redact.
    if not results:
        return "[REDACTED]"
        
    # 3. HTML Fix: Override the default <PERSON> tag with a web-safe string
    anonymized = anonymizer.anonymize(
        text=text, 
        analyzer_results=results,
        operators={
            "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})
        }
    )
    
    # 4. Partial Redaction Fix: If it only caught the surname (e.g. "Ishita [REDACTED]"), scrub the whole cell.
    if "[REDACTED]" in anonymized.text:
        return "[REDACTED]"
        
    return anonymized.text

@app.get("/api/export-training-data")
def export_data():
    """Intercepts database extraction and enforces DPDP consent."""
    
    # 1. Intercept the Data
    conn = sqlite3.connect('argon_proxy.db')
    df = pd.read_sql_query("SELECT * FROM customers LIMIT 50", conn) 
    conn.close()

    # 2. Check Consent Flags (DPDP Enforcement)
    consented_df = df[df['ai_training_consent'] == 1].copy()
    denied_df = df[df['ai_training_consent'] == 0].copy()

    # 3. Apply Microsoft Presidio (NLP Text Masking)
    denied_df['customer_name'] = denied_df['customer_name'].apply(apply_presidio_mask)
    denied_df['customer_email'] = denied_df['customer_email'].apply(apply_presidio_mask)

    # 4. Apply Custom Differential Privacy (Mathematical Noise to numerical data)
    denied_df['revenue'] = denied_df['revenue'].apply(
        lambda x: add_laplace_noise(x, epsilon=1.0, sensitivity=500.0)
    )

    # 5. Sorting Fix: Restore the original database index so left and right tables match perfectly
    safe_df = pd.concat([consented_df, denied_df]).sort_index()
    
    # 6. Output both datasets as a JSON payload for the frontend dashboard
    return {
        "raw_data": df.to_dict(orient="records"),
        "safe_data": safe_df.to_dict(orient="records")
    }

# Mount the HTML dashboard (MUST be placed at the bottom after API routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")