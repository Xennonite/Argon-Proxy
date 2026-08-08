from fastapi import FastAPI
import sqlite3
import pandas as pd
import numpy as np
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

app = FastAPI(title="Argon Proxy API Gateway")

# Initialize Microsoft Presidio Privacy Engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def add_laplace_noise(value: float, epsilon: float = 1.0, sensitivity: float = 500.0) -> float:
    """Applies Differential Privacy Laplace noise using numpy."""
    if pd.isnull(value):
        return value
    # The scale of the noise is determined by the sensitivity divided by the privacy budget (epsilon)
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale)
    return round(float(value) + noise, 2)

def apply_presidio_mask(text: str) -> str:
    """Uses NLP to detect and redact entities like Names and Emails."""
    if not isinstance(text, str) or text.strip() == "":
        return text
    
    # 1. Analyze the text for sensitive entities
    results = analyzer.analyze(text=text, entities=["PERSON", "EMAIL_ADDRESS"], language="en")
    
    # 2. If nothing is found, return original text. Otherwise, anonymize it.
    if not results:
        return text
        
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text

@app.get("/api/export-training-data")
def export_data():
    """Intercepts database extraction and enforces DPDP consent."""
    
    # 1. Intercept the Data (Simulating a Data Scientist's SQL query)
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

    # 5. Recombine the secure dataset
    safe_df = pd.concat([consented_df, denied_df])
    
    # 6. Output the data as a JSON payload for the frontend dashboard
    return safe_df.to_dict(orient="records")