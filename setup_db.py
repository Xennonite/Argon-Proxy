import pandas as pd
import sqlite3
import numpy as np
from faker import Faker

def setup_database():
    print("Reading raw_data.csv...")
    try:
        # Read the Kaggle dataset
        df = pd.read_csv('raw_data.csv')
    except FileNotFoundError:
        print("Error: raw_data.csv not found.")
        return

    print("Injecting realistic PII (Names & Emails)...")
    # Initialize Faker with an Indian locale to generate realistic regional data
    fake = Faker('en_IN')
    
    # Generate fake names and emails for every row in the dataset
    df['customer_name'] = [fake.name() for _ in range(len(df))]
    
    # Create realistic emails based on the names (removing spaces and adding domains)
    df['customer_email'] = [
        f"{str(name).replace(' ', '.').lower()}@example.com" for name in df['customer_name']
    ]

    print("Adding DPDP AI Training consent logic...")
    # DPDP Logic: 30% of users consent to AI training, 70% deny it
    df['ai_training_consent'] = np.random.choice([True, False], size=len(df), p=[0.3, 0.7])

    # Reorder columns so PII and Consent are at the front for easier viewing
    cols = ['customer_id', 'customer_name', 'customer_email', 'ai_training_consent'] + \
           [c for c in df.columns if c not in ['customer_id', 'customer_name', 'customer_email', 'ai_training_consent']]
    df = df[cols]

    print("Connecting to SQLite database...")
    # This creates the argon_proxy.db file in your folder
    conn = sqlite3.connect('argon_proxy.db')
    
    print("Writing data to 'customers' table...")
    df.to_sql('customers', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Success! 'argon_proxy.db' created with injected PII and DPDP consent.")

if __name__ == "__main__":
    setup_database()