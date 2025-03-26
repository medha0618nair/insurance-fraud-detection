import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of records to generate
n_records = 10000

# Generate customer details
data = {
    'customer_id': [f'CUST{i:05d}' for i in range(n_records)],
    'customer_name': [fake.name() for _ in range(n_records)],
    'age': np.random.randint(25, 80, n_records),
    'gender': np.random.choice(['M', 'F'], n_records),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_records),
    'occupation': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Business', 'Other'], n_records),
    'income': np.random.randint(30000, 150000, n_records),
    'address': [fake.address() for _ in range(n_records)],
    'phone': [fake.phone_number() for _ in range(n_records)],
    'email': [fake.email() for _ in range(n_records)],
    'policy_number': [f'POL{i:06d}' for i in range(n_records)],
    'policy_type': np.random.choice(['Health', 'Life', 'Auto', 'Property'], n_records),
    'policy_start_date': [(datetime.now() - timedelta(days=np.random.randint(0, 365*3))).strftime('%Y-%m-%d') for _ in range(n_records)],
    'claim_date': [(datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d') for _ in range(n_records)],
    'claim_amount': np.random.randint(1000, 50000, n_records),
    'claim_reason': np.random.choice(['Accident', 'Illness', 'Property Damage', 'Theft', 'Natural Disaster'], n_records),
    'number_of_documents': np.random.randint(1, 10, n_records),
    'past_claims_count': np.random.randint(0, 5, n_records),
    'time_to_submit_claim': np.random.randint(1, 30, n_records),
    'claim_status': np.random.choice(['Pending', 'Approved', 'Rejected'], n_records)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate fraud labels based on various conditions
def determine_fraud(row):
    fraud_score = 0
    
    # High claim amount
    if row['claim_amount'] > 30000:
        fraud_score += 1
    
    # Low number of documents
    if row['number_of_documents'] < 3:
        fraud_score += 1
    
    # Frequent past claims
    if row['past_claims_count'] > 2:
        fraud_score += 1
    
    # Quick claim submission
    if row['time_to_submit_claim'] < 3:
        fraud_score += 1
    
    # High income but low claim amount
    if row['income'] > 100000 and row['claim_amount'] < 5000:
        fraud_score += 1
    
    # Return 1 if fraud score is high enough, 0 otherwise
    return 1 if fraud_score >= 2 else 0

# Add fraud labels
df['is_fraud'] = df.apply(determine_fraud, axis=1)

# Save to CSV
output_file = 'insurance_claims_data.csv'
df.to_csv(output_file, index=False)

print(f"Generated {n_records} records and saved to {output_file}")
print("\nData Summary:")
print(f"Total records: {len(df)}")
print(f"Fraud cases: {df['is_fraud'].sum()}")
print(f"Genuine cases: {len(df) - df['is_fraud'].sum()}")
print(f"\nFraud percentage: {(df['is_fraud'].sum() / len(df) * 100):.2f}%")

# Display sample of the data
print("\nSample of generated data:")
print(df.head()) 