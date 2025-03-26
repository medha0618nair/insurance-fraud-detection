import pandas as pd
import numpy as np
from datetime import datetime
from fraud_detection import FraudDetectionSystem

def create_fraud_flags(df):
    """Create fraud flags based on various heuristics"""
    # Convert date columns to datetime
    date_columns = ['TXN_DATE_TIME', 'POLICY_EFF_DT', 'LOSS_DT', 'REPORT_DT']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    # Calculate days between policy effective date and loss date
    df['days_to_loss'] = (df['LOSS_DT'] - df['POLICY_EFF_DT']).dt.days
    
    # Calculate claim amount to premium ratio
    df['claim_premium_ratio'] = df['CLAIM_AMOUNT'] / df['PREMIUM_AMOUNT']
    
    # Calculate suspicious score
    def calculate_suspicious_score(row):
        score = 0
        
        # High claim amount compared to premium (more than 5x)
        if row['claim_premium_ratio'] > 5:
            score += 1
        
        # Very short policy tenure before claim (less than 15 days)
        if row['days_to_loss'] < 15:
            score += 1
        
        # No police report for high severity incidents
        if row['INCIDENT_SEVERITY'] == 'Total Loss' and row['POLICE_REPORT_AVAILABLE'] == 0:
            score += 1
        
        # Suspicious incident timing (very late night)
        if row['INCIDENT_HOUR_OF_THE_DAY'] > 23 or row['INCIDENT_HOUR_OF_THE_DAY'] < 4:
            score += 1
        
        # Missing important documentation for high-value claims
        if pd.isna(row['AUTHORITY_CONTACTED']) and row['CLAIM_AMOUNT'] > 30000:
            score += 1
        
        # High risk segmentation with very high claim amount
        if row['RISK_SEGMENTATION'] == 'H' and row['CLAIM_AMOUNT'] > 75000:
            score += 1
        
        return score
    
    # Create fraud flags based on suspicious score
    df['suspicious_score'] = df.apply(calculate_suspicious_score, axis=1)
    df['FRAUD_FLAG'] = (df['suspicious_score'] >= 2).astype(int)
    
    return df

def analyze_fraud_patterns(df):
    """Analyze patterns in fraudulent claims"""
    fraud_cases = df[df['FRAUD_FLAG'] == 1]
    legitimate_cases = df[df['FRAUD_FLAG'] == 0]
    
    print("\nFraud Analysis Summary:")
    print("======================")
    print(f"Total Records: {len(df)}")
    print(f"Fraudulent Claims: {len(fraud_cases)}")
    print(f"Legitimate Claims: {len(legitimate_cases)}")
    print(f"Fraud Rate: {(len(fraud_cases) / len(df) * 100):.2f}%")
    
    print("\nFraud by Insurance Type:")
    print(fraud_cases['INSURANCE_TYPE'].value_counts(normalize=True))
    
    print("\nFraud by Incident Severity:")
    print(fraud_cases['INCIDENT_SEVERITY'].value_counts(normalize=True))
    
    print("\nFraud by Risk Segmentation:")
    print(fraud_cases['RISK_SEGMENTATION'].value_counts(normalize=True))
    
    print("\nAverage Claim Amount:")
    print(f"Fraudulent Claims: ${fraud_cases['CLAIM_AMOUNT'].mean():.2f}")
    print(f"Legitimate Claims: ${legitimate_cases['CLAIM_AMOUNT'].mean():.2f}")
    
    print("\nFraud Indicators:")
    print("----------------")
    print(f"Claims without police report: {len(df[df['POLICE_REPORT_AVAILABLE'] == 0])}")
    print(f"Claims with missing authority contact: {df['AUTHORITY_CONTACTED'].isna().sum()}")
    print(f"High-risk claims: {len(df[df['RISK_SEGMENTATION'] == 'H'])}")
    print(f"Total loss incidents: {len(df[df['INCIDENT_SEVERITY'] == 'Total Loss'])}")
    
    print("\nSuspicious Score Distribution:")
    print(df['suspicious_score'].value_counts().sort_index())

def main():
    # Load the data
    print("Loading insurance data...")
    df = pd.read_csv('insurance_data.csv')
    
    # Create fraud flags
    print("\nCreating fraud flags...")
    df = create_fraud_flags(df)
    
    # Analyze fraud patterns
    analyze_fraud_patterns(df)
    
    # Initialize fraud detection system
    fraud_system = FraudDetectionSystem()
    
    # Prepare features and train model
    print("\nPreparing features and training model...")
    X, y = fraud_system.prepare_features(df)
    
    if y is not None:
        print("\nTraining fraud detection model...")
        metrics = fraud_system.train_model(X, y)
        
        print("\nModel Performance:")
        print(metrics['classification_report'])
        
        # Save the model
        fraud_system.save_model()
        print("\nModel saved successfully!")
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = fraud_system.predict_fraud(df)
        
        # Analyze predictions
        predicted_fraud = sum(predictions['predictions'])
        print(f"\nPredicted Fraudulent Claims: {predicted_fraud}")
        print(f"Predicted Fraud Rate: {(predicted_fraud / len(df) * 100):.2f}%")
        
        # Save processed data with predictions
        df['predicted_fraud'] = predictions['predictions']
        df['fraud_probability'] = [prob[1] for prob in predictions['probabilities']]
        df.to_csv('processed_insurance_data.csv', index=False)
        print("\nProcessed data saved to 'processed_insurance_data.csv'")
    else:
        print("No fraud labels found in the data!")

if __name__ == "__main__":
    main() 