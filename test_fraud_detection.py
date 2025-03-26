import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from process_insurance_data import create_fraud_flags
from fraud_detection import FraudDetectionSystem

def generate_test_claim(
    insurance_type='Health',
    claim_amount=10000,
    premium_amount=1000,
    incident_severity='Minor Loss',
    police_report=1,
    incident_hour=12,
    risk_segment='L',
    authority_contacted='Police',
    days_since_policy=100
):
    """Generate a single test claim with specified parameters"""
    
    current_date = datetime.now()
    policy_date = current_date - timedelta(days=days_since_policy)
    loss_date = current_date - timedelta(days=5)
    
    return {
        'TXN_DATE_TIME': current_date.strftime('%Y-%m-%d %H:%M:%S'),
        'TRANSACTION_ID': f'TEST_{np.random.randint(10000)}',
        'CUSTOMER_ID': f'CUST_{np.random.randint(10000)}',
        'POLICY_NUMBER': f'POL_{np.random.randint(10000)}',
        'POLICY_EFF_DT': policy_date.strftime('%Y-%m-%d'),
        'LOSS_DT': loss_date.strftime('%Y-%m-%d'),
        'REPORT_DT': current_date.strftime('%Y-%m-%d'),
        'INSURANCE_TYPE': insurance_type,
        'PREMIUM_AMOUNT': premium_amount,
        'CLAIM_AMOUNT': claim_amount,
        'INCIDENT_SEVERITY': incident_severity,
        'POLICE_REPORT_AVAILABLE': police_report,
        'INCIDENT_HOUR_OF_THE_DAY': incident_hour,
        'RISK_SEGMENTATION': risk_segment,
        'AUTHORITY_CONTACTED': authority_contacted,
        'AGE': np.random.randint(25, 75),
        'TENURE': np.random.randint(1, 20),
        'EMPLOYMENT_STATUS': np.random.choice(['Employed', 'Self-employed', 'Unemployed']),
        'MARITAL_STATUS': np.random.choice(['Single', 'Married', 'Divorced']),
        'SOCIAL_CLASS': np.random.choice(['Upper', 'Middle', 'Lower']),
        'HOUSE_TYPE': np.random.choice(['Own', 'Rent', 'Company Provided']),
        'ANY_INJURY': np.random.randint(0, 2),
        'NO_OF_FAMILY_MEMBERS': np.random.randint(1, 6),
        'CUSTOMER_EDUCATION_LEVEL': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
        'CLAIM_STATUS': np.random.choice(['Pending', 'Approved', 'Rejected'])
    }

def run_test_scenarios():
    """Run various test scenarios"""
    
    test_cases = [
        # Test Case 1: Normal legitimate claim
        {
            'description': 'Normal legitimate claim',
            'params': {
                'claim_amount': 5000,
                'premium_amount': 1000,
                'incident_severity': 'Minor Loss',
                'police_report': 1,
                'incident_hour': 14,
                'risk_segment': 'L',
                'days_since_policy': 180
            },
            'expected_fraud': False
        },
        
        # Test Case 2: Highly suspicious claim
        {
            'description': 'Highly suspicious claim',
            'params': {
                'claim_amount': 100000,
                'premium_amount': 1000,
                'incident_severity': 'Total Loss',
                'police_report': 0,
                'incident_hour': 2,
                'risk_segment': 'H',
                'days_since_policy': 5
            },
            'expected_fraud': True
        },
        
        # Test Case 3: Borderline case
        {
            'description': 'Borderline case',
            'params': {
                'claim_amount': 30000,
                'premium_amount': 5000,
                'incident_severity': 'Major Loss',
                'police_report': 1,
                'incident_hour': 20,
                'risk_segment': 'M',
                'days_since_policy': 45
            },
            'expected_fraud': False
        },
        
        # Test Case 4: Late night claim with missing police report
        {
            'description': 'Late night claim with missing police report',
            'params': {
                'claim_amount': 25000,
                'premium_amount': 3000,
                'incident_severity': 'Major Loss',
                'police_report': 0,
                'incident_hour': 3,
                'risk_segment': 'M',
                'days_since_policy': 90
            },
            'expected_fraud': True
        },
        
        # Test Case 5: Quick claim after policy start
        {
            'description': 'Quick claim after policy start',
            'params': {
                'claim_amount': 15000,
                'premium_amount': 2000,
                'incident_severity': 'Minor Loss',
                'police_report': 1,
                'incident_hour': 15,
                'risk_segment': 'L',
                'days_since_policy': 10
            },
            'expected_fraud': True
        }
    ]
    
    # Generate test data
    test_claims = []
    for case in test_cases:
        claim = generate_test_claim(**case['params'])
        test_claims.append(claim)
    
    # Convert to DataFrame
    df = pd.DataFrame(test_claims)
    
    # Process with fraud detection system
    df = create_fraud_flags(df)
    
    # Print results
    print("\nFraud Detection Test Results")
    print("===========================")
    
    for i, (case, row) in enumerate(zip(test_cases, df.itertuples())):
        print(f"\nTest Case {i+1}: {case['description']}")
        print(f"Expected Fraud: {case['expected_fraud']}")
        print(f"Detected Fraud: {bool(row.FRAUD_FLAG)}")
        print(f"Suspicious Score: {row.suspicious_score}")
        print(f"Claim Amount: ${row.CLAIM_AMOUNT}")
        print(f"Premium Amount: ${row.PREMIUM_AMOUNT}")
        print(f"Claim/Premium Ratio: {row.claim_premium_ratio:.2f}")
        print(f"Days Since Policy: {row.days_to_loss}")
        print("Match: ", "✓" if bool(row.FRAUD_FLAG) == case['expected_fraud'] else "✗")
        print("-" * 50)

def test_model_predictions():
    """Test the trained model's predictions"""
    
    # Load the trained model
    fraud_system = FraudDetectionSystem()
    if not fraud_system.load_model():
        print("No trained model found. Please train the model first.")
        return
    
    # Generate test cases
    test_cases = []
    for _ in range(10):
        # Generate random test cases
        claim_amount = np.random.randint(1000, 100000)
        premium_amount = np.random.randint(1000, 10000)
        incident_hour = np.random.randint(0, 24)
        days_since_policy = np.random.randint(1, 365)
        
        test_cases.append(generate_test_claim(
            claim_amount=claim_amount,
            premium_amount=premium_amount,
            incident_hour=incident_hour,
            days_since_policy=days_since_policy
        ))
    
    # Convert to DataFrame and preprocess
    df = pd.DataFrame(test_cases)
    df = create_fraud_flags(df)  # This will add days_to_loss and claim_premium_ratio
    
    # Get model predictions
    predictions = fraud_system.predict_fraud(df)
    
    # Print results
    print("\nModel Prediction Test Results")
    print("============================")
    for i, (row, pred, prob) in enumerate(zip(df.itertuples(), 
                                            predictions['predictions'],
                                            predictions['probabilities'])):
        print(f"\nTest Case {i+1}:")
        print(f"Claim Amount: ${row.CLAIM_AMOUNT}")
        print(f"Premium Amount: ${row.PREMIUM_AMOUNT}")
        print(f"Claim/Premium Ratio: {row.claim_premium_ratio:.2f}")
        print(f"Days Since Policy: {row.days_to_loss}")
        print(f"Incident Hour: {row.INCIDENT_HOUR_OF_THE_DAY}")
        print(f"Risk Segmentation: {row.RISK_SEGMENTATION}")
        print(f"Incident Severity: {row.INCIDENT_SEVERITY}")
        print(f"Rule-Based Fraud Flag: {bool(row.FRAUD_FLAG)}")
        print(f"Model Predicted Fraud: {bool(pred)}")
        print(f"Fraud Probability: {prob[1]:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    print("Running fraud detection tests...")
    print("\n1. Testing Rule-Based Detection")
    print("==============================")
    run_test_scenarios()
    
    print("\n2. Testing Model Predictions")
    print("===========================")
    test_model_predictions() 