import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime

class FraudDetectionSystem:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.is_trained = False

    def load_data(self, file_path):
        """
        Load the insurance dataset and perform initial preprocessing
        """
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Data Cleaning and Preprocessing
        # Convert date columns to datetime
        date_columns = ['TXN_DATE_TIME', 'POLICY_EFF_DT', 'LOSS_DT', 'REPORT_DT']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Feature Engineering
        # Calculate days between policy effective date and loss date
        if 'POLICY_EFF_DT' in df.columns and 'LOSS_DT' in df.columns:
            df['days_to_loss'] = (df['LOSS_DT'] - df['POLICY_EFF_DT']).dt.days
        
        # Calculate claim amount to premium ratio
        if 'CLAIM_AMOUNT' in df.columns and 'PREMIUM_AMOUNT' in df.columns:
            df['claim_premium_ratio'] = df['CLAIM_AMOUNT'] / df['PREMIUM_AMOUNT']
        
        return df

    def prepare_features(self, df):
        """
        Prepare features for fraud detection model
        """
        # Select features for the model
        categorical_features = [
            'INSURANCE_TYPE', 'MARITAL_STATUS', 'EMPLOYMENT_STATUS', 
            'RISK_SEGMENTATION', 'HOUSE_TYPE', 'SOCIAL_CLASS',
            'CUSTOMER_EDUCATION_LEVEL', 'CLAIM_STATUS', 'INCIDENT_SEVERITY'
        ]
        
        numerical_features = [
            'PREMIUM_AMOUNT', 'CLAIM_AMOUNT', 'AGE', 'TENURE', 
            'NO_OF_FAMILY_MEMBERS', 'days_to_loss', 'claim_premium_ratio',
            'INCIDENT_HOUR_OF_THE_DAY', 'ANY_INJURY'
        ]
        
        # Filter available features
        available_categorical = [col for col in categorical_features if col in df.columns]
        available_numerical = [col for col in numerical_features if col in df.columns]
        
        # Prepare the feature matrix and target variable
        X = df[available_categorical + available_numerical]
        y = df['FRAUD_FLAG'] if 'FRAUD_FLAG' in df.columns else None
        
        return X, y

    def train_model(self, X, y):
        """
        Train the fraud detection model
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocessing for numerical and categorical data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
                ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object']).columns)
            ])
        
        # Create a pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced', 
                random_state=42
            ))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics

    def predict_fraud(self, new_data):
        """
        Predict fraud for new insurance claims
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Please train the model first.")
        
        # Prepare features
        X, _ = self.prepare_features(new_data)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }

    def save_model(self, model_path='fraud_detection_model.joblib', preprocessor_path='fraud_detection_preprocessor.joblib'):
        """
        Save the trained model and preprocessor
        """
        if self.is_trained:
            joblib.dump(self.model, model_path)
            joblib.dump(self.preprocessor, preprocessor_path)
            return True
        return False

    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            self.model = joblib.load('fraud_detection_model.joblib')
            self.preprocessor = joblib.load('fraud_detection_preprocessor.joblib')
            print("Model and preprocessor loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model files: {str(e)}")
            print("Falling back to rule-based detection only")
            self.model = None
            self.preprocessor = None

    def analyze_fraud_indicators(self, df):
        """
        Analyze fraud indicators in the dataset
        """
        if 'FRAUD_FLAG' not in df.columns:
            return None
        
        fraud_cases = df[df['FRAUD_FLAG'] == 1]
        insights = {
            'total_fraud_cases': len(fraud_cases),
            'fraud_rate': len(fraud_cases) / len(df),
            'indicators': {}
        }
        
        # Analyze fraud indicators
        indicator_columns = [
            'INSURANCE_TYPE', 'RISK_SEGMENTATION', 
            'INCIDENT_SEVERITY', 'EMPLOYMENT_STATUS'
        ]
        
        for col in indicator_columns:
            if col in df.columns:
                insights['indicators'][col] = fraud_cases[col].value_counts(normalize=True).to_dict()
        
        return insights 

def create_fraud_flags(df):
    """Create fraud flags based on business rules"""
    df = df.copy()
    
    # Calculate suspicious score
    df['suspicious_score'] = 0
    
    # High claim amount
    if 'claim_amount' in df.columns:
        df.loc[df['claim_amount'] > 50000, 'suspicious_score'] += 1
    
    # High claim to premium ratio
    if 'premium_amount' in df.columns and 'claim_amount' in df.columns:
        df['claim_premium_ratio'] = df['claim_amount'] / df['premium_amount']
        df.loc[df['claim_premium_ratio'] > 10, 'suspicious_score'] += 1
    
    # Quick claim after policy start
    if 'days_to_loss' in df.columns:
        df.loc[df['days_to_loss'] < 30, 'suspicious_score'] += 1
    
    # Late night incidents
    if 'incident_hour' in df.columns:
        df.loc[df['incident_hour'].between(23, 4), 'suspicious_score'] += 1
    
    # Missing police report
    if 'police_report_available' in df.columns:
        df.loc[~df['police_report_available'], 'suspicious_score'] += 1
    
    # Risk segmentation
    if 'risk_segmentation' in df.columns:
        df.loc[df['risk_segmentation'] == 'H', 'suspicious_score'] += 1
    
    # Set fraud flag based on suspicious score
    df['fraud_detected'] = df['suspicious_score'] >= 2
    
    # Add risk level
    df['risk_level'] = pd.cut(df['suspicious_score'], 
                             bins=[-np.inf, 1, 2, np.inf],
                             labels=['Low', 'Medium', 'High'])
    
    return df

def detect_fraud(data):
    """
    Detect fraud in insurance claims
    
    Parameters:
    data (pd.DataFrame): DataFrame containing claim information
    
    Returns:
    pd.DataFrame: DataFrame with fraud detection results
    """
    # Convert to DataFrame if single claim
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame([data])
    
    # Ensure all column names are lowercase
    data.columns = data.columns.str.lower()
    
    # Create fraud flags
    results = create_fraud_flags(data)
    
    # Load the model if available
    try:
        model = joblib.load('fraud_detection_model.joblib')
        preprocessor = joblib.load('fraud_detection_preprocessor.joblib')
        
        # Prepare features
        X = results[['claim_amount', 'premium_amount', 'days_to_loss', 
                    'incident_hour', 'risk_segmentation', 'incident_severity']]
        
        # Make predictions
        probabilities = model.predict_proba(X)
        results['fraud_probability'] = probabilities[:, 1]
        
    except Exception as e:
        # If model is not available, use only rule-based detection
        results['fraud_probability'] = results['suspicious_score'] / 5.0
    
    return results 