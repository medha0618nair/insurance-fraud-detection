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

    def load_model(self, model_path='fraud_detection_model.joblib', preprocessor_path='fraud_detection_preprocessor.joblib'):
        """
        Load a trained model and preprocessor
        """
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

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