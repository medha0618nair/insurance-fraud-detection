from flask import Flask, request, jsonify
from flask_cors import CORS
from fraud_detection import detect_fraud
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Basic security middleware
@app.before_request
def validate_request():
    # Add your security checks here
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_claim():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['claim_amount', 'premium_amount', 'days_to_loss', 
                         'incident_hour', 'risk_segmentation', 'incident_severity']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        
        # Detect fraud
        result = detect_fraud(df)
        
        return jsonify({
            'status': 'success',
            'fraud_detected': bool(result['fraud_detected'].iloc[0]),
            'fraud_probability': float(result['fraud_probability'].iloc[0]),
            'suspicious_score': int(result['suspicious_score'].iloc[0]),
            'risk_level': result['risk_level'].iloc[0]
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({
                'status': 'error',
                'message': 'Request body must be an array of claims'
            }), 400
        
        # Convert input data to DataFrame
        df = pd.DataFrame(data)
        
        # Detect fraud for all claims
        results = detect_fraud(df)
        
        # Convert results to list of dictionaries
        response_data = []
        for idx, row in results.iterrows():
            response_data.append({
                'claim_id': idx,
                'fraud_detected': bool(row['fraud_detected']),
                'fraud_probability': float(row['fraud_probability']),
                'suspicious_score': int(row['suspicious_score']),
                'risk_level': row['risk_level']
            })
        
        return jsonify({
            'status': 'success',
            'results': response_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port) 