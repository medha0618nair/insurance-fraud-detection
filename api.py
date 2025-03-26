from flask import Flask, request, jsonify
from fraud_detection import detect_fraud
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_claim():
    try:
        data = request.get_json()
        
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
    app.run(debug=True) 