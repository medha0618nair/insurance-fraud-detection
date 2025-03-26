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
        if not data:
            return jsonify({"error": "No data provided"}), 400

        result = detect_fraud(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({"error": "No data provided or invalid format"}), 400

        results = []
        for claim in data:
            result = detect_fraud(claim)
            results.append(result)

        return jsonify({
            "results": results,
            "total_claims": len(results),
            "fraudulent_claims": sum(1 for r in results if r.get('fraud_detected', False)),
            "legitimate_claims": sum(1 for r in results if not r.get('fraud_detected', False))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port) 