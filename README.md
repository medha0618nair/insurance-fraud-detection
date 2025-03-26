# Insurance Fraud Detection & False Claims System

A comprehensive system for detecting fraudulent insurance claims and false claims using machine learning and rule-based approaches.

## Features

- Rule-based fraud detection
- Machine learning model for fraud prediction
- API endpoints for real-time claim analysis
- Comprehensive test suite
- Detailed fraud analysis reports

## Installation

1. Clone the repository:

```bash
git clone https://github.com/medha0618nair/insurance-fraud-detection.git
cd insurance-fraud-detection
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the API Server

```bash
python api.py
```

### Running Tests

```bash
python test_fraud_detection.py
```

### Processing Data

```bash
python process_insurance_data.py
```

## API Endpoints

### Single Claim Analysis

- **Endpoint**: `POST /analyze`
- **Description**: Analyze a single insurance claim for fraud
- **Request Body**:

```json
{
  "claim_amount": 5000,
  "premium_amount": 1000,
  "days_to_loss": 175,
  "incident_hour": 16,
  "risk_segmentation": "L",
  "incident_severity": "Minor Loss",
  "police_report_available": true
}
```

- **Response**:

```json
{
  "status": "success",
  "fraud_detected": false,
  "fraud_probability": 0.23,
  "suspicious_score": 0,
  "risk_level": "Low"
}
```

### Batch Analysis

- **Endpoint**: `POST /batch_analyze`
- **Description**: Analyze multiple insurance claims for fraud
- **Request Body**: Array of claim objects (same structure as single claim)
- **Response**: Array of analysis results for each claim

## Project Structure

- `api.py`: Main API server
- `process_insurance_data.py`: Data processing and model training
- `test_fraud_detection.py`: Test suite
- `requirements.txt`: Project dependencies

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
