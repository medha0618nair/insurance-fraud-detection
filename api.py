from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from pdf_processor import process_pdf
import shutil
from typing import List
import json
import pandas as pd
from fraud_detection import FraudDetectionSystem

app = FastAPI(
    title="Insurance Claims Fraud Detection API",
    description="API for processing insurance claim PDFs and detecting potential fraud",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = "pdf_claims"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Initialize fraud detection system
fraud_system = FraudDetectionSystem()

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Insurance Claims Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/upload": "Upload PDF files for processing",
            "/process": "Process all uploaded PDFs",
            "/results": "Get processing results",
            "/train": "Train the fraud detection model",
            "/predict": "Predict fraud for new claims",
            "/analyze": "Analyze fraud indicators"
        }
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse(
            content={
                "message": f"File {file.filename} uploaded successfully",
                "filename": file.filename
            },
            status_code=201
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_pdfs():
    """Process all uploaded PDF files"""
    try:
        results = []
        for filename in os.listdir(UPLOAD_DIR):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(UPLOAD_DIR, filename)
                claim_info = process_pdf(pdf_path)
                if claim_info:
                    results.append(claim_info)
        
        # Save results to CSV
        if results:
            df = pd.DataFrame(results)
            df.to_csv('processed_claims.csv', index=False)
            
            # Add fraud flag based on heuristics
            df['FRAUD_FLAG'] = df.apply(lambda row: int(
                row['claim_amount'] > 30000 or
                row['number_of_documents'] < 3 or
                row['past_claims_count'] > 2 or
                row['time_to_submit_claim'] < 3
            ), axis=1)
            
            df.to_csv('processed_claims.csv', index=False)
        
        return {
            "message": f"Processed {len(results)} files successfully",
            "processed_files": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def get_results():
    """Get the processing results"""
    try:
        if not os.path.exists('processed_claims.csv'):
            raise HTTPException(status_code=404, detail="No results found")
        
        df = pd.read_csv('processed_claims.csv')
        return {
            "total_records": len(df),
            "data": df.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model():
    """Train the fraud detection model"""
    try:
        if not os.path.exists('processed_claims.csv'):
            raise HTTPException(status_code=404, detail="No training data found")
        
        # Load and prepare data
        df = fraud_system.load_data('processed_claims.csv')
        X, y = fraud_system.prepare_features(df)
        
        if y is None:
            raise HTTPException(status_code=400, detail="No fraud labels found in the data")
        
        # Train the model
        metrics = fraud_system.train_model(X, y)
        
        # Save the model
        fraud_system.save_model()
        
        return {
            "message": "Model trained successfully",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_fraud():
    """Predict fraud for processed claims"""
    try:
        if not fraud_system.is_trained:
            raise HTTPException(status_code=400, detail="Model is not trained")
        
        if not os.path.exists('processed_claims.csv'):
            raise HTTPException(status_code=404, detail="No claims data found")
        
        # Load data and make predictions
        df = fraud_system.load_data('processed_claims.csv')
        predictions = fraud_system.predict_fraud(df)
        
        return {
            "message": "Predictions completed successfully",
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze")
async def analyze_fraud():
    """Analyze fraud indicators in the processed data"""
    try:
        if not os.path.exists('processed_claims.csv'):
            raise HTTPException(status_code=404, detail="No data found for analysis")
        
        # Load data and analyze
        df = fraud_system.load_data('processed_claims.csv')
        insights = fraud_system.analyze_fraud_indicators(df)
        
        if insights is None:
            raise HTTPException(status_code=400, detail="No fraud labels found in the data")
        
        return {
            "message": "Analysis completed successfully",
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup")
async def cleanup():
    """Clean up uploaded files and results"""
    try:
        # Remove all files in upload directory
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Remove results file if exists
        if os.path.exists('processed_claims.csv'):
            os.remove('processed_claims.csv')
        
        return {"message": "Cleanup completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 