import PyPDF2
import pandas as pd
import re
from datetime import datetime
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            return text
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        return None

def extract_claim_info(text):
    """Extract relevant information from the PDF text."""
    # Initialize dictionary to store extracted information
    claim_info = {
        'claim_number': None,
        'claim_date': None,
        'claim_amount': None,
        'claim_type': None,
        'customer_name': None,
        'policy_number': None,
        'claim_status': None
    }
    
    # Common patterns for extracting information
    patterns = {
        'claim_number': r'Claim\s*Number\s*[:#]?\s*([A-Z0-9-]+)',
        'claim_date': r'Date\s*[:#]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        'claim_amount': r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        'claim_type': r'Type\s*[:#]?\s*([A-Za-z\s]+)',
        'customer_name': r'Name\s*[:#]?\s*([A-Za-z\s]+)',
        'policy_number': r'Policy\s*Number\s*[:#]?\s*([A-Z0-9-]+)',
        'claim_status': r'Status\s*[:#]?\s*([A-Za-z\s]+)'
    }
    
    # Extract information using regex patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            claim_info[key] = match.group(1).strip()
    
    return claim_info

def process_pdf(pdf_path):
    """Process a PDF file and extract claim information."""
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None
    
    # Extract claim information
    claim_info = extract_claim_info(text)
    
    # Add file information
    claim_info['file_name'] = os.path.basename(pdf_path)
    claim_info['processed_date'] = datetime.now().strftime('%Y-%m-%d')
    
    return claim_info

def main():
    # Create a directory for PDF files if it doesn't exist
    pdf_dir = 'pdf_claims'
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    print("PDF Insurance Claims Processor")
    print("=============================")
    print(f"Please place your PDF files in the '{pdf_dir}' directory")
    print("The script will process all PDF files in this directory")
    
    # Process all PDF files in the directory
    results = []
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"\nProcessing: {filename}")
            
            claim_info = process_pdf(pdf_path)
            if claim_info:
                results.append(claim_info)
                print("Successfully processed!")
            else:
                print("Failed to process the file.")
    
    # Convert results to DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        output_file = 'processed_claims.csv'
        df.to_csv(output_file, index=False)
        print(f"\nProcessed {len(results)} files successfully!")
        print(f"Results saved to: {output_file}")
        print("\nSample of processed data:")
        print(df.head())
    else:
        print("\nNo PDF files were successfully processed.")

if __name__ == "__main__":
    main() 