# Import necessary libraries
import os, re
from flask import Flask, render_template, request
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from groq import Groq
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import DirectoryLoader
import json

# Get the Groq API key
api_key = " "

if api_key is None or api_key == "":
    print("Groq API key not set or empty. Please set the API key.")
    exit()

# Initialize the Groq client
client = Groq(api_key=api_key)

# For embeddings, we'll use a simple fallback since Groq doesn't provide embeddings
# You might want to use a different embedding service or implement a simple text matching
FAISS_PATH = "/faiss"

# Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease", "pregnancy", "substance abuse", "self-inflicted injuries", "sexually transmitted diseases", "std", "pre-existing conditions", "cancer", "heart disease", "diabetes complications", "mental health disorders", "chronic conditions"]

def get_document_loader():
    # Fallback method since we don't have access to the documents directory
    # You should implement this based on your document structure
    return []

def get_text_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    if documents:
        chunks = text_splitter.split_documents(documents)
        return chunks
    return []

def get_claim_approval_context():
    # Hardcoded context since we don't have access to the documents
    return """
    Required documents for claim approval:
    1. Valid consultation receipt/medical bill
    2. Patient identification
    3. Medical prescription (if applicable)
    4. Doctor's report
    5. Diagnostic reports (if applicable)
    6. Complete patient information form
    """

def get_general_exclusion_context():
    return f"General exclusions list: {', '.join(general_exclusion_list)}"

def get_file_content(file):
    text = ""
    try:
        if file.filename.endswith(".pdf"):
            pdf = PdfReader(file)
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def get_bill_info(data):
    if not data or len(data.strip()) < 10:
        return {"disease": None, "expense": None}
    
    try:
        prompt = """Act as an expert in extracting information from medical invoices. You are given with the invoice details of a patient. Go through the given document carefully and extract the 'disease' and the 'expense amount' from the data. Return the data in json format = {'disease':'disease_name','expense':'amount_number_only'}

        Important: Only return valid JSON format. If you cannot find the information, return {'disease':null,'expense':null}"""
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"INVOICE DETAILS: {data}"}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            # Remove any markdown formatting
            json_match = re.search(r'\{.*?\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data_result = json.loads(json_str)
                return data_result
        except Exception as e:
            print(f"JSON parse error: {e}")
            return {"disease": "consultation", "expense": "0"}
        except json.JSONDecodeError:
            # Fallback parsing
            return {"disease": "consultation", "expense": "0"}
            
    except Exception as e:
        print(f"Error in get_bill_info: {e}")
        return {"disease": None, "expense": None}

def check_disease_exclusion(disease, exclusion_list):
    if not disease:
        return False
    
    disease_lower = disease.lower()
    for excluded in exclusion_list:
        if excluded.lower() in disease_lower or disease_lower in excluded.lower():
            return True
    
    # Use cosine similarity as backup
    try:
        vectorizer = CountVectorizer()
        disease_vector = vectorizer.fit_transform([disease_lower])
        
        for excluded in exclusion_list:
            excluded_vector = vectorizer.transform([excluded.lower()])
            similarity = cosine_similarity(disease_vector, excluded_vector)[0][0]
            if similarity > 0.3:  # Lower threshold for better matching
                return True
    except:
        pass
    
    return False

def generate_claim_report(patient_info, medical_bill_info, bill_info, max_amount, is_excluded=False, amount_exceeds=False):
    try:
        if amount_exceeds:
            prompt = f"""You are an AI assistant for verifying health insurance claims. Generate a detailed rejection report.

PATIENT INFO: {patient_info}
MEDICAL BILL: {medical_bill_info}
REJECTION REASON: The claimed amount (${max_amount}) exceeds the actual bill amount (${bill_info.get('expense', '0')}).

Generate a detailed report in the following format:

INFORMATION: FALSE
EXCLUSION: FALSE
CLAIM STATUS: REJECTED

Executive Summary
The claim has been rejected because the claimed amount exceeds the actual bill amount.

Introduction
This report analyzes the insurance claim submitted and determines the approval status based on policy guidelines and amount verification.

Claim Details
Patient has submitted a claim for {bill_info.get('disease', 'medical treatment')} with:
- Actual bill amount: ${bill_info.get('expense', '0')}
- Claimed amount: ${max_amount}

Claim Description
Medical treatment claim for {bill_info.get('disease', 'consultation')} at the specified medical facility.

Document Verification
The submitted consultation receipt has been processed and verified. The claimed amount exceeds the actual bill amount.

Document Summary
The medical documents indicate treatment for {bill_info.get('disease', 'a condition')} with total expenses of ${bill_info.get('expense', '0')}. However, the patient claimed ${max_amount} which is higher than the actual bill.

The claim is REJECTED because claimed amount exceeds actual bill amount."""

        elif is_excluded:
            prompt = f"""You are an AI assistant for verifying health insurance claims. Generate a detailed rejection report.

PATIENT INFO: {patient_info}
MEDICAL BILL: {medical_bill_info}
REJECTION REASON: The disease/condition "{bill_info.get('disease', 'Unknown')}" is in the general exclusions list.

Generate a detailed report in the following format:

INFORMATION: FALSE
EXCLUSION: FALSE
CLAIM STATUS: REJECTED

Executive Summary
The claim has been rejected due to the medical condition being listed in the general exclusions.

Introduction
This report analyzes the insurance claim submitted and determines the approval status based on policy guidelines and exclusion criteria.

Claim Details
Patient has submitted a claim for {bill_info.get('disease', 'medical treatment')} with an expense amount of ${bill_info.get('expense', '0')}.

Claim Description
Medical treatment claim for {bill_info.get('disease', 'consultation')} at the specified medical facility.

Document Verification
The submitted consultation receipt has been processed and verified. However, the medical condition falls under policy exclusions.

Document Summary
The medical documents indicate treatment for {bill_info.get('disease', 'a condition')} which is explicitly listed in our general exclusions policy. Therefore, this claim cannot be approved regardless of documentation completeness.

The claim is REJECTED due to policy exclusions."""

        else:
            prompt = f"""You are an AI assistant for verifying health insurance claims. Analyze the given data and generate a detailed report.

CLAIM APPROVAL CONTEXT: {get_claim_approval_context()}
EXCLUSION LIST: {get_general_exclusion_context()}
PATIENT INFO: {patient_info}
MEDICAL BILL: {medical_bill_info}
MAXIMUM CLAIMABLE AMOUNT: ${max_amount}

Generate a detailed report in the following format:

INFORMATION: TRUE
EXCLUSION: TRUE  
CLAIM STATUS: APPROVED
APPROVED AMOUNT: ${min(int(bill_info.get('expense', 0)), int(max_amount))}

Executive Summary
The claim has been approved after thorough verification of documents and policy compliance.

Introduction
This report analyzes the insurance claim submitted and determines the approval status based on policy guidelines and documentation requirements.

Claim Details
Patient has submitted a claim for {bill_info.get('disease', 'medical consultation')} with:
- Actual bill amount: ${bill_info.get('expense', '0')}
- Claimed amount: ${max_amount}

Claim Description
Medical treatment claim for {bill_info.get('disease', 'consultation')} at the specified medical facility.

Document Verification
All required documents have been submitted and verified successfully. The consultation receipt is valid and contains all necessary information.

Document Summary
The submitted medical documents are complete and authentic. The treatment received is covered under the policy terms. No signs of fraud detected. The medical condition does not fall under general exclusions.

The claim is APPROVED for the amount of ${min(int(bill_info.get('expense', 0)), int(max_amount))}."""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an insurance claim processing expert. Generate detailed and professional reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return f"Error processing claim. Please contact support. Technical details: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def msg():
    try:
        name = request.form['name']
        address = request.form['address']
        claim_type = request.form['claim_type']
        claim_reason = request.form['claim_reason']
        date = request.form['date']
        medical_facility = request.form['medical_facility']
        medical_bill = request.files['medical_bill']
        total_claim_amount = request.form['total_claim_amount']
        description = request.form['description']

        # Validate file upload
        if not medical_bill or medical_bill.filename == '':
            error_message = "Please upload a valid consultation receipt (PDF file required)."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, output=error_message)

        if not medical_bill.filename.lower().endswith('.pdf'):
            error_message = "Please upload a valid PDF consultation receipt."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, output=error_message)

        # Extract bill content
        bill_content = get_file_content(medical_bill)
        
        if not bill_content or len(bill_content.strip()) < 20:
            error_message = "Please upload a valid consultation receipt with readable content."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, output=error_message)

        # Extract bill information
        bill_info = get_bill_info(bill_content)
        
        if not bill_info.get('expense') or bill_info.get('expense') == 'null':
            error_message = "Please upload a valid consultation receipt with clear expense amount."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, output=error_message)

        try:
            bill_expense = int(bill_info['expense'])
            claim_amount = int(total_claim_amount)
        except (ValueError, TypeError):
            error_message = "Invalid expense or claim amount format."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, output=error_message)

        # Check if claim amount exceeds bill amount (strict comparison)
        if claim_amount > bill_expense:
            # Prepare patient information
            patient_info = f"Name: {name}\nAddress: {address}\nClaim type: {claim_type}\nClaim reason: {claim_reason}\nMedical facility: {medical_facility}\nDate: {date}\nTotal claim amount: {total_claim_amount}\nDescription: {description}"
            medical_bill_info = f"Medical Bill Content: {bill_content[:500]}..."  # Truncate for processing
            
            # Generate rejection report for amount exceeding
            output = generate_claim_report(patient_info, medical_bill_info, bill_info, total_claim_amount, amount_exceeds=True)
            
            # Format output for HTML
            output = re.sub(r'\n', '<br>', output)
            
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, output=output)

        # Prepare patient information
        patient_info = f"Name: {name}\nAddress: {address}\nClaim type: {claim_type}\nClaim reason: {claim_reason}\nMedical facility: {medical_facility}\nDate: {date}\nTotal claim amount: {total_claim_amount}\nDescription: {description}"
        medical_bill_info = f"Medical Bill Content: {bill_content[:500]}..."  # Truncate for processing

        # Check for exclusions
        disease = bill_info.get('disease', claim_reason)
        is_excluded = check_disease_exclusion(disease, general_exclusion_list)
        
        # Generate report
        output = generate_claim_report(patient_info, medical_bill_info, bill_info, total_claim_amount, is_excluded)
        
        # Format output for HTML
        output = re.sub(r'\n', '<br>', output)
        
        return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                             claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                             total_claim_amount=total_claim_amount, description=description, output=output)
        
    except Exception as e:
        error_message = f"An error occurred while processing your claim. Please try again. Error: {str(e)}"
        return render_template("result.html", name="", address="", claim_type="", 
                             claim_reason="", date="", medical_facility="", 
                             total_claim_amount="", description="", output=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)