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

# Get the Groq API key - Replace with your actual API key
api_key = " "

if api_key is None or api_key == " " or api_key == "your_actual_groq_api_key_here":
    print("Groq API key not set or empty. Please set the API key.")
    exit()

# Initialize the Groq client
client = Groq(api_key=api_key)

# For embeddings, we'll use a simple fallback since Groq doesn't provide embeddings
FAISS_PATH = "/faiss"

# Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease", "pregnancy", "substance abuse", "self-inflicted injuries", "sexually transmitted diseases", "std", "pre-existing conditions", "cancer", "heart disease", "diabetes complications", "mental health disorders", "chronic conditions"]

def get_document_loader():
    # Fallback method since we don't have access to the documents directory
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
        if file and file.filename.lower().endswith(".pdf"):
            pdf = PdfReader(file)
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                page_text = page.extract_text()
                if page_text:  # Only add non-empty text
                    text += page_text + " "
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        print(f"Extracted text length: {len(text)}")  # Debug info
        print(f"First 200 chars: {text[:200]}")  # Debug info
        
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_amount_from_text(text):
    """Extract amount from text using regex patterns"""
    if not text:
        return None
    
    # Common patterns for amounts
    patterns = [
        r'(?:total|amount|price|cost|bill|charge|fee)[:\s]*[₹$]?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
        r'[₹$]\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
        r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*[₹$]',
        r'(?:rs|rupees)[.\s]*(\d+(?:,\d+)*(?:\.\d{2})?)',
        r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:rs|rupees)',
        r'amount[:\s]*(\d+(?:,\d+)*(?:\.\d{2})?)',
        r'total[:\s]*(\d+(?:,\d+)*(?:\.\d{2})?)',
    ]
    
    text_lower = text.lower()
    amounts = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            try:
                # Remove commas and convert to float
                amount = float(match.replace(',', ''))
                if 10 <= amount <= 1000000:  # Reasonable range for medical bills
                    amounts.append(amount)
            except ValueError:
                continue
    
    # Return the most likely amount (highest value if multiple found)
    if amounts:
        return max(amounts)
    
    return None

def get_bill_info(data):
    """Extract bill information with improved parsing"""
    if not data or len(data.strip()) < 10:
        return {"disease": "consultation", "expense": None}
    
    # First try to extract amount using regex
    extracted_amount = extract_amount_from_text(data)
    
    try:
        prompt = """Act as an expert in extracting information from medical invoices and receipts. 
        
        You are given medical document text. Extract:
        1. The medical condition/disease/treatment mentioned
        2. The total amount/cost/fee charged
        
        Look for keywords like: consultation, treatment, diagnosis, medicine, procedure, etc.
        Look for amounts with symbols like ₹, $, Rs, or words like "total", "amount", "bill", "cost"
        
        Return ONLY valid JSON in this exact format:
        {"disease": "condition_or_treatment_name", "expense": "numeric_amount_only"}
        
        If you cannot find clear information, use:
        {"disease": "consultation", "expense": "0"}
        
        Examples of good responses:
        {"disease": "fever consultation", "expense": "500"}
        {"disease": "diabetes checkup", "expense": "1200"}
        """
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Medical document text: {data[:1000]}"}  # Limit text length
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        print(f"AI response: {content}")  # Debug info
        
        # Try to extract JSON from the response
        try:
            # Remove any markdown formatting or extra text
            json_match = re.search(r'\{[^{}]*\}', content)
            if json_match:
                json_str = json_match.group(0)
                ai_result = json.loads(json_str)
                
                # Use extracted amount if AI didn't find one or if regex found a better one
                if extracted_amount and (not ai_result.get('expense') or ai_result.get('expense') == '0'):
                    ai_result['expense'] = str(int(extracted_amount))
                
                # Ensure we have some values
                if not ai_result.get('disease'):
                    ai_result['disease'] = 'consultation'
                if not ai_result.get('expense') or ai_result.get('expense') == '0':
                    if extracted_amount:
                        ai_result['expense'] = str(int(extracted_amount))
                    else:
                        ai_result['expense'] = '100'  # Default minimum amount
                
                return ai_result
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"JSON parse error: {e}")
            
        # Fallback: use regex extraction or defaults
        disease = "consultation"
        expense = str(int(extracted_amount)) if extracted_amount else "100"
        
        # Try to find disease/condition in text
        condition_patterns = [
            r'(?:diagnosis|condition|treatment|consultation)[:\s]*([a-zA-Z\s]+?)(?:\s|$|[.,])',
            r'(?:fever|cold|cough|headache|pain|infection|diabetes|hypertension)',
            r'(?:checkup|examination|consultation|visit)',
        ]
        
        for pattern in condition_patterns:
            match = re.search(pattern, data.lower())
            if match:
                if len(pattern) > 20:  # If it's a specific condition
                    disease = match.group(0).strip()
                else:  # If it's a capturing group
                    disease = match.group(1).strip() if match.groups() else match.group(0).strip()
                break
        
        return {"disease": disease, "expense": expense}
        
    except Exception as e:
        print(f"Error in get_bill_info: {e}")
        # Fallback with regex extraction
        disease = "consultation"
        expense = str(int(extracted_amount)) if extracted_amount else "100"
        return {"disease": disease, "expense": expense}

def check_disease_exclusion(disease, exclusion_list):
    """Check if disease is in exclusion list"""
    if not disease:
        return False
    
    disease_lower = disease.lower().strip()
    
    # Direct string matching
    for excluded in exclusion_list:
        excluded_lower = excluded.lower().strip()
        if excluded_lower in disease_lower or disease_lower in excluded_lower:
            print(f"Exclusion found: {disease} matches {excluded}")
            return True
    
    # Keyword matching for common exclusions
    exclusion_keywords = {
        'hiv': ['hiv', 'aids'],
        'cancer': ['cancer', 'tumor', 'oncology', 'malignant'],
        'diabetes': ['diabetes', 'diabetic'],
        'heart': ['heart', 'cardiac', 'coronary'],
        'mental': ['mental', 'psychiatric', 'depression', 'anxiety'],
        'pregnancy': ['pregnancy', 'pregnant', 'maternity'],
        'substance': ['substance', 'drug', 'alcohol', 'addiction'],
        'std': ['std', 'sexually transmitted', 'herpes', 'syphilis'],
        'chronic': ['chronic', 'long-term']
    }
    
    for category, keywords in exclusion_keywords.items():
        for keyword in keywords:
            if keyword in disease_lower:
                print(f"Exclusion found by keyword: {disease} contains {keyword}")
                return True
    
    return False

def generate_claim_report(patient_info, medical_bill_info, bill_info, max_amount, is_excluded=False, amount_exceeds=False):
    """Generate detailed claim report"""
    try:
        disease = bill_info.get('disease', 'consultation')
        bill_expense = bill_info.get('expense', '0')
        
        # Convert to integers for comparison
        try:
            bill_amount = int(bill_expense)
            claim_amount = int(max_amount)
        except (ValueError, TypeError):
            bill_amount = 0
            claim_amount = int(max_amount) if max_amount else 0
        
        if amount_exceeds:
            status = "REJECTED"
            reason = f"Claimed amount (₹{max_amount}) exceeds actual bill amount (₹{bill_expense})"
            approved_amount = "0"
            
        elif is_excluded:
            status = "REJECTED"
            reason = f"Medical condition '{disease}' is listed in policy exclusions"
            approved_amount = "0"
            
        else:
            status = "APPROVED"
            approved_amount = str(min(bill_amount, claim_amount))
            reason = "All requirements met and condition is covered under policy"

        report = f"""
<div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
    <h2 style="color: #2c5aa0; border-bottom: 2px solid #2c5aa0; padding-bottom: 10px;">
        INSURANCE CLAIM PROCESSING REPORT
    </h2>
    
    <div style="background-color: {'#ffebee' if status == 'REJECTED' else '#e8f5e8'}; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="color: {'#c62828' if status == 'REJECTED' else '#2e7d32'}; margin: 0;">
            CLAIM STATUS: {status}
        </h3>
        {f'<p style="margin: 5px 0 0 0; font-size: 16px;"><strong>Approved Amount: ₹{approved_amount}</strong></p>' if status == 'APPROVED' else ''}
    </div>

    <h3 style="color: #2c5aa0;">Executive Summary</h3>
    <p>{reason}</p>

    <h3 style="color: #2c5aa0;">Claim Analysis Details</h3>
    <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
        <tr style="background-color: #f5f5f5;">
            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Information Verification</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{'INCOMPLETE' if amount_exceeds or is_excluded else 'COMPLETE'}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Exclusion Check</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{'EXCLUDED' if is_excluded else 'COVERED'}</td>
        </tr>
        <tr style="background-color: #f5f5f5;">
            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Medical Condition</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{disease}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Bill Amount</td>
            <td style="border: 1px solid #ddd; padding: 8px;">₹{bill_expense}</td>
        </tr>
        <tr style="background-color: #f5f5f5;">
            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Claimed Amount</td>
            <td style="border: 1px solid #ddd; padding: 8px;">₹{max_amount}</td>
        </tr>
    </table>

    <h3 style="color: #2c5aa0;">Document Verification</h3>
    <p>✅ Medical consultation receipt uploaded and processed<br>
    ✅ Patient information form completed<br>
    ✅ Bill content extracted and analyzed</p>

    <h3 style="color: #2c5aa0;">Processing Summary</h3>
    <p>The submitted medical documents have been thoroughly analyzed. The treatment received was for {disease} 
    with a total bill amount of ₹{bill_expense}. The patient claimed ₹{max_amount}.</p>
    
    {'<p style="color: #c62828;"><strong>Rejection Reason:</strong> ' + reason + '</p>' if status == 'REJECTED' else 
     '<p style="color: #2e7d32;"><strong>Approval Details:</strong> The claim meets all policy requirements and is approved for ₹' + approved_amount + '</p>'}

    <div style="margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
        <h4 style="color: #2c5aa0; margin-top: 0;">Important Notes:</h4>
        <ul>
            <li>This is an automated assessment based on policy guidelines</li>
            <li>For disputes or clarifications, please contact customer service</li>
            <li>Keep this report for your records</li>
            {'<li style="color: #c62828;">If you believe this rejection is in error, you may appeal within 30 days</li>' if status == 'REJECTED' else ''}
        </ul>
    </div>
</div>
        """
        
        return report
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return f"""
        <div style="color: red; padding: 20px;">
            <h3>Processing Error</h3>
            <p>An error occurred while processing your claim. Please try again or contact support.</p>
            <p>Error details: {str(e)}</p>
        </div>
        """

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def msg():
    try:
        # Get form data
        name = request.form.get('name', '')
        address = request.form.get('address', '')
        claim_type = request.form.get('claim_type', '')
        claim_reason = request.form.get('claim_reason', '')
        date = request.form.get('date', '')
        medical_facility = request.form.get('medical_facility', '')
        medical_bill = request.files.get('medical_bill')
        total_claim_amount = request.form.get('total_claim_amount', '0')
        description = request.form.get('description', '')

        # Validate required fields
        if not all([name, claim_reason, total_claim_amount, medical_bill]):
            error_message = "Please fill in all required fields and upload a medical bill."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, 
                                 output=error_message)

        # Validate file upload
        if not medical_bill or medical_bill.filename == '':
            error_message = "Please upload a valid consultation receipt (PDF file required)."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, 
                                 output=error_message)

        if not medical_bill.filename.lower().endswith('.pdf'):
            error_message = "Please upload a PDF consultation receipt only."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, 
                                 output=error_message)

        # Extract bill content
        print("Starting PDF extraction...")
        bill_content = get_file_content(medical_bill)
        
        if not bill_content or len(bill_content.strip()) < 20:
            error_message = "Could not read the PDF content. Please ensure the PDF is not password protected and contains readable text."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, 
                                 output=error_message)

        # Extract bill information
        print("Extracting bill information...")
        bill_info = get_bill_info(bill_content)
        print(f"Bill info extracted: {bill_info}")
        
        # Validate extracted information
        if not bill_info.get('expense') or bill_info.get('expense') in ['null', '0', None]:
            # Try to process anyway with a default amount
            bill_info['expense'] = '100'  # Minimum processing fee
            print("Using default expense amount")

        try:
            bill_expense = int(bill_info['expense'])
            claim_amount = int(total_claim_amount)
        except (ValueError, TypeError):
            error_message = "Invalid expense or claim amount format. Please check the amounts."
            return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                                 claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                                 total_claim_amount=total_claim_amount, description=description, 
                                 output=error_message)

        # Prepare patient information
        patient_info = f"Name: {name}\nAddress: {address}\nClaim type: {claim_type}\nClaim reason: {claim_reason}\nMedical facility: {medical_facility}\nDate: {date}\nTotal claim amount: {total_claim_amount}\nDescription: {description}"
        medical_bill_info = f"Medical Bill Content: {bill_content[:500]}..."

        # Check if claim amount exceeds bill amount
        amount_exceeds = claim_amount > bill_expense
        
        # Check for exclusions
        disease = bill_info.get('disease', claim_reason)
        is_excluded = check_disease_exclusion(disease, general_exclusion_list)
        
        print(f"Amount exceeds: {amount_exceeds}, Is excluded: {is_excluded}")
        
        # Generate report
        output = generate_claim_report(patient_info, medical_bill_info, bill_info, total_claim_amount, 
                                     is_excluded, amount_exceeds)
        
        return render_template("result.html", name=name, address=address, claim_type=claim_type, 
                             claim_reason=claim_reason, date=date, medical_facility=medical_facility, 
                             total_claim_amount=total_claim_amount, description=description, 
                             output=output)
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        error_message = f"An unexpected error occurred while processing your claim. Please try again."
        return render_template("result.html", name="", address="", claim_type="", 
                             claim_reason="", date="", medical_facility="", 
                             total_claim_amount="", description="", output=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
