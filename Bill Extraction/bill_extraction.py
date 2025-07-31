import os
import json
import re
import yaml
import pytesseract
from pdf2image import convert_from_path
from groq import Groq

CONFIG_PATH = "config.yaml"

# 1. Load Groq API key
def load_config():
    try:
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
            return data.get('groq_api_key')
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None

# 2. Convert PDF to text using OCR
def extract_text_from_pdf(pdf_path):
    try:
        pages = convert_from_path(pdf_path, dpi=300)
        full_text = ""
        for page in pages:
            text = pytesseract.image_to_string(page)
            full_text += text + "\n"

        # Save extracted text for debugging
        with open("ocr_output.txt", "w", encoding="utf-8") as f:
            f.write(full_text)

        return full_text
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

# 3. Call Groq LLM with better universal prompt
def get_invoice_info_from_llm(document_text):
    api_key = load_config()
    if not api_key:
        raise ValueError("API key not found. Add it to config.yaml")

    client = Groq(api_key=api_key)

    prompt = (
        "You are a medical billing assistant AI. Extract from any type of universal-format medical bill:\n"
        "1. Disease or diagnosis (look for conditions, symptoms, reasons for consultation)\n"
        "2. Bill amount (look for labels like 'bill amount', 'total charge', 'grand total', 'amount due')\n"
        "3. Claimed amount (if mentioned)\n"
        "Return output in JSON format like this:\n"
        "{'disease': '', 'bill_amount': '', 'claimed_amount': ''}\n"
        "If not found, return 'Not found'.\n\n"
        f"DOCUMENT:\n{document_text}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3-70b-8192",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": document_text}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        raw_response = response.choices[0].message.content
        print("\n[LLM RESPONSE]\n", raw_response)

        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(raw_response)

    except Exception as e:
        print(f"LLM error: {e}")
        return {
            "disease": "Error",
            "bill_amount": "Error",
            "claimed_amount": "Error"
        }

# 4. Regex fallback if LLM fails
def extract_amounts_regex(text):
    patterns = {
        'bill_amount': r'(bill amount|grand total|amount due|total charges?)[:\s₹$]*([\d,]+(?:\.\d{1,2})?)',
        'claimed_amount': r'(claimed amount|amount claimed)[:\s₹$]*([\d,]+(?:\.\d{1,2})?)'
    }
    amounts = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text.lower())
        if match:
            amounts[key] = match.group(2).replace(',', '')
    return amounts

# 5. Main function
def main():
    pdf_path = "Bills/MedicalBill1.pdf"

    if not os.path.exists(pdf_path):
        print("❌ PDF not found. Check path:", pdf_path)
        return

    print("📄 Extracting text via OCR...")
    ocr_text = extract_text_from_pdf(pdf_path)
    if not ocr_text.strip():
        print("⚠️ OCR returned no text.")
        return

    print("🤖 Querying LLM...")
    llm_result = get_invoice_info_from_llm(ocr_text)

    # Apply regex fallback if LLM failed
    regex_result = extract_amounts_regex(ocr_text)
    for key in ["bill_amount", "claimed_amount"]:
        if llm_result.get(key) in ["0", "$0", "Not found", "", None] and key in regex_result:
            llm_result[key] = regex_result[key]

    print("\n✅ Final Extracted Info:")
    for k, v in llm_result.items():
        print(f"{k.title().replace('_', ' ')}: {v}")

if __name__ == "__main__":
    main()
