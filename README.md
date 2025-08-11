# 🛡 VeriPol.AI — AI-Powered Insurance Claim Processing

## 📌 Overview  
**VeriPol.AI** is an intelligent system designed to **automate insurance claim processing and policy verification**. It instantly analyzes claim requests, checks them against policy rules, and provides real-time feedback to both customers and executives. By removing manual bottlenecks, it speeds up approvals, reduces errors, and enhances customer satisfaction.

## 🚨 Problem Statement  
Insurance claim processing is **slow**, often requiring **manual verification** that causes delays. Customers face confusion due to complex policy terms, leading to incomplete or invalid claims. Executives waste hours on repetitive checks. **VeriPol.AI** solves this by instantly analyzing claims, matching them with policy rules, and providing **accurate, instant decisions**.

## 🧰 Technologies Used  
- **Python** (Flask for backend, Pandas for data handling)  
- **Groq API** (LLM for natural language understanding)  
- **FAISS** (semantic search for policy rules)  
- **EasyOCR** (scanned document text extraction)  
- **PyPDF2** (policy/claim PDF parsing)  
- **HTML/CSS + Jinja2** (frontend UI)

## ⚙️ Features  
- 📄 **Policy Understanding** – Reads & interprets insurance documents  
- 📝 **Claim Validation** – Checks claims in real time against rules  
- 📚 **Policy Simplification** – Explains terms in plain language  
- ⚡ **Instant Feedback** – Supports both customers & staff  
- 🔍 **OCR Support** – Reads scanned claim forms

## 🚀 How It Works  
1. User uploads claim documents (PDF or scanned image)  
2. OCR & PDF parser extract claim and policy data  
3. FAISS retrieves relevant clauses from policy rules  
4. Groq API evaluates the claim’s validity  
5. Chatbot returns instant decision & explanation

## 🖥 Setup Instructions  
```bash
# Clone repository
git clone https://github.com/yourusername/veripol-ai.git
cd veripol-ai

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
