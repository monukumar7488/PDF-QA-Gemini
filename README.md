# PDF-QA-Gemini 🤖📄

A high-performance **Retrieval-Augmented Generation (RAG)** application that allows you to chat with your local PDF documents. It securely leverages Google's advanced **Gemini 3.6 Flash** model and uses semantic vector searching to extract exact, grounded answers directly from document text.

## ✨ Features
* **Smart PDF Parsing:** Seamlessly reads text pages using `PyMuPDF`.
* **Semantic Vector Search:** Chunking text strings with LangChain chunk utilities and processing dense vector spaces via `sentence-transformers` and a lightning-fast `FAISS` database.
* **Production-Ready API Setup:** Connects directly to the modern `google-genai` Python SDK utilizing structured chat session controllers to completely eliminate Automatic Function Calling (AFC) execution warnings.
* **Beautiful Terminal UI:** Renders clean, highly scannable Markdown responses natively inside your console screen using the `rich` library layout.

---

## 🛠️ Project Stack
* **LLM Engine:** Google Gemini (`gemini-3.6-flash`)
* **Text Processing:** LangChain Text Splitters & PyMuPDF (`PyMuPDF`)
* **Embedding Model:** `all-MiniLM-L6-v2` (via Sentence-Transformers)
* **Vector DB Engine:** Facebook AI Similarity Search (`faiss-cpu`)
* **Terminal Interface:** `rich`

---

## 🚀 Getting Started

Follow these steps to set up and run the application locally on your machine.

### 1. Prerequisites & Installation
Clone the repository and navigate into the project workspace directory:
```bash
git clone https://github.com
cd PDF-QA-Gemini
```

Create and activate a fresh Python virtual environment (`venv`):
```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
.\venv\Scripts\Activate.ps1

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

Install all required application development packages seamlessly:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
For security, this application reads your credentials via local environment configurations. 

1. Create a file named `.env` in the root folder of your project.
2. Generate your Gemini API token via Google AI Studio and inject it into the file without any inverted commas:
```env
GEMINI_API_KEY=AIzaSyYourActualSecretKeyHere
```

### 3. Load Your Document
Drop the PDF file you want to query inside your data folder asset path. (By default, the script looks for your policy layout matching `data/Placement Policy 2027_B.Tech_MCA.pdf`).

### 4. Run the RAG Pipeline
Boot your application directly using your active terminal environment wrapper:
```bash
python app.py
```

Type your questions naturally inside the prompt. Type `exit` whenever you are ready to terminate the conversational query loop block.

---

## 🔒 Security Notice
Your private `.env` key mapping configurations and bulky `venv/` dependency binary structures are explicitly sandboxed locally inside the `.gitignore` setup configuration, preventing security token exposures across your cloud repository branches.
