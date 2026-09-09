import time
import os
import pymupdf as fitz  # Eliminates legacy warnings
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from rich.console import Console
from rich.markdown import Markdown

# Initialize the rich terminal console
console = Console()

# ==========================================
# CONFIGURATION
# ==========================================

PDF_PATH = "data/Placement Policy 2027_B.Tech_MCA.pdf"


# ==========================================
# GEMINI SETUP
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found.")
    print("Check your .env file.")
    exit()

client = genai.Client(api_key=api_key)


# ==========================================
# PDF TEXT EXTRACTION
# ==========================================

def extract_text_from_pdf(pdf_path):
    print("\nReading PDF...")
    document = fitz.open(pdf_path)
    text = ""

    for page_number, page in enumerate(document):
        page_text = page.get_text()
        text += f"\n[Page {page_number + 1}]\n"
        text += page_text

    page_count = len(document)
    document.close()

    print("PDF loaded successfully.")
    print("Pages:", page_count)
    return text


# ==========================================
# CHUNKING
# ==========================================

def create_chunks(text):
    print("\nCreating text chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    print("Chunks created:", len(chunks))
    return chunks


# ==========================================
# EMBEDDINGS
# ==========================================

def create_embeddings(chunks):
    print("\nCreating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    print("Embeddings created.")
    return model, embeddings


# ==========================================
# FAISS
# ==========================================

def create_vector_database(embeddings):
    print("\nCreating FAISS database...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    print("FAISS database created.")
    return index


# ==========================================
# SEARCH
# ==========================================

def search_document(question, model, index, chunks):
    question_embedding = model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        4
    )
    results = []
    for i in indices[0]:
        if i < len(chunks):
            results.append(chunks[i])
    return results


# ==========================================
# GEMINI (Your Layout + Formatting Rule Added)
# ==========================================

def ask_gemini(chat_session, question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question ONLY using the information
provided in the PDF context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not present in the PDF, say:
   "Sorry, I could not find this information in the provided PDF."
4. Give a clear, structured, and simple answer. Use Markdown formatting (bold text, bullet points) to make it highly readable.
5. Mention the page number when possible.

PDF CONTEXT:
-------------------------
{context}
-------------------------

QUESTION:
{question}

ANSWER:
"""

    # Try Gemini up to 3 times using the persistent chat session object
    for attempt in range(3):
        try:
            print(f"Calling Gemini... Attempt {attempt + 1}/3")
            response = chat_session.send_message(prompt)
            return response.text

        except Exception as e:
            print(f"Gemini request failed: {e}")
            if attempt < 2:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return (
                    "Gemini API is temporarily unavailable. "
                    "Please try your question again."
                )


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():
    print("\n")
    print("=" * 50)
    print("       PDF QUESTION ANSWERING AI")
    print("=" * 50)

    # Step 1: Read PDF
    text = extract_text_from_pdf(PDF_PATH)
    if not text.strip():
        print("\nNo text was found in the PDF.")
        return

    # Step 2: Create Chunks
    chunks = create_chunks(text)

    # Step 3: Embeddings
    model, embeddings = create_embeddings(chunks)

    # Step 4: Vector Database
    index = create_vector_database(embeddings)

    # Initialize a persistent chat session to eliminate the AFC warning natively
    chat_session = client.chats.create(model="gemini-3.6-flash")

    print("\n")
    print("=" * 50)
    print("PDF processing completed!")
    print("You can now ask questions.")
    print("Type 'exit' to quit.")
    print("=" * 50)

    # Step 5: Query Loop
    while True:
        question = input("\nAsk your question:\n> ")

        if question.lower().strip() == "exit":
            print("\nGoodbye!")
            break

        if not question.strip():
            print("Please enter a question.")
            continue

        print("\nSearching PDF...")
        relevant_chunks = search_document(question, model, index, chunks)

        print("Generating answer...")
        answer = ask_gemini(chat_session, question, relevant_chunks)

        # Styled output display block using rich
        print("\n" + "=" * 50)
        console.print("[bold cyan]AI ANSWER[/bold cyan]")
        print("=" * 50)
        
        # This renders Gemini's markdown response cleanly in the terminal
        md_answer = Markdown(answer)
        console.print(md_answer)


# ==========================================
# START PROGRAM
# ==========================================

if __name__ == "__main__":
    main()
