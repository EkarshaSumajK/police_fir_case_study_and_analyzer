# load_law_corpus.py
import os
import csv
import pandas as pd
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai
import time
import fitz  # PyMuPDF for PDF processing
from pathlib import Path
import time

# --- CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)
CHROMA_PATH = "fir_vector_db"
LAW_COLLECTION_NAME = "laws_collection"
EMBEDDING_MODEL = "models/text-embedding-004"
PDF_DOCS_DIR = "fir_docs"  # Directory containing PDF law documents

# --- HELPER FUNCTION FOR EMBEDDING ---
def embed_text_gai(text, model):
    """Generates embedding for a given text using Google AI."""
    try:
        result = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_DOCUMENT")
        return result['embedding']
    except Exception as e:
        print(f"Error embedding text: {text[:50]}... Error: {e}")
        time.sleep(1) # Rate limit handling
        return embed_text_gai(text, model) # Retry

# --- HELPER FUNCTION FOR PDF TEXT EXTRACTION ---
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def get_pdf_metadata(pdf_path):
    """Extracts metadata from PDF filename and content."""
    filename = Path(pdf_path).stem

    # Determine the law code and section based on filename
    if "bns" in filename.lower() or "250882" in filename:
        code = "BNS"
        name = "Bharatiya Nyaya Sanhita"
    elif "crpc" in filename.lower() or "criminalprocedure" in filename.lower() or "250883" in filename:
        code = "CrPC"
        name = "Code of Criminal Procedure"
    elif "ipc" in filename.lower() or "penal" in filename.lower():
        code = "IPC"
        name = "Indian Penal Code"
    elif "amendment" in filename.lower() or "amended" in filename.lower():
        code = "Amendment"
        name = "Criminal Law Amendment"
    else:
        code = "Law"
        name = filename.replace("_", " ").title()

    return {
        'code': code,
        'name': name,
        'filename': filename,
        'filepath': pdf_path
    }

# --- MAIN SCRIPT ---
def main():
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

        # 1. ===== LOAD LAWS COLLECTION =====
    print(f"\n--- Loading PDF Law Documents into '{LAW_COLLECTION_NAME}' ---")

    # Get or create collection (don't delete existing data)
    existing_collections = [c.name for c in client.list_collections()]
    if LAW_COLLECTION_NAME in existing_collections:
        print(f"Collection '{LAW_COLLECTION_NAME}' already exists. Adding PDF documents to existing collection.")
        law_collection = client.get_collection(name=LAW_COLLECTION_NAME)
        existing_count = law_collection.count()
        print(f"Existing documents in collection: {existing_count}")
    else:
        print(f"Creating new collection '{LAW_COLLECTION_NAME}'.")
        law_collection = client.create_collection(name=LAW_COLLECTION_NAME)

    # Check if PDF directory exists
    if not os.path.exists(PDF_DOCS_DIR):
        print(f"Error: The directory '{PDF_DOCS_DIR}' was not found.")
        return

    # Get all PDF files from the directory
    pdf_files = list(Path(PDF_DOCS_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DOCS_DIR}' directory.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    # Process each PDF file
    for idx, pdf_file in enumerate(pdf_files):
        pdf_path = str(pdf_file)
        print(f"Processing PDF: {pdf_file.name}")

        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            print(f"Warning: No text extracted from {pdf_file.name}, skipping.")
            continue

        # Get metadata from filename
        metadata = get_pdf_metadata(pdf_path)

        # Split long texts into chunks for better embedding
        chunk_size = 2000  # characters per chunk
        text_chunks = [pdf_text[i:i + chunk_size] for i in range(0, len(pdf_text), chunk_size)]

        print(f"  - Split into {len(text_chunks)} text chunks")

        # Create documents for each chunk
        for chunk_idx, chunk in enumerate(text_chunks):
            doc_content = f"{metadata['name']} ({metadata['code']}): {chunk[:200]}..."  # First 200 chars as preview

            documents.append(chunk)
            metadatas.append({
                'code': metadata['code'],
                'law_name': metadata['name'],
                'filename': metadata['filename'],
                'filepath': pdf_path,
                'chunk_index': chunk_idx,
                'total_chunks': len(text_chunks),
                'full_text_preview': chunk[:500] + "..." if len(chunk) > 500 else chunk,
                # Add compatibility fields for FIR analysis
                'section_number': metadata['name'],  # Use law name as section number for compatibility
                'section_name': metadata['name'],
                'full_text': chunk,  # The actual chunk content
                'url': pdf_path  # File path as URL
            })
            # Create unique ID to avoid conflicts with existing documents
            unique_suffix = str(int(time.time() * 1000000))[-8:]  # Last 8 digits of microsecond timestamp
            ids.append(f"{metadata['code'].lower()}_{idx}_{chunk_idx}_{unique_suffix}")

    # Embed and add to Chroma in batches for stability
    print("Embedding and adding PDF law documents to the collection in batches...")
    batch_size = 50  # Smaller batch size for PDF content
    total = len(documents)
    print(f"Total documents to process: {total}")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"- Processing {start+1}-{end} / {total}")
        batch_docs = documents[start:end]
        batch_metas = metadatas[start:end]
        batch_ids = ids[start:end]

        # Generate embeddings for the batch
        batch_embeddings = []
        for doc in batch_docs:
            embedding = embed_text_gai(doc, EMBEDDING_MODEL)
            batch_embeddings.append(embedding)

        # Add batch to collection
        law_collection.add(
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
        )

    final_count = law_collection.count()
    print(f"Successfully added {len(documents)} PDF documents to '{LAW_COLLECTION_NAME}'.")
    print(f"Total documents in collection: {final_count}")
    print("\n✅ PDF corpus loading complete - all documents in same collection.")

if __name__ == "__main__":
    main()