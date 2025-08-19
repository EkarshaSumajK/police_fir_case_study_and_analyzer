import os
import chromadb
import google.generativeai as genai
from PyPDF2 import PdfReader
from pathlib import Path
import hashlib
from datetime import datetime
import time
from tqdm import tqdm
import sys

# Configuration
CHROMA_PATH = "police_vector_db"
DOCUMENTS_COLLECTION = "documents_collection"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class ProgressTracker:
    """Enhanced progress tracking with detailed metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_files = 0
        self.processed_files = 0
        self.total_chunks = 0
        self.processed_chunks = 0
        self.errors = []
        
    def set_total_files(self, count):
        self.total_files = count
        
    def file_started(self, filename):
        print(f"\n📄 [{self.processed_files + 1}/{self.total_files}] Processing: {filename}")
        
    def file_completed(self, filename, chunk_count):
        self.processed_files += 1
        self.processed_chunks += chunk_count
        elapsed = time.time() - self.start_time
        rate = self.processed_files / elapsed if elapsed > 0 else 0
        
        print(f"✅ Completed {filename} - {chunk_count} chunks")
        print(f"📊 Progress: {self.processed_files}/{self.total_files} files "
              f"({(self.processed_files/self.total_files)*100:.1f}%) - "
              f"Rate: {rate:.2f} files/sec")
        
    def add_error(self, filename, error):
        self.errors.append((filename, str(error)))
        print(f"❌ Error in {filename}: {error}")
        
    def show_summary(self):
        elapsed = time.time() - self.start_time
        print(f"\n{'='*50}")
        print(f"🎉 INGESTION COMPLETE!")
        print(f"{'='*50}")
        print(f"📊 Files processed: {self.processed_files}/{self.total_files}")
        print(f"📊 Total chunks created: {self.processed_chunks}")
        print(f"⏱️  Total time: {elapsed:.2f} seconds")
        print(f"⚡ Average rate: {self.processed_files/elapsed:.2f} files/sec")
        
        if self.errors:
            print(f"\n⚠️  Errors encountered ({len(self.errors)}):")
            for filename, error in self.errors:
                print(f"   • {filename}: {error}")

def setup_chromadb():
    """Initialize ChromaDB with progress indication"""
    print("🔧 Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(DOCUMENTS_COLLECTION)
        print(f"✅ Found existing collection with {collection.count()} documents")
    except:
        collection = client.create_collection(DOCUMENTS_COLLECTION)
        print("📁 Created new documents collection")
    return collection

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with progress indication"""
    print("   📖 Extracting text from PDF...")
    reader = PdfReader(pdf_path)
    text = ""
    
    # Show page processing progress for large PDFs
    pages = reader.pages
    if len(pages) > 5:  # Show progress for larger PDFs
        for i, page in enumerate(tqdm(pages, desc="   Processing pages", leave=False)):
            text += page.extract_text() + "\n"
    else:
        for page in pages:
            text += page.extract_text() + "\n"
            
    print(f"   📄 Extracted text from {len(pages)} pages ({len(text):,} characters)")
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks with progress indication"""
    print(f"   ✂️  Creating chunks (size: {chunk_size}, overlap: {overlap})...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    # Filter out very small chunks
    valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 50]
    print(f"   📦 Created {len(valid_chunks)} valid chunks (filtered {len(chunks) - len(valid_chunks)} small chunks)")
    return valid_chunks

def generate_embedding(text):
    """Generate embedding for text"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return result["embedding"]

def ingest_document(collection, file_path, progress_tracker):
    """Process and ingest a single document with detailed progress"""
    progress_tracker.file_started(file_path.name)
    
    try:
        # Extract text
        if file_path.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(file_path)
        else:
            print("   📖 Reading text file...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"   📄 Read {len(text):,} characters")
        
        # Create chunks
        chunks = chunk_text(text)
        
        if not chunks:
            print(f"   ⚠️  No valid chunks created from {file_path.name}")
            progress_tracker.file_completed(file_path.name, 0)
            return
        
        # Prepare data for ChromaDB
        print(f"   🧠 Generating embeddings for {len(chunks)} chunks...")
        documents = []
        metadatas = []
        embeddings = []
        ids = []
        
        # Process chunks with progress bar
        for i, chunk in enumerate(tqdm(chunks, desc="   Creating embeddings", leave=False)):
            # Generate unique ID
            chunk_id = f"{file_path.stem}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            
            # Generate embedding
            try:
                embedding = generate_embedding(chunk)
            except Exception as e:
                print(f"   ⚠️  Failed to generate embedding for chunk {i}: {e}")
                continue
            
            documents.append(chunk)
            metadatas.append({
                "document_name": file_path.name,
                "document_path": str(file_path),
                "chunk_index": i,
                "ingested_at": datetime.now().isoformat(),
                "file_type": file_path.suffix[1:],
                "chunk_length": len(chunk)
            })
            embeddings.append(embedding)
            ids.append(chunk_id)
        
        if not documents:
            print(f"   ❌ No valid embeddings generated for {file_path.name}")
            progress_tracker.add_error(file_path.name, "No valid embeddings generated")
            return
        
        # Add to ChromaDB
        print(f"   💾 Storing {len(documents)} chunks in database...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        
        progress_tracker.file_completed(file_path.name, len(documents))
        
    except Exception as e:
        progress_tracker.add_error(file_path.name, e)

def main():
    """Main ingestion process with comprehensive progress tracking"""
    print("🚀 Starting Document Ingestion Process")
    print("="*50)
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker()
    
    # Setup
    print("🔑 Configuring Google AI...")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    collection = setup_chromadb()
    
    # Define documents directory
    docs_dir = Path("Document.py")  # Note: This should probably be a directory like "documents/"
    if not docs_dir.exists():
        docs_dir.mkdir()
        print(f"📁 Created directory: {docs_dir}")
        print("Please add your case documents (PDF/TXT) to this directory")
        return
    
    # Find supported files
    print("🔍 Scanning for documents...")
    supported_formats = ['.pdf', '.txt', '.md']
    files = [f for f in docs_dir.iterdir() if f.suffix.lower() in supported_formats]
    
    if not files:
        print("❌ No supported documents found!")
        print(f"Supported formats: {', '.join(supported_formats)}")
        print(f"Please add documents to: {docs_dir.absolute()}")
        return
    
    # Show file summary
    print(f"📚 Found {len(files)} documents to process:")
    for i, file_path in enumerate(files, 1):
        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"   {i}. {file_path.name} ({size_mb:.2f} MB)")
    
    progress_tracker.set_total_files(len(files))
    
    # Process all documents
    print(f"\n🔄 Starting batch processing...")
    for file_path in files:
        ingest_document(collection, file_path, progress_tracker)
    
    # Final summary
    progress_tracker.show_summary()
    final_count = collection.count()
    print(f"🗄️  Final database size: {final_count} total chunks")

if __name__ == "__main__":
    main()