# app.py
import streamlit as st
import google.generativeai as genai
import chromadb
import os
import json
from dotenv import load_dotenv
import PyPDF2
import base64
from datetime import datetime, timedelta
import logging
import hashlib
from pathlib import Path
from PyPDF2 import PdfReader

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Police AI Analysis Assistant",
    page_icon="⚖️",
    layout="wide"
)

# --- APPLICATION STYLING ---
_APP_STYLE = """
<style>
:root { --bg: #ffffff; --card-bg: #ffffff; --text: #111827; --muted: #6b7280; --border: rgba(0,0,0,0.08); --shadow: 0 1px 2px rgba(0,0,0,0.05); --quote-bg: #0e1117; --quote-text: #e6e6e6; --badge-green-bg: #e7f7ef; --badge-green-text: #127c51; --badge-amber-bg: #fff3e0; --badge-amber-text: #8a5a00; --badge-red-bg: #fdecea; --badge-red-text: #8a1c1c; --divider: rgba(0,0,0,0.08); }
@media (prefers-color-scheme: dark) { :root { --bg: #0b0f14; --card-bg: #111827; --text: #e5e7eb; --muted: #9ca3af; --border: rgba(255,255,255,0.08); --shadow: 0 1px 2px rgba(0,0,0,0.3); --quote-bg: #0b0f14; --quote-text: #e5e7eb; --badge-green-bg: #0f2e24; --badge-green-text: #34d399; --badge-amber-bg: #2a1e0a; --badge-amber-text: #fbbf24; --badge-red-bg: #2a0f0f; --badge-red-text: #f87171; --divider: rgba(255,255,255,0.08); } }
.main > div { padding-top: 0.75rem; color: var(--text); } .card { border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.25rem; background: var(--card-bg); box-shadow: var(--shadow); } .badge { display:inline-block; padding:0.25rem 0.6rem; border-radius:999px; font-size:0.8rem; font-weight:600; margin-left:0.5rem; } .badge-green { background:var(--badge-green-bg); color:var(--badge-green-text); } .badge-amber { background:var(--badge-amber-bg); color:var(--badge-amber-text); } .badge-red { background:var(--badge-red-bg); color:var(--badge-red-text); } .quote-block { background:var(--quote-bg); color:var(--quote-text); padding:0.75rem 0.9rem; border-radius:8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; white-space: pre-wrap; } .divider { height:1px; background:var(--divider); margin:0.75rem 0 1rem; } .small-muted { color: var(--muted); font-size: 0.85rem; }
.source-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0; }
.confidence-high { border-left: 4px solid #22c55e; } .confidence-medium { border-left: 4px solid #f59e0b; } .confidence-low { border-left: 4px solid #ef4444; }
.chunk-text { background: var(--quote-bg); color: var(--quote-text); padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.9rem; line-height: 1.4; margin-top: 0.5rem; border-left: 3px solid var(--muted); max-height: 200px; overflow-y: auto; }
.chunk-preview { background: var(--quote-bg); color: var(--quote-text); padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.85rem; line-height: 1.4; margin-top: 0.5rem; border-left: 3px solid var(--muted); }
</style>
"""
st.markdown(_APP_STYLE, unsafe_allow_html=True)

# --- SETUP AND CONFIGURATION ---
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError:
    st.error("🚨 Google API Key not found. Please set it in your .env file.", icon="🚨")
    st.stop()

# --- Fixed Configuration Section ---
LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
CHROMA_PATH = "fir_vector_db"  # For laws collection
DOCUMENTS_CHROMA_PATH = "police_vector_db"  # For case documents collection
LAW_COLLECTION_NAME = "laws_collection"
DOCUMENTS_COLLECTION_NAME = "documents_collection"  # Case documents collection
LLM_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 65535
USER_INPUT_CHAR_LIMIT = 400

# --- Simplified Cached Functions ---
@st.cache_resource(show_spinner=False)
def _get_law_collection():
    """Simplified collection getter"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        return client.get_collection(LAW_COLLECTION_NAME)
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB: {e}")
        return None

@st.cache_resource(show_spinner=False)
def _get_documents_collection():
    """Get documents collection for case documents"""
    try:
        client = chromadb.PersistentClient(path=DOCUMENTS_CHROMA_PATH)  # Different path
        return client.get_collection(DOCUMENTS_COLLECTION_NAME)
    except Exception as e:
        st.error(f"Failed to connect to Documents ChromaDB: {e}")
        return None

@st.cache_data(show_spinner=False)
def translate_to_te(text: str) -> str:
    """Simplified translation function"""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        model = genai.GenerativeModel(
            LLM_MODEL_NAME,
            generation_config={
                "response_mime_type": "text/plain",
                "temperature": 0.0,
                "max_output_tokens": 2048,
            }
        )
        resp = model.generate_content(
            f"Translate to Telugu. Preserve legal meaning: {text}"
        )
        return getattr(resp, "text", "").strip()
    except Exception:
        return text

# --- COMMON UTILITY FUNCTIONS ---
def _fallback_te(te_val: str, en_val: str) -> str:
    """Fallback to English if Telugu translation is missing or invalid."""
    bads = {"", None, "తెలుగు పాఠ్యం సందర్భంలో అందించబడలేదు.", "తెలుగు అనువాదం అందుబాటులో లేదు", "N/A"}
    te = (te_val or "").strip()
    return te if te and te not in bads else (translate_to_te(en_val.strip()) if en_val else "")

# --- FIR ANALYSIS FUNCTIONS ---
def call_gemini_for_fir(system_prompt):
    """Call the Gemini API for FIR analysis and return the parsed JSON response."""
    response_text = ""
    try:
        model = genai.GenerativeModel(
            LLM_MODEL_NAME,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": LLM_TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
        )
        response = model.generate_content(system_prompt)
        response_text = getattr(response, "text", "")
        return json.loads(response_text)
    except Exception as e:
        st.error(f"An error occurred with the Gemini API call: {e}")
        try:
            clean_text = (response_text or "").strip().replace("```json", "").replace("```", "")
            if clean_text:
                return json.loads(clean_text)
        except Exception:
            pass
        if response_text:
            st.code(response_text, language="text")
        return None

def _embed_query_text(text: str):
    """Embed query text for analysis."""
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text, task_type="RETRIEVAL_QUERY")
        return result["embedding"]
    except Exception as e: 
        st.error(f"Failed to embed query text: {e}")
        return None

# --- ChromaDB Initialization ---
@st.cache_resource(show_spinner=False)
def get_law_collection():
    """Initialize and return the ChromaDB collection"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        return client.get_collection(LAW_COLLECTION_NAME)
    except Exception as e:
        st.error(f"🚨 Failed to connect to Vector Database: {e}")
        st.stop()

@st.cache_resource(show_spinner=False)
def get_documents_collection():
    """Initialize and return the documents ChromaDB collection, create if not exists"""
    try:
        client = chromadb.PersistentClient(path=DOCUMENTS_CHROMA_PATH)
        return client.get_or_create_collection(DOCUMENTS_COLLECTION_NAME)
    except Exception as e:
        st.error(f"Failed to connect to Documents ChromaDB: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file-like object"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return [chunk for chunk in chunks if len(chunk.strip()) >= 50]

def ingest_document(collection, file):
    """Process and ingest a single PDF document"""
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    
    if not chunks:
        return False
    
    documents = []
    metadatas = []
    embeddings = []
    ids = []
    
    file_name = file.name
    for i, chunk in enumerate(chunks):
        chunk_id = f"{Path(file_name).stem}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
        
        embedding = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=chunk,
            task_type="RETRIEVAL_DOCUMENT"
        )["embedding"]
        
        documents.append(chunk)
        metadatas.append({
            "document_name": file_name,
            "document_path": file_name,  # Since it's uploaded, use name
            "chunk_index": i,
            "ingested_at": datetime.now().isoformat(),
            "file_type": "pdf",
            "chunk_length": len(chunk)
        })
        embeddings.append(embedding)
        ids.append(chunk_id)
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )
    return True

# Initialize collections at startup
law_collection = get_law_collection()
documents_collection = get_documents_collection()

@st.cache_data(show_spinner=False)
def retrieve_law_sections(query_text, k=5):
    """Retrieve relevant law sections from ChromaDB"""
    if not law_collection:
        st.error("Law collection not available")
        return []

    embedding = _embed_query_text(query_text)
    if embedding is None:
        return []

    results = law_collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["metadatas", "documents"],  # Include documents for full text access
    )

    if not results or not results.get('metadatas'):
        return []

    # Combine metadata with document content for better FIR analysis
    law_sections = []
    for i, metadata in enumerate(results['metadatas'][0]):
        # Create a comprehensive law section entry
        law_section = {
            'section_number': metadata.get('section_number', metadata.get('law_name', 'N/A')),
            'section_name': metadata.get('section_name', metadata.get('law_name', 'N/A')),
            'full_text': results['documents'][0][i] if results.get('documents') and i < len(results['documents'][0]) else metadata.get('full_text', ''),
            'code': metadata.get('code', 'Law'),
            'filename': metadata.get('filename', ''),
            'chunk_index': metadata.get('chunk_index', 0),
            'url': metadata.get('filepath', metadata.get('url', ''))
        }
        law_sections.append(law_section)

    return law_sections

@st.cache_data(show_spinner=False)
def retrieve_case_documents(query_text, k=5):
    """Retrieve relevant case document chunks from ChromaDB"""
    if not documents_collection:
        return []
    
    embedding = _embed_query_text(query_text)
    if embedding is None:
        return []
        
    try:
        results = documents_collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return {
            'documents': results.get('documents', [[]])[0],
            'metadatas': results.get('metadatas', [[]])[0],
            'distances': results.get('distances', [[]])[0]
        }
    except Exception as e:
        st.error(f"Error retrieving case documents: {e}")
        return []

# --- FIR Analysis Prompt ---
def build_fir_system_prompt(fir_summary, relevant_laws, jurisdiction):
    """Constructs the system prompt for analyzing an FIR."""
    # Build laws context with better formatting for PDF chunks
    laws_context = ""
    if relevant_laws:
        for i, law in enumerate(relevant_laws, 1):
            code = law.get('code', 'Law')
            section_name = law.get('section_name', 'N/A')
            filename = law.get('filename', '')
            chunk_info = f" (Chunk {law.get('chunk_index', 0)})" if law.get('chunk_index') is not None else ""

            laws_context += f"\n{i}. {code} - {section_name}{chunk_info}"
            if filename:
                laws_context += f" (from {filename})"
            laws_context += f":\n{law.get('full_text', '')}\n"
    else:
        laws_context = "No relevant laws found."

    json_schema = {
        "sections": [{
            "section": "BNS/CrPC/IPC Section Number (e.g., BNS 420 or CrPC 41 or IPC 41)",
            "rationale_en": "2-3 sentence rationale in English.",
            "rationale_te": "2-3 sentence rationale in Telugu.",
            "reasoning_quote_en": "Verbatim quote from the provided law text.",
            "reasoning_quote_te": "చట్ట పాఠ్యంలోని సంబంధిత వాక్యాన్ని తెలుగు లో యథాతధంగా ఇవ్వండి.",
            "fir_quote_en": "Verbatim excerpt from the FIR summary.",
            "fir_quote_te": "కేసు వివరాల్లోని సంబంధిత వాక్యాన్ని తెలుగు లో యథాతధంగా ఇవ్వండి.",
            "law_citation": {
                "title_en": "Official name of the section in English",
                "title_te": "తెలుగు శీర్షిక.",
                "full_text_en": "Complete official text in English.",
                "full_text_te": "తెలుగు పూర్తి పాఠ్యం.",
                "url": "Official India Code URL (if available)."
            }
        }],
        "actions_en": ["Step 1 in English", "Step 2 in English"],
        "actions_te": ["Step 1 in Telugu", "Step 2 in Telugu"]
    }

    prompt = f"""You are an expert AI legal assistant for Indian Police. Analyze the case details and recommend BNS/CrPC/IPC sections based ONLY on the provided context. Provide rationale, verbatim quotes, and procedural steps. Your output MUST be a single, valid JSON object conforming EXACTLY to the schema.

Potentially Relevant Law Sections from PDF Documents:
{laws_context}

Jurisdiction: {jurisdiction}

Case Details to Analyze:
```
{fir_summary}
```

IMPORTANT: Your final output must be a valid JSON object matching this schema:
```json
{json.dumps(json_schema, indent=2)}
```

Note: The law sections above are extracted from PDF documents and may contain chunked content. Use the most relevant portions for your analysis.
"""
    return prompt

# --- Document Chat Functions ---
def generate_contextual_response(question: str, context_chunks: list, metadata: list):
    """Generate response using retrieved context chunks"""
    
    # Build context from retrieved chunks
    context_text = ""
    source_info = []
    
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadata)):
        context_text += f"\n[Source {i+1}]: {chunk}\n"
        source_info.append({
            'chunk_index': i+1,
            'document': meta.get('document_name', 'Unknown'),
            'chunk_id': meta.get('chunk_index', 'N/A'),
            'file_type': meta.get('file_type', 'unknown'),
            'chunk_text': chunk  # Add the actual chunk text here
        })
    
    # Create prompt for the AI model
    system_prompt = f"""You are an expert police case analysis assistant. Answer the user's question based on the provided case document context. 

INSTRUCTIONS:
- Use only the information provided in the context
- If the context doesn't contain enough information to answer the question, say so clearly
- Be precise and factual in your analysis
- Reference specific parts of the documents when relevant
- Provide actionable insights for police investigation when appropriate

**IMPORTANT: use the same text as it is in the document. **
CONTEXT FROM CASE DOCUMENTS:
{context_text}

USER QUESTION: {question}

Provide a comprehensive answer based on the available context."""

    try:
        model = genai.GenerativeModel(
            LLM_MODEL_NAME,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 4096,
            }
        )
        response = model.generate_content(system_prompt)
        return response.text, source_info
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while analyzing the documents.", []

def render_sources(sources: list, distances: list = None):
    """Render source information with confidence indicators and chunk text"""
    if not sources:
        return
    
    st.markdown("### 📚 Sources Used")
    
    for i, source in enumerate(sources):
        # Determine confidence based on distance (if available)
        confidence_class = "confidence-medium"
        confidence_text = "Medium"
        
        if distances and i < len(distances):
            distance = distances[i]
            if distance < 0.3:
                confidence_class = "confidence-high"
                confidence_text = "High"
            elif distance > 0.7:
                confidence_class = "confidence-low"
                confidence_text = "Low"
        
        # Get chunk text and create preview
        chunk_text = source.get('chunk_text', '')
        chunk_preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        
        st.markdown(f"""
        <div class="source-card {confidence_class}">
            <strong>📄 {source['document']}</strong> 
            <span class="badge badge-amber">Chunk {source['chunk_id']}</span>
            <span class="badge badge-green">Confidence: {confidence_text}</span>
            <br>
            <span class="small-muted">File type: {source['file_type'].upper()}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Show chunk preview
        if chunk_preview:
            st.markdown(f"""
            <div class="chunk-preview">
                <strong>📄 Content Preview:</strong><br>
                {chunk_preview}
            </div>
            """, unsafe_allow_html=True)
        
        # Add expandable section for full chunk text if it's long
        if len(chunk_text) > 200:
            # Remove the inner expander and directly show the full chunk text
            st.markdown(f"""
            <div class="chunk-text">
            {chunk_text}
            </div>
            """, unsafe_allow_html=True)

# --- FIR Render Function ---
def _render_fir_analysis(res: dict, relevant_laws: list = None):
    """Renders the structured JSON output for FIR analysis."""
    if not res: st.info("No analysis available."); return
    if relevant_laws is None: relevant_laws = []
    st.subheader("✅ Recommended BNS/CrPC Sections")
    if res.get("sections"):
        for sec in res["sections"]:
            law = sec.get("law_citation", {}) or {}
            section_label = sec.get("section", "N/A").replace("IPC", "BNS")
            st.markdown(f"<div class='card'><strong>Section:</strong> <code>{section_label}</code>", unsafe_allow_html=True)

            # Add source information for PDF-based laws
            if relevant_laws and len(relevant_laws) > 0:
                # Try to find matching law from retrieved sections
                matching_law = None
                for rl in relevant_laws:
                    if (rl.get('section_name') in section_label or
                        rl.get('code') in section_label or
                        str(rl.get('section_number', '')) in section_label):
                        matching_law = rl
                        break

                if matching_law:
                    source_info = []
                    if matching_law.get('filename'):
                        source_info.append(f"📄 {matching_law['filename']}")
                    if matching_law.get('code'):
                        source_info.append(f"📋 {matching_law['code']}")
                    if matching_law.get('chunk_index') is not None:
                        source_info.append(f"📊 Chunk {matching_law['chunk_index']}")

                    if source_info:
                        st.markdown(f"<small class='small-muted'>Source: {' • '.join(source_info)}</small>", unsafe_allow_html=True)
            
            col_en, col_te = st.columns(2)
            with col_en:
                st.markdown("**English**")
                st.markdown(f"- Title: {law.get('title_en', 'N/A')}")
                st.info(f"Rationale: {sec.get('rationale_en', 'N/A')}")
                if quote_en := (sec.get("reasoning_quote_en") or "").strip():
                    st.markdown("Exact law quote (EN)")
                    st.markdown(f"<div class='quote-block'>{quote_en}</div>", unsafe_allow_html=True)
            with col_te:
                st.markdown("**తెలుగు**")
                st.markdown(f"- శీర్షిక: {_fallback_te(law.get('title_te'), law.get('title_en'))}")
                st.info(f"తర్కం: {_fallback_te(sec.get('rationale_te'), sec.get('rationale_en'))}")
                if quote_te := _fallback_te(sec.get("reasoning_quote_te"), sec.get("reasoning_quote_en")):
                    st.markdown("యథాతధ చట్ట కోట్ (TE)")
                    st.markdown(f"<div class='quote-block'>{quote_te}</div>", unsafe_allow_html=True)

            st.markdown("<details><summary>View full law text and source</summary>", unsafe_allow_html=True)
            if full_en := law.get("full_text_en") or law.get("full_text") or "": st.markdown("**Full text (English)**"); st.code(full_en, language="text")
            if full_te := _fallback_te(law.get("full_text_te"), full_en): st.markdown("**పూర్తి పాఠ్యం (తెలుగు)**"); st.code(full_te, language="text")
            if law.get("url"): st.caption(f"Source: {law.get('url')}")
            st.markdown("</details>", unsafe_allow_html=True)

            if fir_en := (sec.get("fir_quote_en") or "").strip():
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True); st.markdown("**FIR excerpt supporting this recommendation**")
                col_fir_en, col_fir_te = st.columns(2)
                with col_fir_en: st.markdown("FIR excerpt (EN)"); st.markdown(f"<div class='quote-block'>{fir_en}</div>", unsafe_allow_html=True)
                with col_fir_te: st.markdown("FIR ఉద్దరణ (TE)"); st.markdown(f"<div class='quote-block'>{_fallback_te(sec.get('fir_quote_te'), fir_en)}</div>", unsafe_allow_html=True)
            st.markdown("</div><div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("📋 Suggested Procedural Actions")
    col_en_a, col_te_a = st.columns(2)
    with col_en_a:
        st.markdown("**English**")
        for i, action in enumerate(res.get("actions_en", []), 1): st.markdown(f"{i}. {action}")
    with col_te_a:
        st.markdown("**తెలుగు**")
        actions_en, actions_te = res.get("actions_en", []), res.get("actions_te", [])
        if not actions_te and actions_en: actions_te = [translate_to_te(a) for a in actions_en]
        for i, action in enumerate(actions_te, 1): st.markdown(f"{i}. {action}")

# --- JUDGMENT ANALYSIS FUNCTIONS ---
def call_gemini_for_judgment(system_prompt):
    """Call the Gemini API for judgment analysis and return the parsed JSON response."""
    response_text = ""
    try:
        model = genai.GenerativeModel(LLM_MODEL_NAME, generation_config={"response_mime_type": "application/json", "temperature": LLM_TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS})
        response = model.generate_content(system_prompt)
        response_text = getattr(response, "text", "")
        
        if not response_text:
            st.error("Gemini API returned an empty response for judgment analysis. Please check the prompt or try again.")
            return None

        clean_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            st.error("Failed to parse Gemini API response for judgment analysis as JSON. Raw response:")
            st.code(clean_text, language="text")
            return None
    except Exception as e:
        st.error(f"An error occurred with the Gemini API call for judgment analysis: {str(e)}")
        return None

def build_judgment_prompt(judgment_text: str):
    """Constructs the system prompt for analyzing a court judgment."""
    json_schema = {
        "summary_en": "A comprehensive and in-depth summary of the entire judgment, covering the case background, key arguments, the court's reasoning, and the final verdict. Should be several paragraphs long.",
        "summary_te": "A comprehensive and in-depth summary in Telugu, translated faithfully from the English version.",
    }
    
    prompt = f"""You are an expert AI legal analyst. Your task is to read the following court judgment and provide a detailed, structured analysis.

    Analysis Requirements:
    - Provide a comprehensive, in-depth summary (several paragraphs long) that is neutral and covers all key aspects: the background, the facts, the parties' arguments, the legal issues, the court's reasoning, and the final verdict.
    - Detail the facts of the case and the main arguments from both parties.
    - Identify the core legal issues the court decided upon.
    - State the final verdict clearly.
    - List the major legal principles or precedents cited.
    - Provide faithful Telugu translations for all relevant fields.
    - Your output MUST be a single, valid JSON object conforming EXACTLY to the schema.

    Court Judgment to Analyze:
    ```
    {judgment_text}
    ```

    IMPORTANT: Your final output MUST be a single, valid JSON object. Do not add any text or explanation outside of the JSON structure. The JSON object must conform EXACTLY to the following schema:
    ```json
    {json.dumps(json_schema, indent=2)}
    ```
    """
    return prompt

def _render_judgment_analysis(res: dict):
    """Renders the structured JSON output for judgment analysis (strictly follows schema)."""
    if not res:
        st.info("No analysis available.")
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    st.subheader("📜 Judgment Summary")
    col_s_en, col_s_te = st.columns(2)
    with col_s_en:
        st.markdown("**Summary (English)**")
        st.write(res.get("summary_en", "N/A"))
    with col_s_te:
        st.markdown("**సారాంశం (తెలుగు)**")
        st.write(_fallback_te(res.get("summary_te"), res.get("summary_en")))
    
    st.markdown("</div>", unsafe_allow_html=True)

def process_judgment_text(judgment_text: str):
    """Main workflow for processing judgment text from any source."""
    st.session_state.judgment_messages.append({"role": "user", "content": judgment_text})
    with st.spinner("📜 Analyzing Judgment... This may take a moment."):
        system_prompt = build_judgment_prompt(judgment_text)
        analysis_result = call_gemini_for_judgment(system_prompt)
    st.session_state.judgment_messages.append({"role": "assistant", "analysis_result": analysis_result or {}})

# --- Initialize all session state variables ---
if "fir_messages" not in st.session_state:
    st.session_state.fir_messages = []
if "judgment_messages" not in st.session_state:
    st.session_state.judgment_messages = []
if "jurisdiction" not in st.session_state:
    st.session_state.jurisdiction = "Telangana"
if "top_k_laws" not in st.session_state:
    st.session_state.top_k_laws = 5
if "doc_chat_history" not in st.session_state:
    st.session_state.doc_chat_history = []
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "fir_text" not in st.session_state:
    st.session_state.fir_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Updated Streamlit UI ---
st.title("⚖️ Police AI Analysis Assistant")
st.warning("**Disclaimer:** For internal police use only. All outputs must be verified by a qualified officer. Do not include sensitive PII.", icon="⚠️")

# Create 3 tabs
tab1, tab2, tab3 = st.tabs(["⚖️ FIR Analysis", "📜 Judgment Analysis", "📄 Police Case Analysis Assistant"])

# Tab 1: FIR Analysis (unchanged)
with tab1:
    st.info("Paste FIRs to analyze against BNS/CrPC sections.")
    
    DEFAULT_SECTIONS = 10
    
    def process_fir_input(text):
        relevant_laws = retrieve_law_sections(text, k=DEFAULT_SECTIONS)
        system_prompt = build_fir_system_prompt(text, relevant_laws, st.session_state.jurisdiction)
        analysis_result = call_gemini_for_fir(system_prompt)
        # Return both analysis result and relevant laws for source tracking
        return analysis_result, relevant_laws
    
    for msg in st.session_state.fir_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "analysis_result" in msg:
                # Get relevant laws for source information (if available)
                relevant_laws = msg.get("relevant_laws", [])
                _render_fir_analysis(msg["analysis_result"], relevant_laws)

    if fir_input := st.chat_input("Enter FIR or case details...", key="fir_input"):
        with st.chat_message("user"):
            st.markdown(fir_input)
        
        st.session_state.fir_messages.append({"role": "user", "content": fir_input})

        with st.spinner(f"Analyzing for relevant BNS/CrPC sections..."):
            analysis_result, relevant_laws = process_fir_input(fir_input)
            st.session_state.fir_messages.append({
                "role": "assistant",
                "content": "FIR analysis complete",
                "analysis_result": analysis_result,
                "relevant_laws": relevant_laws  # Store for source tracking
            })
        
        st.rerun()

# Tab 2: Judgment Analysis (unchanged)
with tab2:
    st.info("Upload or paste the full text of a court judgment for a detailed summary and analysis.")
    
    uploaded_file = st.file_uploader("Upload a judgment document", type=["txt", "md", "pdf"], help="Supports text, markdown, and PDF files.")
    
    if uploaded_file:
        if st.button(f"Analyze '{uploaded_file.name}'"):
            judgment_text = ""
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == ".pdf":
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text: judgment_text += page_text + "\n"
                except Exception as e: st.error(f"Error reading PDF file: {e}")
            else:
                judgment_text = uploaded_file.getvalue().decode("utf-8")
            
            if judgment_text.strip():
                process_judgment_text(judgment_text)
                st.rerun()
            else:
                st.warning("Could not extract text from the uploaded file.")
    
    st.markdown("---")
    
    for i, msg in enumerate(st.session_state.judgment_messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                content = msg["content"]
                if len(content) > USER_INPUT_CHAR_LIMIT:
                    summary = content[:USER_INPUT_CHAR_LIMIT].strip() + "..."
                    with st.expander(f"Your document: *'{summary}'*"):
                        st.markdown(content)
                else:
                    st.markdown(content)
            elif msg["role"] == "assistant":
                mode = msg.get("analysis_mode", "detailed")
                mode_label = "📜 AI Judgment Summary" if mode == "simplified" else "📜 AI Judgment Summary & Analysis"
                is_last_message = (i == len(st.session_state.judgment_messages) - 1)
                with st.expander(f"{mode_label} ({mode.title()} Mode)", expanded=is_last_message):
                    _render_judgment_analysis(msg.get("analysis_result", {}))

    if judgment_input := st.chat_input("Or paste court judgment text here…", key="judgment_input"):
        process_judgment_text(judgment_input)
        st.rerun()

# Tab 3: Updated Document Chat with ChromaDB
with tab3:
    st.title("📄 Police Case Analysis Assistant")
    

    
    if documents_collection:
        doc_count = documents_collection.count()
        # st.success(f"✅ Connected to case documents database ({doc_count} document chunks available)")
        
        # Show collection statistics
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.metric("📄 Document Chunks", doc_count)
        # with col2:
        #     # Get unique documents
        #     try:
        #         all_metadata = documents_collection.get(include=["metadatas"])
        #         unique_docs = len(set(meta.get('document_name', 'Unknown') 
        #                             for meta in all_metadata['metadatas']))
        #         st.metric("📋 Unique Documents", unique_docs)
        #     except:
        #         st.metric("📋 Unique Documents", "N/A")
        # with col3:
        #     st.metric("🔍 Search Ready", "Yes" if doc_count > 0 else "No")
        
        st.markdown("---")
        
        # Display chat history
        for q, a, sources in st.session_state.doc_chat_history:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)
                # Removed the expander for sources
        
                # Sidebar for PDF ingestion
        st.sidebar.title("Document Upload")
        st.sidebar.markdown("Upload your case documents here to analyze them. only applicable for Police Case Analysis Assistant")
        uploaded_pdfs = st.sidebar.file_uploader(
            "Upload PDF documents:", 
            type="pdf", 
            key="sidebar_pdf_upload",
            accept_multiple_files=True
        )

        if uploaded_pdfs:
            if st.sidebar.button("Ingest PDFs"):
                collection = get_documents_collection()
                if collection:
                    success_count = 0
                    total_files = len(uploaded_pdfs)
                    
                    # Create progress containers
                    progress_bar = st.sidebar.progress(0)
                    status_text = st.sidebar.empty()
                    
                    for i, uploaded_pdf in enumerate(uploaded_pdfs):
                        # Update progress
                        progress = (i) / total_files
                        progress_bar.progress(progress)
                        status_text.text(f"Ingesting: {uploaded_pdf.name}")
                        
                        # Ingest document
                        success = ingest_document(collection, uploaded_pdf)
                        if success:
                            success_count += 1
                    
                    # Final progress update
                    progress_bar.progress(1.0)
                    status_text.text("Ingestion complete!")
                    
                    # Show results
                    if success_count == total_files:
                        st.sidebar.success(f"✅ All {total_files} PDFs ingested successfully!")
                    elif success_count > 0:
                        st.sidebar.warning(f"⚠️ {success_count}/{total_files} PDFs ingested successfully")
                    else:
                        st.sidebar.error("❌ Failed to ingest any PDFs")
                else:
                    st.sidebar.error("Could not access database")
        
        # Chat interface
        if question := st.chat_input("Ask about your case documents...", key="doc_chat_input"):
            # Show question immediately
            with st.chat_message("user"):
                st.markdown(question)
            
            # Add placeholder to history
            st.session_state.doc_chat_history.append((question, "", {}))
            
            # Retrieve relevant context and generate response
            with st.spinner("🔍 Searching case documents and generating analysis..."):
                try:
                    # Retrieve relevant chunks
                    retrieval_results = retrieve_case_documents(question, k=5)
                    
                    if retrieval_results and retrieval_results.get('documents'):
                        # Generate contextual response
                        response, source_info = generate_contextual_response(
                            question, 
                            retrieval_results['documents'], 
                            retrieval_results['metadatas']
                        )
                        
                        # Prepare source information for display
                        sources_data = {
                            'sources': source_info,
                            'distances': retrieval_results.get('distances', [])
                        }
                    else:
                        response = "I couldn't find any relevant information in the case documents to answer your question. Please make sure your documents have been properly ingested into the database."
                        sources_data = {}
                    
                    # Update history with actual response
                    st.session_state.doc_chat_history[-1] = (question, response, sources_data)
                    
                except Exception as e:
                    error_response = f"I encountered an error while searching the documents: {str(e)}"
                    st.session_state.doc_chat_history[-1] = (question, error_response, {})
            
            # Show response immediately
            with st.chat_message("assistant"):
                st.markdown(st.session_state.doc_chat_history[-1][1])
            
            st.rerun()
            
    else:
        # No documents collection found
        st.warning("📋 No case documents database found")
        st.info("""
        **To use the Case Analysis Assistant:**
        
        1. **Ingest Documents**: Use the parallel document ingestion script to process your case documents
        2. **Run the ingestion script**: 
           ```bash
           python parallel_ingestion.py
           ```
        3. **Add your documents** to the `documents/` folder (PDF, TXT, MD files)
        4. **Refresh this page** after ingestion is complete
        
        The system will create a `police_vector_db` database with your case documents for searchable analysis.
        """)
        
        # Show ingestion instructions
        with st.expander("🔧 Document Ingestion Instructions", expanded=True):
            st.markdown("""
            ### Step-by-Step Setup:
            
            **1. Prepare Your Documents**
            - Create a `documents/` folder in your project directory
            - Add case files: PDF reports, text statements, evidence logs, etc.
            - Supported formats: `.pdf`, `.txt`, `.md`
            
            **2. Run Document Ingestion**
            ```bash
            python parallel_ingestion.py
            ```
            
            **3. Database Location**
            - Case documents will be stored in: `police_vector_db/`
            - This is separate from the laws database (`fir_vector_db/`)
            - Each document is chunked and embedded for semantic search
            
            **4. Start Analysis**
            - Refresh this page after ingestion completes
            - Ask questions about your case documents
            - Get contextual answers with source citations
            
            ### Example Questions:
            - "What are the key facts mentioned in the police report?"
            - "Who are the witnesses mentioned in the case?"
            - "What evidence was collected?"
            - "Summarize the incident timeline"
            - "What charges are recommended based on the evidence?"
            """)
        
        # Show current database paths for clarity
        with st.expander("🗄️ Database Configuration", expanded=False):
            st.markdown(f"""
            **Database Paths:**
            - **Laws Database**: `{CHROMA_PATH}/` (for BNS/CrPC sections)
            - **Case Documents Database**: `{DOCUMENTS_CHROMA_PATH}/` (for your case files)
            
            **Collections:**
            - **Laws Collection**: `{LAW_COLLECTION_NAME}`
            - **Documents Collection**: `{DOCUMENTS_COLLECTION_NAME}`
            
            Make sure your parallel ingestion script uses the same paths!
            """)
        
        # Quick status check
        if st.button("🔄 Check for Documents Database", type="secondary"):
            st.rerun()