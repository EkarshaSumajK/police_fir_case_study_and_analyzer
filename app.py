# app.py
import streamlit as st
import google.generativeai as genai
import chromadb
import os
import json
from dotenv import load_dotenv
import PyPDF2

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

LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
CHROMA_PATH = "fir_vector_db"
LAW_COLLECTION_NAME = "laws_collection"
LLM_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 65535
USER_INPUT_CHAR_LIMIT = 400

@st.cache_resource(show_spinner=False)
def _get_law_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(LAW_COLLECTION_NAME)

try:
    law_collection = _get_law_collection()
except Exception as e:
    st.error(f"🚨 Failed to connect to the Vector Database. Have you run `load_law_corpus.py`? Error: {e}", icon="🚨")
    st.stop()

if "fir_messages" not in st.session_state: st.session_state.fir_messages = []
if "judgment_messages" not in st.session_state: st.session_state.judgment_messages = []
if "jurisdiction" not in st.session_state: st.session_state.jurisdiction = "Telangana"
if "top_k_laws" not in st.session_state: st.session_state.top_k_laws = 5

# --- COMMON UTILITY FUNCTIONS ---
@st.cache_data(show_spinner=False)
def translate_to_te(text: str) -> str:
    """Translate text to Telugu using Gemini. Preserves legal meaning and formatting."""
    text = (text or "").strip()
    if not text: return ""
    try:
        model = genai.GenerativeModel(LLM_MODEL_NAME, generation_config={"response_mime_type": "text/plain", "temperature": 0.0, "max_output_tokens": 2048})
        resp = model.generate_content(f"Translate to Telugu. Preserve legal meaning, names, and formatting. Only output the translation:\n\n{text}")
        return (getattr(resp, "text", "") or "").strip()
    except Exception: return text

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
    """Embed query text for FIR analysis."""
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text, task_type="RETRIEVAL_QUERY")
        return result["embedding"]
    except Exception as e: st.error(f"Failed to embed query text: {e}"); return None

@st.cache_data(show_spinner=False)
def retrieve_law_sections(query_text, k=5):
    """Retrieve relevant law sections for FIR analysis."""
    embedding = _embed_query_text(query_text)
    if embedding is None: return []
    results = law_collection.query(query_embeddings=[embedding], n_results=k, include=["metadatas"])
    return results['metadatas'][0] if results and results.get('metadatas') else []

# --- FIR Analysis Prompt ---
def build_fir_system_prompt(fir_summary, relevant_laws, jurisdiction):
    """Constructs the system prompt for analyzing an FIR."""
    laws_context = "\n".join([f"- Section {law['section_number']} ({law['section_name']}): {law['full_text']}" for law in relevant_laws]) if relevant_laws else "No relevant laws found."
    
    json_schema = {
        "sections": [{
            "section": "BNS/CrPC Section Number (e.g., BNS 420 or CrPC 41)",
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

    prompt = f"""You are an expert AI legal assistant for Indian Police. Analyze the case details and recommend BNS/CrPC sections based ONLY on the provided context. Provide rationale, verbatim quotes, and procedural steps. Your output MUST be a single, valid JSON object conforming EXACTLY to the schema.

Potentially Relevant Law Sections:
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
"""
    return prompt

# --- FIR Render Function ---
def _render_fir_analysis(res: dict):
    """Renders the structured JSON output for FIR analysis."""
    if not res: st.info("No analysis available."); return
    st.subheader("✅ Recommended BNS/CrPC Sections")  # Updated
    if res.get("sections"):
        for sec in res["sections"]:
            law = sec.get("law_citation", {}) or {}
            section_label = sec.get("section", "N/A").replace("IPC", "BNS")  # Updated
            st.markdown(f"<div class='card'><strong>Section:</strong> <code>{section_label}</code>", unsafe_allow_html=True)
            
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
        "facts_of_the_case_en": "Key facts of the case presented in English.",
        "facts_of_the_case_te": "Key facts of the case in Telugu.",
        "parties_arguments": {
            "petitioner_en": "Summary of petitioner/appellant arguments.",
            "petitioner_te": "Summary in Telugu.",
            "respondent_en": "Summary of respondent arguments.",
            "respondent_te": "Summary in Telugu."
        },
        "key_legal_issues_en": ["List of legal questions addressed by the court in English."],
        "key_legal_issues_te": ["List of legal questions in Telugu."],
        "final_verdict_en": "The final decision or ruling of the court in English.",
        "final_verdict_te": "The final decision in Telugu.",
        "legal_principles_cited": [{
            "principle": "Name or description of the legal principle or precedent.",
            "citation": "Case law or statute cited."
        }]
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
    """Renders the structured JSON output for judgment analysis."""
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
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.subheader("📖 Case Details")
    col_d_en, col_d_te = st.columns(2)
    with col_d_en:
        st.markdown("**Facts of the Case (English)**")
        st.info(res.get("facts_of_the_case_en", "N/A"))
        if args := res.get("parties_arguments"):
            st.markdown("**Petitioner/Appellant Arguments**")
            st.warning(args.get("petitioner_en", "N/A"))
            st.markdown("**Respondent Arguments**")
            st.warning(args.get("respondent_en", "N/A"))
    with col_d_te:
        st.markdown("**కేసు వాస్తవాలు (తెలుగు)**")
        st.info(_fallback_te(res.get("facts_of_the_case_te"), res.get("facts_of_the_case_en")))
        if args := res.get("parties_arguments"):
            st.markdown("**పిటిషనర్/అప్పీలుదారు వాదనలు**")
            st.warning(_fallback_te(args.get("petitioner_te"), args.get("petitioner_en")))
            st.markdown("**ప్రతివాది వాదనలు**")
            st.warning(_fallback_te(args.get("respondent_te"), args.get("respondent_en")))

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("⚖️ Legal Analysis & Verdict")
    col_v_en, col_v_te = st.columns(2)
    with col_v_en:
        st.markdown("**Key Legal Issues (English)**")
        for issue in res.get("key_legal_issues_en", []):
            st.markdown(f"- {issue.replace('IPC', 'BNS')}")  # Updated
        st.markdown("**Final Verdict**")
        st.success(res.get("final_verdict_en", "N/A"))
    with col_v_te:
        st.markdown("**కీలక చట్టపరమైన సమస్యలు (తెలుగు)**")
        issues_te = res.get("key_legal_issues_te", []) or [translate_to_te(i) for i in res.get("key_legal_issues_en", [])]
        for issue in issues_te:
            st.markdown(f"- {issue.replace('IPC', 'BNS')}")  # Updated
        st.markdown("**తుది తీర్పు**")
        st.success(_fallback_te(res.get("final_verdict_te"), res.get("final_verdict_en")))

    if principles := res.get("legal_principles_cited"):
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.subheader("Cited Legal Principles & Precedents")
        for p in principles:
            principle = p.get('principle', 'N/A').replace('IPC', 'BNS')  # Updated
            citation = p.get('citation', 'N/A').replace('IPC', 'BNS')  # Updated
            st.markdown(f"- **Principle:** {principle}")
            st.caption(f"Citation: {citation}")
            
    st.markdown("</div>", unsafe_allow_html=True)

def process_judgment_text(judgment_text: str):
    """Main workflow for processing judgment text from any source."""
    st.session_state.judgment_messages.append({"role": "user", "content": judgment_text})
    with st.spinner("📜 Analyzing Judgment... This may take a moment."):
        system_prompt = build_judgment_prompt(judgment_text)
        analysis_result = call_gemini_for_judgment(system_prompt)
    st.session_state.judgment_messages.append({"role": "assistant", "analysis_result": analysis_result or {}})

# --- STREAMLIT UI LAYOUT ---
st.title("⚖️ Police AI Analysis Assistant")
st.warning("**Disclaimer:** For internal police use only. All outputs must be verified by a qualified officer. Do not include sensitive PII.", icon="⚠️")
tab1, tab2 = st.tabs(["⚖️ FIR Analysis", "📜 Judgment Analysis"])

# --- TAB 1: FIR ANALYSIS ---
with tab1:
    st.info("Paste FIRs to analyze against BNS/CrPC sections.")  # Updated
    
    for i, msg in enumerate(st.session_state.fir_messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                content = msg["content"]
                if len(content) > USER_INPUT_CHAR_LIMIT:
                    summary = content[:USER_INPUT_CHAR_LIMIT].strip() + "..."
                    with st.expander(f"You asked: *'{summary}'*"):
                        st.markdown(content)
                else:
                    st.markdown(content)
            elif msg["role"] == "assistant":
                num_sections = len(msg.get("analysis_result", {}).get("sections", []))
                label = f"💡 AI Analysis: {num_sections} BNS/CrPC Section(s) Recommended" if num_sections > 0 else "💡 AI Analysis"  # Updated from IPC to BNS
                is_last_message = (i == len(st.session_state.fir_messages) - 1)
                with st.expander(label, expanded=is_last_message):
                    _render_fir_analysis(msg.get("analysis_result", {}))

    if fir_input := st.chat_input("Enter case details to analyze…", key="fir_input"):
        st.session_state.fir_messages.append({"role": "user", "content": fir_input})
        with st.spinner("🧠 Analyzing FIR... This may take a moment."):
            relevant_laws = retrieve_law_sections(fir_input, k=st.session_state.top_k_laws)
            if not relevant_laws:
                st.warning("No relevant laws found for the given input. Try rephrasing or expanding the case details.")
                st.session_state.fir_messages.append({"role": "assistant", "analysis_result": {}})
                st.rerun()

            system_prompt = build_fir_system_prompt(fir_input, relevant_laws, st.session_state.jurisdiction)
            analysis_result = call_gemini_for_fir(system_prompt)
            
            if not analysis_result:
                st.error("Failed to generate analysis. Please try again or check the input.")
                st.session_state.fir_messages.append({"role": "assistant", "analysis_result": {}})
            else:
                st.session_state.fir_messages.append({"role": "assistant", "analysis_result": analysis_result})
            
            st.rerun()

# --- TAB 2: JUDGMENT ANALYSIS ---
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
                label = "📜 AI Judgment Summary & Analysis"
                is_last_message = (i == len(st.session_state.judgment_messages) - 1)
                with st.expander(label, expanded=is_last_message):
                    _render_judgment_analysis(msg.get("analysis_result", {}))

    if judgment_input := st.chat_input("Or paste court judgment text here…", key="judgment_input"):
        process_judgment_text(judgment_input)
        st.rerun()