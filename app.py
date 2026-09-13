import streamlit as st
import os
from glob import glob
from pypdf import PdfReader
from google import genai
from google.genai import types

st.set_page_config(page_title="Postal Rules & Orders AI", page_icon="📮", layout="wide")
st.title("📮 Postal Department Orders & Rules Reference")

# 1. Handle API Key securely
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.info("👈 Please enter your free Gemini API key in the sidebar to begin.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

# 2. Extract text from uploaded PDFs
@st.cache_data(show_spinner=False)
def load_all_pdfs():
    pdf_files = glob("*.pdf") + glob("**/*.pdf", recursive=True)
    all_chunks = []
    
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    filename = os.path.basename(pdf_path)
                    all_chunks.append({
                        "file": filename,
                        "page": page_num + 1,
                        "text": text.strip()
                    })
        except Exception as e:
            st.error(f"Error reading {pdf_path}: {e}")
            
    return all_chunks

with st.spinner("Indexing uploaded departmental PDFs..."):
    doc_database = load_all_pdfs()

if not doc_database:
    st.warning("⚠️ No PDF files detected in your repository yet. Upload some postal PDFs to start asking questions.")
    st.stop()

st.sidebar.success(f"Loaded {len(doc_database)} pages from your PDFs.")

# 3. User Query Box
query = st.text_input("🔍 Ask a question about a departmental rule, limit, or procedure:")

if query:
    with st.spinner("Searching records and formulating rule citation..."):
        # Combine context with clear file and page labels
        context_blocks = []
        for doc in doc_database:
            context_blocks.append(f"--- Document: {doc['file']} | Page: {doc['page']} ---\n{doc['text']}")
        
        full_context = "\n\n".join(context_blocks)

        system_instruction = (
            "You are an authoritative assistant for India Post departmental rules and orders.\n"
            "Your job is to answer queries strictly using the provided documents.\n"
            "MANDATORY INSTRUCTIONS:\n"
            "1. You MUST explicitly state the exact Rule Number, Section, Manual volume, or Order Number/Date whenever mentioned in the text.\n"
            "2. State the exact Document Name and Page Number where the rule is located.\n"
            "3. If the answer cannot be determined strictly from the provided text, state: "
            "'This specific rule or order is not found in the currently uploaded departmental documents.' "
            "Never guess, extrapolate, or hallucinate rule numbers.\n"
        )

        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    f"Context Documents:\n{full_context}\n\nUser Question: {query}"
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0, # Zero temperature guarantees strictly factual, deterministic output
                ),
            )
            
            st.subheader("Official Citation & Answer:")
            st.write(response.text)
            
        except Exception as e:
            st.error(f"API Error: {e}")
