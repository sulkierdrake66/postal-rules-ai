import streamlit as st
import os
import re
from glob import glob
from pypdf import PdfReader
from google import genai
from google.genai import types

st.set_page_config(page_title="Postal Rules AI", page_icon="📮", layout="wide")
st.title("📮 Postal Department Orders & Rules Reference")

# 1. API Key handling
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.info("👈 Please enter your Gemini API key in the sidebar to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Extract & cache text page-by-page
@st.cache_data(show_spinner=False)
def load_all_pdfs():
    pdf_files = glob("*.pdf") + glob("**/*.pdf", recursive=True)
    pages_data = []
    
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            filename = os.path.basename(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({
                        "file": filename,
                        "page": page_num + 1,
                        "text": text.strip()
                    })
        except Exception as e:
            st.error(f"Error reading {pdf_path}: {e}")
            
    return pages_data

with st.spinner("Indexing departmental PDFs..."):
    all_pages = load_all_pdfs()

if not all_pages:
    st.warning("⚠️ No PDF files detected in your repository yet.")
    st.stop()

st.sidebar.success(f"Indexed {len(all_pages)} pages from your PDFs.")

# 3. Simple & efficient keyword-scoring retrieval (keeps token count well under 250k)
def retrieve_relevant_pages(query, pages, top_k=5):
    # Extract search terms (ignore very short words)
    words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
    if not words:
        return pages[:top_k]
    
    scored = []
    for page in pages:
        score = 0
        content_lower = page["text"].lower()
        for word in words:
            # Count keyword hits on this specific page
            score += content_lower.count(word)
        if score > 0:
            scored.append((score, page))
            
    # Sort by relevance score
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [item[1] for item in scored[:top_k]]
    
    # Fallback if no exact keyword matched
    return results if results else pages[:top_k]

# 4. Search and Question Answering
query = st.text_input("🔍 Ask a question about a departmental rule, limit, or procedure:")

if query:
    relevant_pages = retrieve_relevant_pages(query, all_pages, top_k=5)
    
    with st.spinner("Searching relevant rule pages..."):
        context_blocks = []
        for doc in relevant_pages:
            context_blocks.append(f"--- Document: {doc['file']} | Page: {doc['page']} ---\n{doc['text']}")
        
        filtered_context = "\n\n".join(context_blocks)

        system_instruction = (
            "You are an authoritative assistant for India Post departmental rules and orders.\n"
            "Answer queries strictly using the provided excerpts.\n"
            "MANDATORY INSTRUCTIONS:\n"
            "1. You MUST explicitly cite the exact Rule Number, Section, Volume, or Circular/Order Number and Date if present in the text.\n"
            "2. State the Document Name and Page Number for your citation.\n"
            "3. If the answer cannot be determined strictly from the provided text, state: "
            "'This specific rule or order is not found in the indexed excerpts.' "
            "Never hallucinate rule numbers."
        )

        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[
                    f"Context:\n{filtered_context}\n\nQuestion: {query}"
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                ),
            )
            
            st.subheader("Official Citation & Answer:")
            st.write(response.text)
            
            with st.expander("📄 View Matched Source Pages"):
                for page in relevant_pages:
                    st.markdown(f"**{page['file']}** (Page {page['page']})")
                    st.caption(page['text'][:400] + "...")
            
        except Exception as e:
            st.error(f"API Error: {e}")
