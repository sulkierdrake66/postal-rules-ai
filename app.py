import streamlit as st
import os
import re
from glob import glob
from pypdf import PdfReader
from google import genai
from google.genai import types

st.set_page_config(page_title="Postal Rules Reference", page_icon="📮", layout="wide")
st.title("📮 Postal Department Orders & Rules Agent")

# 1. Handle API Key
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.info("👈 Please enter your Gemini API key in the sidebar to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Text extraction with layout normalization
@st.cache_data(show_spinner=False)
def load_all_pdfs():
    pdf_files = glob("*.pdf") + glob("**/*.pdf", recursive=True)
    pages_data = []
    
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            filename = os.path.basename(pdf_path)
            for page_num, page in enumerate(reader.pages):
                # extraction_mode="layout" preserves paragraphs and columns much better
                text = page.extract_text(extraction_mode="layout") or ""
                # Clean up excess whitespace and broken newline strings
                cleaned_text = re.sub(r'[ \t]+', ' ', text)
                cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
                
                if cleaned_text:
                    pages_data.append({
                        "file": filename,
                        "page": page_num + 1,
                        "text": cleaned_text
                    })
        except Exception as e:
            st.error(f"Error reading {pdf_path}: {e}")
            
    return pages_data

with st.spinner("Indexing departmental PDFs..."):
    all_pages = load_all_pdfs()

if not all_pages:
    st.warning("⚠️ No PDF files detected in your repository.")
    st.stop()

st.sidebar.success(f"Indexed {len(all_pages)} pages from your documents.")

# 3. Context retrieval
def retrieve_relevant_pages(query, pages, top_k=4):
    terms = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
    if not terms:
        return pages[:top_k]
    
    scored = []
    for page in pages:
        content_lower = page["text"].lower()
        score = sum(content_lower.count(term) * 2 for term in terms)
        
        # Boost matches that mention formal rule keywords
        if "rule" in content_lower or "order" in content_lower or "clause" in content_lower:
            score += 1
            
        if score > 0:
            scored.append((score, page))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [item[1] for item in scored[:top_k]]
    return results if results else pages[:top_k]

# 4. Search and structured answering
query = st.text_input("🔍 Ask a question about a departmental rule, limit, or procedure:")

if query:
    relevant_pages = retrieve_relevant_pages(query, all_pages, top_k=4)
    
    with st.spinner("Extracting verified rule details..."):
        context_blocks = []
        for doc in relevant_pages:
            context_blocks.append(f"--- Document: {doc['file']} (Page {doc['page']}) ---\n{doc['text']}")
        
        filtered_context = "\n\n".join(context_blocks)

        system_instruction = (
            "You are an expert, precise departmental assistant for India Post rules, manuals, and circulars.\n\n"
            "Respond strictly in this standardized markdown layout:\n"
            "### 1. Direct Answer\n"
            "A concise, direct 1-3 sentence summary answering the question.\n\n"
            "### 2. Departmental Citation\n"
            "- **Rule / Order / Clause:** [Cite exact Rule number, volume, or SB/DG order date]\n"
            "- **Source Document:** [Document Name, Page Number]\n\n"
            "### 3. Verbatim / Core Provisions\n"
            "Bullet points summarizing the operational condition, limit, or procedure exactly as laid out in the text.\n\n"
            "STRICT RULES:\n"
            "- If the provided text does not contain the answer, simply write: "
            "'This specific rule or procedure is not found in the indexed pages.'\n"
            "- Never hallucinate rule numbers, figures, or dates."
        )

        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    f"Context Documents:\n{filtered_context}\n\nQuestion: {query}"
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                ),
            )
            
            st.markdown(response.text)
            
            with st.expander("📄 View Source Page Excerpts Sent to AI"):
                for page in relevant_pages:
                    st.markdown(f"**{page['file']}** (Page {page['page']})")
                    st.caption(page['text'][:500] + "...")
            
        except Exception as e:
            st.error(f"Error: {e}")
