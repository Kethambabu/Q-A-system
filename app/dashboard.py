import streamlit as st
import requests
import pdfplumber
import numpy as np
import tempfile
import os
import sys

# ------------------ FIX PYTHON PATH ------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----------------------------------------------------


from sentence_transformers import SentenceTransformer
from agents.chunker_agent import semantic_chunk
from agents.reranker_agent import rerank
from agents.reasoning_agent import reason
from agents.feedback_agent import needs_refinement
from memory.vector_store import VectorStore

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Resource Intelligence System",
    layout="wide"
)

st.title("📄 AI Resource-Based Question Answering System")

st.markdown("""
Ask questions directly on:
- 🔗 A web link
- 📄 A PDF document

Uses **Neural Retrieval + Cross-Encoder Reranking + LLM Reasoning**
""")

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    return embed_model

embed_model = load_models()

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")
top_k = st.sidebar.slider("Top-K Retrieval", 3, 15, 8)
show_debug = st.sidebar.checkbox("Show Debug Info", True)

# ------------------ RESOURCE INPUT ------------------
st.subheader("📥 Provide Resource")

resource_type = st.radio(
    "Choose resource type:",
    ["PDF Upload", "Link (Web / PDF)"]
)

raw_text = ""

if resource_type == "PDF Upload":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                raw_text += page.extract_text() or ""

elif resource_type == "Link (Web / PDF)":
    url = st.text_input("Enter URL")

    if url:
        if url.endswith(".pdf"):
            response = requests.get(url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    raw_text += page.extract_text() or ""

            os.remove(tmp_path)

        else:
            response = requests.get(url)
            raw_text = response.text

# ------------------ QUESTION INPUT ------------------
st.subheader("❓ Ask Your Question")
query = st.text_input(
    "Enter your question",
    placeholder="e.g. Explain the attention mechanism discussed in this document"
)

# ------------------ PROCESS PIPELINE ------------------
if st.button("🚀 Run QA Pipeline"):

    if not raw_text.strip():
        st.error("Please provide a valid resource.")
        st.stop()

    if not query.strip():
        st.error("Please enter a question.")
        st.stop()

    with st.spinner("Processing resource and answering..."):

        # 1️⃣ Chunk the document
        chunks = semantic_chunk(raw_text)

        # 2️⃣ Embed chunks
        embeddings = embed_model.encode(chunks)

        # 3️⃣ Create session vector store
        vector_store = VectorStore(dim=embeddings.shape[1])

        metadata = [{"text": c} for c in chunks]
        vector_store.add(embeddings, metadata)

        # 4️⃣ Embed query
        q_embedding = embed_model.encode([query])

        # 5️⃣ Retrieve
        scores, ids = vector_store.search(np.array(q_embedding), top_k)
        retrieved_chunks = [metadata[i]["text"] for i in ids[0]]

        # 6️⃣ Rerank
        reranked = rerank(query, retrieved_chunks)
        context = "\n\n".join([p[0] for p in reranked[:3]])

        # 7️⃣ Reasoning
        answer = reason(context, query)

        # 8️⃣ Feedback loop
        refined = False
        if needs_refinement(answer):
            refined = True
            context += "\n\nProvide a more detailed explanation."
            answer = reason(context, query)

    # ------------------ OUTPUT ------------------
    st.subheader("🧠 Answer")
    st.success(answer)

    if refined:
        st.warning("Answer was refined due to low confidence.")

    # ------------------ DEBUG INFO ------------------
    if show_debug:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Retrieved Chunks")
            for i, chunk in enumerate(retrieved_chunks[:5]):
                with st.expander(f"Chunk {i+1}"):
                    st.write(chunk[:800])

        with col2:
            st.subheader("🎯 Reranked Passages")
            for i, (text, score) in enumerate(reranked[:5]):
                with st.expander(f"Rank {i+1} | Score: {round(score,3)}"):
                    st.write(text[:800])
