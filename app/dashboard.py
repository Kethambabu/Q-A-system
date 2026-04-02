# app/dashboard.py
"""
Streamlit Dashboard — AI Resource-Based Q&A System
Powered by LangGraph Multi-Agent Orchestration.
Uses Tree-Based Retrieval with LLM Reasoning (no vectors, no embeddings).
"""
import streamlit as st
import os
import sys

# ------------------ FIX PYTHON PATH ------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----------------------------------------------------

from core.pipeline import run_pipeline, build_and_save_tree

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Resource Intelligence System",
    layout="wide"
)

if "tree_built" not in st.session_state:
    st.session_state.tree_built = False
if "current_resource" not in st.session_state:
    st.session_state.current_resource = None

st.title("📄 AI Resource-Based Question Answering System")

st.markdown("""
Ask questions directly on:
- 🔗 A web link
- 📄 A PDF document

Uses **LangGraph Multi-Agent Pipeline** with **Tree-Based Retrieval + LLM Reasoning** — no embeddings, no vector DB.
""")

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")
show_debug = st.sidebar.checkbox("Show Debug Info", True)

# ------------------ RESOURCE INPUT ------------------
st.subheader("📥 Provide Resource")

resource_type = st.radio(
    "Choose resource type:",
    ["PDF Upload", "Link (Web / PDF)"]
)

if resource_type == "PDF Upload":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if st.session_state.current_resource != uploaded_file.name:
            st.session_state.tree_built = False
            st.session_state.current_resource = uploaded_file.name

        if not st.session_state.tree_built:
            with st.spinner("🌳 Building document tree via LangGraph..."):
                try:
                    # Directly pass PDF bytes into the LangGraph build pipeline
                    tree_info = build_and_save_tree(pdf_bytes=uploaded_file.read())
                    st.session_state.tree_built = True
                except Exception as e:
                    st.error(f"Failed to build tree: {e}")

        if st.session_state.tree_built:
            st.success("✅ Tree built successfully! You can now ask multiple questions.")

elif resource_type == "Link (Web / PDF)":
    url = st.text_input("Enter URL")

    if url:
        if st.session_state.current_resource != url:
            st.session_state.tree_built = False
            st.session_state.current_resource = url

        if not st.session_state.tree_built:
            with st.spinner("🌳 Building document tree via LangGraph..."):
                try:
                    # Directly pass URL into the LangGraph build pipeline
                    tree_info = build_and_save_tree(url=url)
                    st.session_state.tree_built = True
                except Exception as e:
                    st.error(f"Failed to fetch resource from URL: {e}")

        if st.session_state.tree_built:
            st.success("✅ Tree built successfully! You can now ask multiple questions.")

# ------------------ QUESTION INPUT ------------------
st.subheader("❓ Ask Your Question")
query = st.text_input(
    "Enter your question",
    placeholder="e.g. Explain the attention mechanism discussed in this document"
)

# ------------------ PROCESS PIPELINE ------------------
if st.button("🚀 Run QA Pipeline"):

    if not st.session_state.tree_built:
        st.error("Please provide a valid resource and wait for the document index to build.")
        st.stop()

    if not query.strip():
        st.error("Please enter a question.")
        st.stop()

    with st.spinner("🤖 Running LangGraph Query Pipeline..."):
        result = run_pipeline(query)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    # ------------------ QUERY REFINEMENT INFO ------------------
    refinement_info = result.get("refinement_info", {})
    if refinement_info.get("method") in ("fast_path", "llm"):
        st.info(f"🔍 **Query refined**: {refinement_info.get('refined_query', '')}")

    # ------------------ OUTPUT ------------------
    st.subheader("🧠 Answer")
    
    if "is not available in the provided document" in result.get("answer", ""):
        st.error("❌ Answer not found in the uploaded document.")
    else:
        st.success(result.get("summary", ""))

        with st.expander("🔎 Show full answer"):
            st.write(result.get("answer", ""))

    if result.get("refined"):
        st.warning("Answer was refined due to low confidence.")

    # ------------------ TREE & TRAVERSAL INFO ------------------
    tree_info = result.get("tree_info", {})
    trav_info = result.get("traversal_info", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌳 Tree Nodes", tree_info.get("total_nodes", "N/A"))
    with col2:
        st.metric("📏 Tree Depth", tree_info.get("tree_depth", "N/A"))
    with col3:
        st.metric("🎯 Nodes Retrieved", len(trav_info.get("retrieved_node_ids", [])))

    # ------------------ DEBUG INFO ------------------
    if show_debug:
        st.subheader("🔍 Debug Information")

        # Query refinement details
        if refinement_info:
            with st.expander("🔄 Query Refinement Details"):
                st.markdown(f"**Original Query:** {refinement_info.get('original_query', 'N/A')}")
                st.markdown(f"**Refined Query:** {refinement_info.get('refined_query', 'N/A')}")
                st.markdown(f"**Method:** `{refinement_info.get('method', 'N/A')}`")
                if trav_info.get("search_query_used"):
                    st.markdown(f"**Search Query Used for Traversal:** {trav_info['search_query_used']}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🧭 Traversal Path**")
            for step in trav_info.get("traversal_path", []):
                depth_indicator = "  " * step["depth"] + "↳ " if step["depth"] > 0 else "🌱 "
                action = step.get("action", "")
                node_id = step.get("node_id", "")
                boost = " 🔢" if step.get("numerical_boost") else ""
                st.text(f"{depth_indicator}[{node_id}] {action}{boost}")

        with col2:
            st.markdown("**📄 Retrieved Node IDs**")
            for nid in trav_info.get("retrieved_node_ids", []):
                st.code(nid)

        with st.expander("📋 Context Sent to LLM"):
            st.write(result.get("context_used", ""))

        with st.expander("🌲 Full Traversal Details (JSON)"):
            st.json(trav_info)
