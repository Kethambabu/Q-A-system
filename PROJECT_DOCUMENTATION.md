# AI Resource-Based Question Answering System - Project Documentation

This document provides an in-depth explanation of the Q-A system. The project is designed as an end-to-end **Tree-Based Retrieval** pipeline with a multi-agent architectural style. It ingests resources (Web Links or PDFs), builds a hierarchical document tree using LLM reasoning, navigates the tree to find relevant content, and generates reliable answers — **without embeddings, vectors, or similarity search**.

---

## 🏗️ Architecture Overview

```
PDF / URL
    ↓
Ingestion Agent (text extraction)
    ↓
Tree Builder Agent (LLM-based structuring)
    ↓
Tree Index (Hierarchical JSON)
    ↓
Tree Traversal Agent (LLM-driven navigation)
    ↓
Relevant Nodes
    ↓
Reasoning Agent (LLM answer generation)
    ↓
Feedback + Summarization
    ↓
Streamlit Dashboard
```

The system comprises several modular layers:
1. **Application/UI Layer (`app/`)**: Handles user interactions, file uploads, link inputs, and displays answers.
2. **Agents Layer (`agents/`)**: Specialized Python modules — each acting as an "agent" for a distinct pipeline stage (ingestion, tree building, tree traversal, reasoning, summarization).
3. **Memory/Storage Layer (`memory/`)**: JSON-based hierarchical tree storage with traversal utilities.
4. **Orchestration Layer (`orchestration/`)**: Wires agents into a cohesive directed workflow.

---

## 🧩 Detailed Module Breakdown

### 1. User Interface (`app/dashboard.py`)
The main entry point, built using **Streamlit**.
- **Resource Input**: Upload a PDF document or provide a Web URL.
- **Processing**: Extracts text, builds document tree, traverses it, and generates an answer.
- **Pipeline Execution**: Calls `orchestration.graph.run_pipeline()` for the full tree-based Q&A flow.
- **Debug Info**: Shows tree metrics (nodes, depth), traversal path, retrieved node IDs, and context sent to the LLM.

### 2. Agents (`agents/`)

- **`ingestion_agent.py`**: Handles raw file parsing. `ingest_pdf_bytes()` extracts text with `pdfplumber`; `ingest_html()` parses HTML with `BeautifulSoup`.
- **`tree_builder_agent.py`**: The core indexing agent. Uses LLM (ChatGroq / Llama-3.1) to analyze document text and produce a hierarchical JSON tree. Handles long documents via segmentation. Each node has: `node_id`, `title`, `summary`, `content`, `children`.
- **`tree_traversal_agent.py`**: The core retrieval agent. Navigates the tree top-down using LLM reasoning — at each level, it presents child summaries to the LLM with the user query and selects the most relevant branches. Recurses until leaf nodes.
- **`reasoning_agent.py`**: Uses ChatGroq (Llama-3.1-8b-instant) to synthesize context from retrieved tree nodes into a multi-step reasoned answer.
- **`feedback_agent.py`**: Heuristic evaluator checking if the answer needs refinement.
- **`summarizer_agent.py`**: ChatGroq LLM call to condense the answer into a concise summary (under 80 words).

### 3. Memory & Storage (`memory/tree_store.py`)
- Replaces the previous FAISS vector store.
- Saves/loads hierarchical JSON trees.
- Provides O(1) node lookup via a flat index.
- Traversal helpers: children summaries, leaf content collection, depth calculation.
- Tree files stored in `memory/trees/`.

### 4. Orchestration (`orchestration/graph.py`)
- `run_pipeline(resource_text, query)` — the full pipeline:
  1. Build tree from document text (LLM-based)
  2. Store tree in TreeStore
  3. Traverse tree with LLM reasoning
  4. Collect content from relevant nodes
  5. Generate answer with reasoning LLM
  6. Refine if needed, summarize



## 🚀 The End-to-End Workflow

1. **Input Phase**: User inputs text via PDF or URL. Text is extracted.
2. **Tree Building**: LLM analyzes the document and creates a hierarchical JSON tree with sections, sub-sections, and leaf nodes.
3. **Tree Storage**: Tree is saved as JSON in `memory/trees/` for persistence and lookup.
4. **Tree Traversal**: User query is compared against tree node summaries using LLM reasoning. The system navigates top-down, selecting the most relevant branches at each level.
5. **Context Assembly**: Content from the selected leaf nodes is collected and combined.
6. **Reasoning Synthesis**: The context is fed to Llama-3.1 (via LangChain/Groq) to construct a comprehensive answer.
7. **Refinement Loop**: The `feedback_agent` evaluates the answer. If inadequate, the model is re-prompted.
8. **Summarization**: The detailed answer is shortened into a concise paragraph.
9. **UI Rendering**: The summary, full answer, tree metrics, traversal path, and debug logs are displayed on the Streamlit dashboard.

---

## ⚙️ Environment and Requirements

- **LLM**: Groq API (`langchain-groq`) for Llama-3.1-8b-instant — used for tree building, traversal, reasoning, and summarization.
- **No local models**: No embeddings, no vector DB, no GPU required.
- **Streamlit**: Interactive web dashboard.
- **Lightweight**: Minimal dependencies (~200MB install vs. previous ~4GB).

---

## 📁 Project Structure

```
Q-A-system-main/
├── agents/
│   ├── __init__.py
│   ├── ingestion_agent.py        # PDF/HTML text extraction
│   ├── tree_builder_agent.py     # LLM-based tree construction
│   ├── tree_traversal_agent.py   # LLM-driven tree navigation
│   ├── reasoning_agent.py        # LLM answer generation
│   ├── feedback_agent.py         # Answer quality gate
│   └── summarizer_agent.py       # LLM summarization
├── app/
│   ├── __init__.py
│   └── dashboard.py              # Streamlit UI
├── memory/
│   ├── __init__.py
│   ├── tree_store.py             # JSON tree persistence & lookup
│   └── trees/                    # Stored tree index files
├── orchestration/
│   ├── __init__.py
│   └── graph.py                  # Pipeline orchestrator
├── .env                          # API keys
├── requirements.txt              # Dependencies
└── PROJECT_DOCUMENTATION.md      # This file
```
