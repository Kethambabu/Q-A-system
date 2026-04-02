# PDF Upload & Processing Pipeline

This document explains the end-to-end pipeline executed when a user uploads a PDF file into the **AI Resource-Based Question Answering System**. The system is orchestrated by **LangGraph**, a multi-agent state-graph framework, and uses a strictly hierarchical *vector-less Tree-Based* retrieval strategy driven natively by Large Language Models (LLMs).

---

## Architecture Overview

The pipeline is split into two distinct **LangGraph execution phases**:

| Phase | Trigger | Graph Path |
|-------|---------|------------|
| **Build** | PDF upload / URL submit | `input_agent → tree_builder_agent → tree_store_agent → END` |
| **Query** | User asks a question | `tree_store_agent → reasoning_agent → answer_agent → END` |

The tree is built **once** per document. All subsequent questions reuse the persisted tree via the query phase.

---

## Technologies Used

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | `langgraph` (StateGraph) | Multi-agent workflow graph with conditional routing |
| **LLM Provider** | `langchain-groq` (Llama-3.1-8b-instant) | Summarization, hierarchy construction, traversal, reasoning |
| **PDF Extraction** | `PyMuPDF` (fitz) | Block-level layout-aware text extraction |
| **HTML Parsing** | `beautifulsoup4` | Web page text extraction |
| **UI Framework** | `streamlit` | Interactive web dashboard |
| **Configuration** | `python-dotenv` | Environment variable management (API keys) |
| **Concurrency** | `concurrent.futures.ThreadPoolExecutor` | Parallel segment summarization |
| **Persistence** | Native Python `json` | Tree index serialization to disk |

---

## LangGraph Agent Nodes

### 1. `input_agent` — Resource Ingestion
**Technology:** `PyMuPDF (fitz)`, `beautifulsoup4`, `requests`

- Accepts raw `pdf_bytes` or a `url` from the dashboard.
- For PDFs: uses `fitz.open()` with block-level extraction (`page.get_text("blocks")`) to produce a pre-segmented `List[str]` of clean text chunks (100–2000 chars each).
- For URLs: detects content type — routes PDFs through fitz, HTML through BeautifulSoup.
- Cleans text: fixes hyphenated line breaks (`inter-\nnational` → `international`), normalizes whitespace.

### 2. `tree_builder_agent` — Hierarchical Index Construction
**Technology:** `langchain-groq`, `concurrent.futures`

- Receives segments from `input_agent`.
- **Parallel Summarization:** Uses `ThreadPoolExecutor(max_workers=5)` to send each segment to the LLM concurrently, producing a `title` + `summary` per chunk (~700 tokens/call).
- **Hierarchy Generation:** A single LLM call groups all summaries into a parent-child tree structure (~1300 tokens/call).
- **Content Binding:** Attaches original full-text segments to leaf nodes without any LLM call.
- **Max tokens per LLM call:** ~1300 (well under 8k limit).

### 3. `tree_store_agent` — Memory Persistence
**Technology:** Python `json`, OS file handlers

- **Build phase:** Serializes the tree to `memory/trees/tree_index.json`.
- **Query phase:** Loads the persisted tree from disk for sub-second recall.
- Provides node count, tree depth, and content lookup APIs.

### 4. `reasoning_agent` — Tree Traversal & Context Retrieval
**Technology:** `langchain-groq`

- Traverses the tree top-down using LLM reasoning on node summaries.
- Selects relevant branches without reading full content — only summaries.
- Collects leaf node content from matched branches.
- Caps combined context at 24,000 characters (~6,000 tokens) for safe LLM delivery.

### 5. `answer_agent` — Final Response Generation
**Technology:** `langchain-groq`

- Invokes the strict document-grounded reasoning prompt.
- Rules enforced:
  1. Answer ONLY using provided context.
  2. MAY infer or summarize if context is related.
  3. Do NOT use external knowledge.
  4. If context is unrelated: "The answer is not available in the provided document."
- Applies feedback refinement if low-confidence is detected.
- Summarizes the answer to ~80 words for the dashboard.

---

## Graph Flow Diagram

```
                    ┌──────────────────────────────────────────────────────┐
                    │                   LangGraph StateGraph               │
                    │                                                      │
    PDF Upload ───► │  input_agent ──► tree_builder_agent ──► tree_store   │──► END
                    │                                          agent       │
                    │                                                      │
    User Query ───► │  tree_store_agent ──► reasoning_agent ──► answer     │──► END  
                    │                                           agent      │
                    └──────────────────────────────────────────────────────┘
```

---

## Pipeline State Schema

The `QAState` TypedDict flows through every node:

| Field | Type | Description |
|-------|------|-------------|
| `action` | `str` | `"build"` or `"query"` — determines graph route |
| `pdf_bytes` | `bytes` | Raw uploaded PDF |
| `url` | `str` | Target web URL |
| `raw_text` | `str \| list` | Extracted text segments |
| `tree` | `dict` | Constructed hierarchical tree |
| `query` | `str` | User's question |
| `context` | `str` | Retrieved content for LLM |
| `node_ids` | `list` | Matched tree node IDs |
| `answer` | `str` | LLM-generated answer |
| `summary` | `str` | Condensed answer |
| `refined` | `bool` | Whether refinement was triggered |
| `error` | `str` | Error message if any |
| `tree_info` | `dict` | Node count, depth, filepath |
| `traversal_info` | `dict` | Traversal path, depth reached |

---

## Key Design Decisions

1. **Vector-less Strategy:** No embeddings, no FAISS, no vector DB. Pure LLM reasoning over hierarchical summaries.
2. **Build Once, Query Many:** The tree is built exactly once per document upload. All queries reuse the persisted index.
3. **Parallel Processing:** 5 concurrent workers for segment summarization — ~5× speedup over sequential.
4. **Strict Document Grounding:** The answer agent refuses to use external knowledge, preventing hallucinations.
5. **Modular Graph:** Each agent is an independent function. Agents can be swapped, extended, or tested in isolation.
