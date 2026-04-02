import os
import sys
import requests
from typing import TypedDict, Optional, Union, List, Any
from langgraph.graph import StateGraph, END

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.ingestion import ingest_pdf_bytes, ingest_html
from agents.tree_builder_agent import build_tree
from agents.tree_traversal_agent import traverse_tree
from agents.reasoning_agent import reason
from agents.feedback_agent import needs_refinement
from agents.summarizer_agent import summarize
from agents.query_refiner_agent import refine_query
from memory.tree_store import TreeStore

# ---------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------
class QAState(TypedDict):
    action: str                        # "build" or "query"
    url: Optional[str]                 # Target URL
    pdf_bytes: Optional[bytes]         # Raw PDF upload
    raw_text: Optional[Union[str, list]] # Ingested segments
    tree: Optional[dict]               # Constructed logic tree
    query: Optional[str]               # User question
    refined_query: Optional[str]       # Refined structured query
    refinement_info: Optional[dict]    # Query refinement metadata
    context: Optional[str]             # Final traversal context
    node_ids: Optional[list]           # Active tree nodes
    answer: Optional[str]              # Response from LLM
    summary: Optional[str]             # Condensed Answer
    refined: Optional[bool]            # Whether hallucination triggered refinement
    error: Optional[str]               
    tree_info: Optional[dict]
    traversal_info: Optional[dict]

# ---------------------------------------------------------
# 2. Agent Definitions (Nodes)
# ---------------------------------------------------------
def input_agent(state: QAState) -> QAState:
    """1. Loads PDF or URL into raw text/segments."""
    if state.get("pdf_bytes"):
        state["raw_text"] = ingest_pdf_bytes(state["pdf_bytes"])
    elif state.get("url"):
        response = requests.get(state["url"], timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type or state["url"].lower().endswith(".pdf"):
            state["raw_text"] = ingest_pdf_bytes(response.content)
        else:
            state["raw_text"] = ingest_html(response.text)
    return state

def tree_builder_agent(state: QAState) -> QAState:
    """2. Builds the tree index from generated text segments."""
    if state.get("raw_text"):
        state["tree"] = build_tree(state["raw_text"])
    return state

def tree_store_agent(state: QAState) -> QAState:
    """3. Saves or loads the tree index to memory store."""
    store = TreeStore()
    if state["action"] == "build":
        if state.get("tree"):
            filepath = store.save(state["tree"])
            state["tree_info"] = {
                "total_nodes": store.get_node_count(),
                "tree_depth": store.get_tree_depth(),
                "filepath": filepath
            }
    elif state["action"] == "query":
        try:
            store.load()
            state["tree_info"] = {
                "total_nodes": store.get_node_count(),
                "tree_depth": store.get_tree_depth(),
            }
        except FileNotFoundError:
            state["error"] = "Tree index not found. Please upload a document to build the index first."
        except Exception as e:
            state["error"] = f"Error loading tree index: {str(e)}"
    return state

def query_refiner_agent_node(state: QAState) -> QAState:
    """3.5 NEW: Refines the user query into a structured, precise query."""
    if state.get("error"):
        return state
    
    original_query = state.get("query", "")
    refinement_result = refine_query(original_query)
    
    state["refined_query"] = refinement_result["refined_query"]
    state["refinement_info"] = {
        "original_query": refinement_result["original_query"],
        "refined_query": refinement_result["refined_query"],
        "method": refinement_result["refinement_method"],
    }
    return state

def reasoning_agent(state: QAState) -> QAState:
    """4. Traverses tree structure based on Query and fetches explicit logical constraints."""
    if state.get("error"):
        return state
    
    store = TreeStore()
    store.load()  # Access memory cache
    
    # Use the refined query for traversal (more precise matching)
    search_query = state.get("refined_query") or state.get("query", "")
    
    traversal_result = traverse_tree(search_query, store)
    node_ids = traversal_result["node_ids"]
    
    if node_ids:
        content_parts = []
        for nid in node_ids:
            node_content = store.get_node_content(nid)
            if node_content:
                content_parts.append(node_content)
            else:
                leaf_content = store.get_all_leaf_content(nid)
                if leaf_content:
                    content_parts.append(leaf_content)
        context = "\n\n".join(content_parts)
    else:
        context = ""
    
    # Smart context truncation: don't cut mid-table
    # Increased limit from 8000 to 10000 for better numerical coverage
    if len(context) > 10000:
        # Try to cut at a paragraph boundary
        cut_point = context.rfind("\n\n", 0, 10000)
        if cut_point > 6000:
            context = context[:cut_point]
        else:
            context = context[:10000]
    
    state["context"] = context
    state["node_ids"] = node_ids
    state["traversal_info"] = {
        "retrieved_node_ids": node_ids,
        "traversal_path": traversal_result["traversal_path"],
        "depth_reached": traversal_result["depth_reached"],
        "search_query_used": search_query,
    }
    return state

def answer_agent(state: QAState) -> QAState:
    """5. Generates the final strict document-grounded response using retrieved contexts."""
    if state.get("error"):
        return state
        
    # Safely get context and query with defaults
    context = state.get("context", "")
    # Use the ORIGINAL query for answering (user's actual intent)
    query = state.get("query", "")
    
    try:
        answer = reason(context, query)
        
        if "The answer is not available in the provided document" in answer:
            refined = False
            summary = "The answer is not available in the provided document."
        else:
            refined = False
            if needs_refinement(answer):
                refined = True
                # Keep refinement context smaller to avoid token limits
                context_plus = context[:6000] + "\n\nProvide a more detailed explanation."
                answer = reason(context_plus, query)
            
            summary = summarize(answer, max_words=80)
            
        state["answer"] = answer
        state["summary"] = summary
        state["refined"] = refined
    except Exception as e:
        state["error"] = f"Error generating answer: {str(e)}"
        state["answer"] = "An error occurred while generating the answer."
        state["summary"] = "An error occurred while generating the answer."
        state["refined"] = False
        
    return state


# ---------------------------------------------------------
# 3. LangGraph Orchestration Connections
# ---------------------------------------------------------
def graph_router(state: QAState) -> str:
    """Entry Router"""
    if state["action"] == "build":
        return "input_agent"
    elif state["action"] == "query":
        return "tree_store_agent"

def store_router(state: QAState) -> str:
    """Mid-Flow Router"""
    if state["action"] == "build":
        return END
    else:
        if state.get("error"):
            return END
        return "query_refiner_agent"

# Compile Graph
graph_builder = StateGraph(QAState)

graph_builder.add_node("input_agent", input_agent)
graph_builder.add_node("tree_builder_agent", tree_builder_agent)
graph_builder.add_node("tree_store_agent", tree_store_agent)
graph_builder.add_node("query_refiner_agent", query_refiner_agent_node)
graph_builder.add_node("reasoning_agent", reasoning_agent)
graph_builder.add_node("answer_agent", answer_agent)

# Set Up Flow
graph_builder.set_conditional_entry_point(
    graph_router,
    {"input_agent": "input_agent", "tree_store_agent": "tree_store_agent"}
)

# Build path: input → tree_builder → tree_store → END
graph_builder.add_edge("input_agent", "tree_builder_agent")
graph_builder.add_edge("tree_builder_agent", "tree_store_agent")
graph_builder.add_conditional_edges("tree_store_agent", store_router)

# Query path: tree_store → query_refiner → reasoning → answer → END
graph_builder.add_edge("query_refiner_agent", "reasoning_agent")
graph_builder.add_edge("reasoning_agent", "answer_agent")
graph_builder.add_edge("answer_agent", END)

qa_graph = graph_builder.compile()


# ---------------------------------------------------------
# 4. Pipeline Execution Endpoints (Backward Compatible Wrappers)
# ---------------------------------------------------------
def build_and_save_tree(raw_text: Union[str, list] = None, pdf_bytes: bytes = None, url: str = None) -> dict:
    """Executes the specialized Build phase of the LangGraph architecture."""
    state_input = {"action": "build"}
    if pdf_bytes:
        state_input["pdf_bytes"] = pdf_bytes
    elif url:
        state_input["url"] = url
    elif raw_text:
        state_input["raw_text"] = raw_text
        
    state = qa_graph.invoke(state_input)
    return state.get("tree_info", {})

def run_pipeline(query: str) -> dict:
    """Executes the specific Query phase seamlessly via reasoning and answer agents."""
    state = qa_graph.invoke({"action": "query", "query": query})
    
    if state.get("error"):
        return {"error": state["error"]}
        
    return {
        "answer": state.get("answer"),
        "summary": state.get("summary"),
        "refined": state.get("refined", False),
        "context_used": state.get("context", "")[:1000],
        "tree_info": state.get("tree_info", {}),
        "traversal_info": state.get("traversal_info", {}),
        "refinement_info": state.get("refinement_info", {}),
    }
