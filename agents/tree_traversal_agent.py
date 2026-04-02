# agents/tree_traversal_agent.py
"""
Tree Traversal Agent — Replaces vector retrieval + reranking.
Uses LLM reasoning to navigate the tree top-down, selecting the most relevant
branches at each level, mimicking how a human would navigate a table of contents.

CRITICAL FIX: Added numerical data awareness
- Nodes flagged with has_numerical_data are boosted in selection
- Queries about numbers/metrics/results auto-include numerical nodes
- Increased retrieval breadth for numerical queries
"""
import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1024,
)

# ---------------------------------------------------------------------------
# Keywords that indicate a query is looking for numerical/quantitative data
# ---------------------------------------------------------------------------
NUMERICAL_QUERY_KEYWORDS = [
    "accuracy", "precision", "recall", "f1", "f1-score", "score",
    "result", "results", "numerical", "number", "numbers",
    "percentage", "percent", "%", "metric", "metrics",
    "performance", "evaluation", "benchmark", "comparison",
    "table", "data", "statistics", "statistical",
    "auc", "roc", "bleu", "rouge", "loss", "error rate",
    "value", "values", "measurement", "quantitative",
]

# ---------------------------------------------------------------------------
# Prompt for the LLM to select the best child node(s) at each tree level.
# CRITICAL FIX: Instructions to prioritize numerical data nodes.
# ---------------------------------------------------------------------------
TRAVERSAL_PROMPT = """You are a document navigation expert. Given a user's question and a list of section summaries, your job is to select the section(s) most likely to contain the answer.

USER QUESTION: {query}

AVAILABLE SECTIONS:
{sections}

INSTRUCTIONS:
1. Analyze the question carefully.
2. Compare it against each section's title and summary.
3. Select 1-3 sections that are MOST RELEVANT to answering the question.
4. IMPORTANT: If the user is asking about numbers, results, metrics, accuracy, percentages, or any quantitative data — ALWAYS include sections marked with [HAS NUMERICAL DATA].
5. Return ONLY a JSON array of the selected node_ids. Example: ["n1.2", "n1.3"]
6. If none are relevant, return an empty array: []
7. Return ONLY the JSON array. No explanation.

Your selection:"""


def _is_numerical_query(query: str) -> bool:
    """Detect if a query is looking for numerical/quantitative information."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in NUMERICAL_QUERY_KEYWORDS)


def _parse_node_ids(raw: str) -> list:
    """Parse a JSON array of node IDs from LLM output."""
    cleaned = raw.strip()
    # Remove markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass

    # Fallback: try to find array in the text
    match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    return []


def traverse_tree(query: str, tree_store, max_depth: int = 4) -> dict:
    """
    Navigate the tree using LLM reasoning.
    
    Process:
    1. Start at root
    2. Present children summaries to LLM with the query
    3. LLM selects most relevant child(ren)
    4. Recurse into selected children
    5. Stop when leaf nodes or max_depth reached
    
    CRITICAL FIX: For numerical queries, also force-include any nodes
    that have has_numerical_data=True, regardless of LLM selection.
    
    Returns:
        dict with:
        - "node_ids": list of relevant leaf node IDs
        - "traversal_path": list of steps taken (for debugging)
        - "depth_reached": how deep the traversal went
    """
    root = tree_store.get_root()
    if root is None:
        return {"node_ids": [], "traversal_path": [], "depth_reached": 0}

    is_num_query = _is_numerical_query(query)
    collected_node_ids = []
    traversal_path = []
    max_reached_depth = 0

    def _traverse(node_id: str, depth: int):
        nonlocal max_reached_depth
        max_reached_depth = max(max_reached_depth, depth)

        if depth > max_depth:
            collected_node_ids.append(node_id)
            return

        children_summaries = tree_store.get_children_summaries(node_id)

        # Leaf node — collect it
        if not children_summaries:
            collected_node_ids.append(node_id)
            traversal_path.append({
                "depth": depth,
                "node_id": node_id,
                "action": "LEAF — collected"
            })
            return

        # Format sections for the LLM — include numerical data flag
        sections_text = ""
        numerical_child_ids = []
        for c in children_summaries:
            num_marker = ""
            # Check the actual node for has_numerical_data flag
            full_node = tree_store.get_node(c["node_id"])
            if full_node and full_node.get("has_numerical_data", False):
                num_marker = " [HAS NUMERICAL DATA]"
                numerical_child_ids.append(c["node_id"])
            sections_text += f'- node_id: "{c["node_id"]}" | title: "{c["title"]}" | summary: "{c["summary"]}"{num_marker}\n'

        prompt = TRAVERSAL_PROMPT.format(query=query, sections=sections_text)
        response = llm.invoke(prompt).content.strip()
        selected_ids = _parse_node_ids(response)

        # CRITICAL FIX: For numerical queries, force-include numerical nodes
        if is_num_query and numerical_child_ids:
            for nid in numerical_child_ids:
                if nid not in selected_ids:
                    selected_ids.append(nid)

        traversal_path.append({
            "depth": depth,
            "node_id": node_id,
            "children_presented": [c["node_id"] for c in children_summaries],
            "llm_selected": selected_ids,
            "numerical_boost": numerical_child_ids if is_num_query else [],
            "action": f"Selected {len(selected_ids)} of {len(children_summaries)} children"
        })

        # If LLM selected nothing, fall back to all children
        if not selected_ids:
            selected_ids = [c["node_id"] for c in children_summaries]

        # Filter to valid children only
        valid_child_ids = {c["node_id"] for c in children_summaries}
        selected_ids = [sid for sid in selected_ids if sid in valid_child_ids]

        if not selected_ids:
            # Safety fallback: just use all children
            selected_ids = [c["node_id"] for c in children_summaries]

        # Recurse into selected children
        for sid in selected_ids:
            _traverse(sid, depth + 1)

    _traverse(root.get("node_id", "root"), depth=0)

    return {
        "node_ids": collected_node_ids,
        "traversal_path": traversal_path,
        "depth_reached": max_reached_depth,
    }
