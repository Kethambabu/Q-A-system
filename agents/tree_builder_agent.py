# agents/tree_builder_agent.py
"""
Tree Builder Agent — Converts raw document text into a hierarchical JSON tree
using LLM reasoning.

KEY DESIGN PRINCIPLES (fixing 413 token overflow):
  1. NEVER send full document text to the LLM in one call.
  2. Split document into small logical segments (paragraphs/sections).
  3. Each LLM call processes ONE small segment → produces title + summary ONLY.
  4. Full text content is stored in a SEPARATE reference map, NOT in tree nodes.
  5. Tree nodes contain ONLY: node_id, title, summary, content_ref, children.
  6. A second pass groups segments into a hierarchy using summaries alone.

CRITICAL FIX: Numerical data preservation
  - Summaries MUST preserve exact numbers, percentages, metrics
  - Segments with [TABLE_DATA] or [NUMERICAL_DATA] tags get special handling
  - Nodes are tagged with "has_numerical_data" for retrieval boosting
"""
import os
import json
import uuid
import re
import concurrent.futures
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,  # Summaries are short — no need for 4096
)

# ---------------------------------------------------------------------------
# Segment sizing — conservative to stay well within 8k context window.
# Each segment ≈ 400-500 tokens input + prompt ≈ 700 tokens total.
# LLM output ≈ 100-150 tokens. Well under 8k.
# ---------------------------------------------------------------------------
MAX_SEGMENT_CHARS = 2000   # ~500 tokens per segment
MIN_SEGMENT_CHARS = 200    # Don't create tiny segments

# ---------------------------------------------------------------------------
# Prompt to summarize a SINGLE small segment (not the full doc!)
# CRITICAL FIX: Explicit instructions to preserve numerical values.
# ---------------------------------------------------------------------------
SEGMENT_SUMMARY_PROMPT = """You are a document analyst. Read the following TEXT SEGMENT and produce a JSON object with exactly three fields:
- "title": a short descriptive title (max 10 words)
- "summary": a 1-2 sentence summary of what this segment covers
- "has_numerical_data": true if the segment contains numbers, metrics, percentages, scores, or table data; false otherwise

CRITICAL RULES FOR SUMMARY:
1. If the segment contains ANY numerical values (percentages, decimals, scores, counts, measurements), you MUST include the KEY numbers in your summary.
2. If you see accuracy, precision, recall, F1-score, AUC, or similar metrics — reproduce them EXACTLY in the summary.
3. For table data, summarize the table structure and include representative values.
4. NEVER abstract away numbers into vague phrases like "good performance" or "high accuracy". Always keep the actual values.

TEXT SEGMENT:
{segment_text}

Return ONLY valid JSON. No markdown fences, no explanation. Example:
{{"title": "Model Performance Results", "summary": "BERT achieved 94.5% accuracy and 92.3% F1-score on the test set, outperforming GPT-2 which scored 91.2% accuracy.", "has_numerical_data": true}}

Your JSON:"""

# ---------------------------------------------------------------------------
# Prompt to group segments into a hierarchy using ONLY their summaries.
# Input: list of titles+summaries (~20-50 tokens each). Total: ~500-1000 tokens.
# ---------------------------------------------------------------------------
HIERARCHY_PROMPT = """You are a document structure expert. Given the following numbered segments with their titles and summaries, organize them into a HIERARCHICAL structure.

SEGMENTS:
{segment_list}

INSTRUCTIONS:
1. Group related segments under common parent topics.
2. Create 2-3 levels of hierarchy.
3. Return a JSON object representing the tree structure.
4. Each node MUST have: "title", "summary", "segment_ids" (array of segment numbers that belong here, empty for non-leaf nodes), "children" (array of child nodes, empty for leaf nodes), "has_numerical_data" (true if ANY child segment contains numerical data).
5. Leaf nodes should reference 1-3 segment numbers in "segment_ids".
6. Non-leaf nodes should have a summary that covers all their children.
7. If ANY segment contains numerical data (metrics, results, percentages), make sure its parent node summary MENTIONS that numerical results exist in this section.
8. The root should represent the entire document.
9. Return ONLY valid JSON. No markdown fences, no explanation.

Your JSON:"""


from utils.text_helpers import split_into_segments
from utils.json_helpers import parse_llm_json

# ===========================================================================
# Utility functions
# ===========================================================================

def _segment_has_numerical_markers(text: str) -> bool:
    """Check if a segment contains numerical data markers or raw numbers."""
    if "[TABLE_DATA]" in text or "[NUMERICAL_DATA]" in text:
        return True
    # Check for meaningful numerical patterns
    numeric_matches = re.findall(r'\d+\.\d+%?|\d+%', text)
    meaningful = [m for m in numeric_matches if '.' in m or '%' in m or len(m) > 2]
    return len(meaningful) >= 2


def _summarize_segment(segment_text: str, segment_index: int) -> dict:
    """
    Send ONE small segment to LLM → get back title + summary ONLY.
    
    Token budget per call:
      - Prompt template: ~120 tokens
      - Segment text: ~500 tokens (2000 chars)
      - Output: ~50-150 tokens
      - Total: ~800 tokens  ✅ (well under 8k limit)
    """
    # Truncate segment if somehow still too large (safety net)
    truncated = segment_text[:MAX_SEGMENT_CHARS]
    
    # Pre-detect numerical content to validate LLM output
    has_numbers = _segment_has_numerical_markers(truncated)
    
    prompt = SEGMENT_SUMMARY_PROMPT.format(segment_text=truncated)
    
    try:
        response = llm.invoke(prompt).content.strip()
        result = parse_llm_json(response)
        
        summary = result.get("summary", truncated[:150] + "...")
        
        # Safety net: if we detected numbers but the LLM summary doesn't contain any,
        # append key numerical values from the original text
        if has_numbers and not re.search(r'\d+\.\d+%?|\d+%', summary):
            numbers_found = re.findall(r'\d+\.?\d*\s*%', truncated)
            if numbers_found:
                top_numbers = numbers_found[:5]
                summary += f" Key values: {', '.join(top_numbers)}."
        
        return {
            "title": result.get("title", f"Section {segment_index + 1}"),
            "summary": summary,
            "has_numerical_data": result.get("has_numerical_data", has_numbers),
        }
    except Exception:
        # Fallback: create summary without LLM
        return {
            "title": f"Section {segment_index + 1}",
            "summary": truncated[:150] + "...",
            "has_numerical_data": has_numbers,
        }


def _build_hierarchy_from_summaries(segment_summaries: list) -> dict:
    """
    Use LLM to group segments into a hierarchy based on summaries ONLY.
    
    Token budget:
      - Each segment summary: ~30-50 tokens
      - 20 segments: ~600-1000 tokens
      - Prompt template: ~250 tokens
      - Output: ~500 tokens
      - Total: ~1500 tokens  ✅
    """
    # Format segment list for the prompt — include numerical data flag
    segment_list = ""
    for i, seg in enumerate(segment_summaries):
        num_tag = " [CONTAINS NUMERICAL DATA]" if seg.get("has_numerical_data") else ""
        segment_list += f'{i}: title="{seg["title"]}" | summary="{seg["summary"]}"{num_tag}\n'
    
    prompt = HIERARCHY_PROMPT.format(segment_list=segment_list)
    
    try:
        response = llm.invoke(prompt).content.strip()
        hierarchy = parse_llm_json(response)
        return hierarchy
    except Exception:
        # Fallback: flat structure grouped by every 3-4 segments
        return _build_fallback_hierarchy(segment_summaries)


def _build_fallback_hierarchy(segment_summaries: list) -> dict:
    """Fallback: group segments into a simple 2-level hierarchy."""
    group_size = max(1, len(segment_summaries) // 3)
    children = []
    
    for i in range(0, len(segment_summaries), group_size):
        group = segment_summaries[i:i + group_size]
        group_ids = list(range(i, min(i + group_size, len(segment_summaries))))
        
        # Check if any segment in the group has numerical data
        group_has_numbers = any(s.get("has_numerical_data", False) for s in group)
        
        children.append({
            "title": group[0]["title"] if group else f"Group {i // group_size + 1}",
            "summary": "; ".join(s["summary"] for s in group),
            "segment_ids": group_ids,
            "children": [],
            "has_numerical_data": group_has_numbers,
        })
    
    return {
        "title": "Document Root",
        "summary": f"Document organized into {len(children)} sections.",
        "segment_ids": [],
        "children": children,
        "has_numerical_data": any(c.get("has_numerical_data") for c in children),
    }


def _assign_node_ids(node: dict, prefix: str = "root") -> dict:
    """Recursively assign unique node_ids to every node."""
    node["node_id"] = prefix
    for i, child in enumerate(node.get("children", [])):
        _assign_node_ids(child, prefix=f"{prefix}.{i+1}")
    return node


def _attach_content_refs(node: dict, segments: list, segment_summaries: list):
    """
    Recursively attach content references and summaries to tree nodes.
    
    - Leaf nodes with segment_ids get: content = concatenated segment text,
      and summary from segment summaries.
    - Non-leaf nodes get: content = "" (summaries only).
    
    CRITICAL FIX: Propagates has_numerical_data flag up the tree.
    """
    seg_ids = node.get("segment_ids", [])
    children = node.get("children", [])
    
    # Ensure required fields
    node.setdefault("node_id", f"n_{uuid.uuid4().hex[:6]}")
    node.setdefault("title", "Untitled")
    node.setdefault("summary", "")
    node.setdefault("content", "")
    node.setdefault("children", [])
    node.setdefault("has_numerical_data", False)
    
    if seg_ids and not children:
        # Leaf node — attach the actual segment content
        content_parts = []
        leaf_has_numbers = False
        for sid in seg_ids:
            if 0 <= sid < len(segments):
                content_parts.append(segments[sid])
            if 0 <= sid < len(segment_summaries):
                if segment_summaries[sid].get("has_numerical_data", False):
                    leaf_has_numbers = True
        node["content"] = "\n\n".join(content_parts)
        
        # Propagate numerical flag
        if leaf_has_numbers:
            node["has_numerical_data"] = True
        
        # Also check the actual content for numbers (belt and suspenders)
        if not node["has_numerical_data"]:
            node["has_numerical_data"] = _segment_has_numerical_markers(node["content"])
        
        # Use the segment summary if node summary is empty
        if not node["summary"]:
            summary_parts = []
            for sid in seg_ids:
                if 0 <= sid < len(segment_summaries):
                    summary_parts.append(segment_summaries[sid]["summary"])
            node["summary"] = " ".join(summary_parts)
    else:
        # Non-leaf node — no verbatim content, just summary
        node["content"] = ""
    
    # Remove segment_ids from final output (internal use only)
    node.pop("segment_ids", None)
    
    # Recurse into children
    child_has_numbers = False
    for child in children:
        _attach_content_refs(child, segments, segment_summaries)
        if child.get("has_numerical_data", False):
            child_has_numbers = True
    
    # Propagate numerical flag upward from children
    if child_has_numbers:
        node["has_numerical_data"] = True


# ===========================================================================
# Main entry point
# ===========================================================================

from typing import Union

def build_tree(document_data: Union[str, list]) -> dict:
    """
    Build a hierarchical document tree using segmented LLM processing.
    
    PIPELINE (each step uses minimal tokens):
    
    Step 1: Receive document list or split document string into logical segments
    Step 2: For each segment, LLM produces title + summary (~800 tokens/call)
    Step 3: LLM groups segment summaries into hierarchy (~1500 tokens/call)
    Step 4: Attach original content to leaf nodes (no LLM call)
    
    MAX tokens per LLM call: ~1500 (well under 8k limit)
    Total LLM calls: N segments + 1 hierarchy call
    
    Returns:
        dict — the complete tree with node_id, title, summary, content, children, has_numerical_data
    """
    if not document_data:
        return {
            "node_id": "root",
            "title": "Empty Document",
            "summary": "No content provided.",
            "content": "",
            "children": [],
            "has_numerical_data": False,
        }

    # ── Step 1: Handle pre-segmented data or Split into logical segments ──
    if isinstance(document_data, list):
        segments = document_data
    else:
        text = document_data.strip()
        if not text:
            return {
                "node_id": "root",
                "title": "Empty Document",
                "summary": "No content provided.",
                "content": "",
                "children": [],
                "has_numerical_data": False,
            }
        segments = split_into_segments(text)
    
    # If very short document (single segment), handle directly
    if len(segments) == 1:
        summary_info = _summarize_segment(segments[0], 0)
        return {
            "node_id": "root",
            "title": summary_info["title"],
            "summary": summary_info["summary"],
            "content": segments[0],
            "children": [],
            "has_numerical_data": summary_info.get("has_numerical_data", False),
        }

    # ── Step 2: Summarize each segment independently ─────────────
    # Each call: ~800 tokens (segment text + prompt + output)
    segment_summaries = [None] * len(segments)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_idx = {
            executor.submit(_summarize_segment, seg, idx): idx 
            for idx, seg in enumerate(segments)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                segment_summaries[idx] = future.result()
            except Exception:
                has_nums = _segment_has_numerical_markers(segments[idx])
                segment_summaries[idx] = {
                    "title": f"Section {idx + 1}",
                    "summary": segments[idx][:150] + "...",
                    "has_numerical_data": has_nums,
                }

    # ── Step 3: Build hierarchy from summaries only ──────────────
    # Single call: ~1500 tokens (all summaries + prompt + output)
    hierarchy = _build_hierarchy_from_summaries(segment_summaries)

    # ── Step 4: Assign node IDs ──────────────────────────────────
    _assign_node_ids(hierarchy, prefix="root")

    # ── Step 5: Attach original content to leaf nodes ────────────
    # NO LLM call — just string concatenation
    _attach_content_refs(hierarchy, segments, segment_summaries)

    return hierarchy
