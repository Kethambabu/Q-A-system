# agents/query_refiner_agent.py
"""
Query Refiner Agent — Transforms vague, human-friendly queries into structured,
precise queries that the retrieval and reasoning pipeline can effectively process.

Example:
  Input:  "give me numerical results in the paper"
  Output: "Extract numerical performance metrics such as accuracy, precision,
           recall, F1-score, AUC, BLEU score from the results, evaluation,
           or experiments section. Include all tables with numerical data."

This agent sits at the START of the query pipeline:
  User Query → Query Refiner → Tree Traversal → Reasoning → Answer
"""
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=256,
)

# ---------------------------------------------------------------------------
# Query refinement prompt
# ---------------------------------------------------------------------------
QUERY_REFINE_PROMPT = """You are a query refinement expert for a document Q&A system. Your job is to transform a user's casual, vague question into a precise, structured query that will help the system find the right information.

USER QUERY: {query}

INSTRUCTIONS:
1. Expand vague terms into specific ones. For example:
   - "results" → "performance metrics, accuracy, precision, recall, F1-score"
   - "numbers" → "numerical values, percentages, scores, measurements, statistics"
   - "how well" → "performance metrics and evaluation scores"
   - "what did they find" → "key findings, conclusions, and reported results"
2. Add relevant section hints (e.g., "from the results section, evaluation section, experiments section")
3. If the query asks for numerical data, explicitly request: tables, figures, metrics, scores, percentages
4. Keep the refined query under 50 words
5. Maintain the original INTENT — do NOT change what the user is asking for

Return ONLY the refined query text. No quotes, no explanation, no prefix like "Refined query:".

Your refined query:"""

# ---------------------------------------------------------------------------
# Fast-path patterns: queries we can refine without LLM (saves API calls)
# ---------------------------------------------------------------------------
FAST_REFINE_PATTERNS = [
    {
        "pattern": r"\b(numerical|number)\s*(result|data|finding|value)",
        "refined": "Extract all numerical performance metrics including accuracy, precision, recall, F1-score, AUC, BLEU, ROUGE scores, percentages, and statistical values from the results, evaluation, and experiments sections. Include all tables containing numerical data."
    },
    {
        "pattern": r"\b(give|show|list|get|find)\s*(me\s*)?(the\s*)?(result|finding|outcome)",
        "refined": "List the key findings and results reported in the paper, including any numerical metrics, performance scores, accuracy values, and conclusions from the experiments and evaluation sections."
    },
    {
        "pattern": r"\b(table|tabular)\s*(data|result|content)",
        "refined": "Extract all tabular data from the document including tables with numerical values, comparison matrices, performance benchmarks, and structured data with metrics."
    },
    {
        "pattern": r"\b(accuracy|precision|recall|f1|performance)\b",
        "refined": None,  # Use LLM — the query is already specific enough but could benefit from expansion
    },
]


def refine_query(query: str) -> dict:
    """
    Refine a user query into a structured, precise query.
    
    Returns:
        dict with:
        - "original_query": the user's original query
        - "refined_query": the structured, expanded query
        - "refinement_method": "fast_path" or "llm"
    """
    if not query or not query.strip():
        return {
            "original_query": query,
            "refined_query": query,
            "refinement_method": "none",
        }
    
    query_stripped = query.strip()
    query_lower = query_stripped.lower()
    
    # Fast-path: check for common patterns that don't need LLM
    for pattern_entry in FAST_REFINE_PATTERNS:
        if re.search(pattern_entry["pattern"], query_lower, re.IGNORECASE):
            if pattern_entry["refined"] is not None:
                return {
                    "original_query": query_stripped,
                    "refined_query": pattern_entry["refined"],
                    "refinement_method": "fast_path",
                }
            else:
                break  # Fall through to LLM refinement
    
    # If the query is already long and specific (>15 words), skip refinement
    word_count = len(query_stripped.split())
    if word_count > 15:
        return {
            "original_query": query_stripped,
            "refined_query": query_stripped,
            "refinement_method": "passthrough",
        }
    
    # LLM-based refinement for complex/ambiguous queries
    try:
        prompt = QUERY_REFINE_PROMPT.format(query=query_stripped)
        refined = llm.invoke(prompt).content.strip()
        
        # Sanitize: remove quotes or prefixes the LLM might add
        refined = re.sub(r'^["\']+|["\']+$', '', refined)
        refined = re.sub(r'^(Refined query|Output|Query):\s*', '', refined, flags=re.IGNORECASE)
        
        # Safety: if refinement is empty or too short, use original
        if not refined or len(refined) < 10:
            refined = query_stripped
            method = "fallback"
        else:
            method = "llm"
        
        return {
            "original_query": query_stripped,
            "refined_query": refined,
            "refinement_method": method,
        }
    except Exception:
        return {
            "original_query": query_stripped,
            "refined_query": query_stripped,
            "refinement_method": "error_fallback",
        }
