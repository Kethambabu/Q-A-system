# agents/reasoning_agent.py
"""
Reasoning Agent — Generates document-grounded answers from retrieved context.

CRITICAL FIX: Updated prompt to explicitly handle numerical data, tables,
and metrics. Instructs the LLM to reproduce exact numbers from the context.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500
)

def reason(context, question):
    # Ensure context and question are strings
    context = str(context) if context is not None else ""
    question = str(question) if question is not None else ""
    
    if not context or not context.strip():
        return "The answer is not available in the provided document."

    prompt = f"""You are a research paper question-answering system.

Rules:
1. Answer ONLY using the provided context.
2. You MAY infer or summarize if the context is related.
3. Do NOT use external knowledge outside the document.
4. If the context is completely unrelated, say:
   "The answer is not available in the provided document."
5. Prefer explaining using available sections even if partial.
6. CRITICAL: If the context contains numerical values (percentages, scores, metrics, measurements), you MUST include them EXACTLY as they appear. Never round, approximate, or omit numbers.
7. If you see table data marked with [TABLE_DATA] or [NUMERICAL_DATA], format the numbers clearly in your answer — use bullet points or a structured list.
8. When asked about results, metrics, or performance — always quote the specific numbers from the context.

Context:
{context}

Question:
{question}

Answer:"""
    return llm.invoke(prompt).content.strip()
