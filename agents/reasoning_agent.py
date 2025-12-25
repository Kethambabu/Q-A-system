# agents/reasoning_agent.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # 👈 this is critical

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)
def reason(context, question):
    prompt = f"""
    Use multi-step reasoning.
    Context:
    {context}

    Question: {question}
    Answer step-by-step.
    """
    return llm.invoke(prompt).content
