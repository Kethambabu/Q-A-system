from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

def summarize(text, max_words=80):
    """
    Summarizes long text into a concise explanation.
    """
    prompt = f"""
    Summarize the following answer in under {max_words} words.
    Keep it clear and informative.

    Answer:
    {text}
    """

    return llm.invoke(prompt).content.strip()
