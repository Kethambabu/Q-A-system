# agents/crawler_agent.py
import requests

def fetch_resource(url: str) -> bytes:
    """
    Fetches content from a URL (PDF or HTML).
    """
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.content
