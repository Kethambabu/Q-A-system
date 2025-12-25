# agents/embedding_agent.py
from sentence_transformers import SentenceTransformer

class EmbeddingAgent:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=False)
