# memory/vector_store.py
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, embeddings, metadata):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding, k=5):
        faiss.normalize_L2(query_embedding)
        scores, ids = self.index.search(query_embedding, k)
        return scores, ids
