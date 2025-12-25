# agents/reranker_agent.py
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, passages):
    pairs = [[query, p] for p in passages]
    scores = reranker.predict(pairs)
    return sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
