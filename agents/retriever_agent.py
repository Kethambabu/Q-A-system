# agents/retriever_agent.py
def retrieve(vector_store, query_embedding, k):
    scores, ids = vector_store.search(query_embedding, k)
    chunks = [vector_store.metadata[i]["text"] for i in ids[0]]
    return chunks
