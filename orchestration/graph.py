# orchestration/graph.py
def run_pipeline(resource_text, query, agents):
    chunks = agents["chunker"](resource_text)
    embeddings = agents["embedder"].embed(chunks)

    vector_store = agents["vector_store"](embeddings.shape[1])
    vector_store.add(embeddings, [{"text": c} for c in chunks])

    retrieved = agents["retriever"](vector_store, agents["embedder"].embed([query]), 8)
    reranked = agents["reranker"](query, retrieved)
    answer = agents["reasoner"]("\n".join([r[0] for r in reranked[:3]]), query)

    return answer
