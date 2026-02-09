import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        if dim <= 0:
            raise ValueError("dim must be greater than 0.")
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []

    def add(self, embeddings, texts):
        if embeddings is None or texts is None:
            raise ValueError("embeddings and texts must be provided.")
        if len(embeddings) == 0 or len(texts) == 0:
            raise ValueError("embeddings and texts must be non-empty.")
        if len(embeddings) != len(texts):
            raise ValueError("embeddings and texts must have the same length.")
        # Normalize vectors before adding for cosine similarity with inner product.
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_emb, k=5):
        if query_emb is None:
            raise ValueError("query_emb must be provided.")
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if self.index.ntotal == 0:
            raise ValueError("vector store is empty; add embeddings before searching.")
        faiss.normalize_L2(query_emb.reshape(1,-1))
        scores, ids = self.index.search(query_emb.reshape(1,-1), k)
        return [(self.texts[i], scores[0][j]) for j,i in enumerate(ids[0])]
