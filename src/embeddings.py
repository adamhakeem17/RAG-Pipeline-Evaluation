from sentence_transformers import SentenceTransformer

def load_e5():
    # Centralized model loader for consistent config.
    return SentenceTransformer("intfloat/e5-base")

def embed_passages(model, passages):
    if model is None:
        raise ValueError("model must be provided to embed passages.")
    if passages is None or len(passages) == 0:
        raise ValueError("passages must be a non-empty list of strings.")
    return model.encode([f"passage: {p}" for p in passages], convert_to_numpy=True)

def embed_query(model, query):
    if model is None:
        raise ValueError("model must be provided to embed query.")
    if query is None or query == "":
        raise ValueError("query must be a non-empty string.")
    return model.encode([f"query: {query}"], convert_to_numpy=True)[0]
