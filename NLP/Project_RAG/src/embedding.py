from sentence_transformers import SentenceTransformer


def load_embedder(model_name="all-MiniLM-L6-v2", device="cuda"):
    return SentenceTransformer(model_name, device=device)


def embed_text(embedder, texts, device='cuda'):
    embeddings = embedder.encode(
        texts,
        device=device,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings