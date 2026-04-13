import os
import logging
from transformers import logging as hf_logging
from sentence_transformers import SentenceTransformer

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()


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