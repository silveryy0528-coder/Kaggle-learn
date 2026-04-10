import faiss
from dataclasses import dataclass
from src.embedding import embed_text


@dataclass
class FaissConfig():
    pass


@dataclass
class FaissFlatConfig(FaissConfig):
    pass


@dataclass
class FaissIvfConfig(FaissConfig):
    nlist: int = 50


@dataclass
class FaissIvfpqConfig(FaissConfig):
    nlist: int = 50
    m: int = 8
    nbits: int = 8


def build_faiss_flat(embeddings, dim):
    index = faiss.IndexFlatL2(dim)
    return index


def build_faiss_ivf(embeddings, dim, nlist=100):
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(embeddings)
    return index


def build_faiss_ivfpq(embeddings, dim, nlist=100, m=8, nbits=8):
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index.train(embeddings)
    return index


def build_faiss_index(embeddings, settings):
    if not isinstance(settings, FaissConfig):
        raise TypeError('Wrong FAISS settings provided.')

    dim = embeddings.shape[1]
    if isinstance(settings, FaissFlatConfig):
        index = build_faiss_flat(embeddings, dim)
    elif isinstance(settings, FaissIvfConfig):
        index = build_faiss_ivf(embeddings, dim, settings.nlist)
    elif isinstance(settings, FaissIvfpqConfig):
        index = build_faiss_ivfpq(
            embeddings, dim, settings.nlist, settings.m, settings.nbits)

    index.add(embeddings)
    return index


def retrieve_top_k(question, chunks, embedder, faiss_index, k=3):
    query_vec = embed_text(embedder, texts=[question])
    scores, indices = faiss_index.search(query_vec, k)
    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks

