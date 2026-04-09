#%%
import numpy as np
import glob
import os
import faiss
import re


def load_all_embeddings(folder, pattern='embeddings_batch_*.npy'):
    def extract_batch_number(filename):
        match = re.search(r"batch_(\d+)", filename)
        return int(match.group(1))

    path_pattern = os.path.join(folder, pattern)
    files = glob.glob(path_pattern)
    files = sorted(files, key=extract_batch_number)

    all_embeddings = []
    for f in files:
        emb = np.load(f)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    return embeddings


def normalize_embeddings(embeddings):
    '''Guarantees that all embeddings are unit vectors (optional)'''
    faiss.normalize_L2(embeddings)
    return embeddings


def build_flat_index(embeddings):
    '''Baseline with exact search, no approximation'''
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    return index


def build_ivf_index(embeddings, nlist=100):
    dim = embeddings.shape[1]

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    index.train(embeddings)
    index.add(embeddings)
    return index


def build_ivfpq_index(embeddings, nlist=100, m=8, nbits=8):
    dim = embeddings.shape[1]
    # m: number of chunks per vector
    # nbits: 2**nbits centroids per chunk
    quantizer = faiss.IndexFlatL2(dim)

    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index.train(embeddings)
    index.add(embeddings)

    return index


def save_index(index, out_folder, filename):
    out_path = os.path.join(out_folder, f'{filename}.index')
    faiss.write_index(index, out_path)


emb_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\real-world-search\data\processed'
output_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\real-world-search\models'

embeddings = load_all_embeddings(emb_folder)
embeddings = normalize_embeddings(embeddings)

#%%
flat_index = build_flat_index(embeddings)
save_index(flat_index, output_folder, 'faiss_flat')

#%%
ivf_index = build_ivf_index(embeddings, nlist=100)
save_index(ivf_index, output_folder, 'faiss_ivf')

#%%
# IVF: search fewer candidates (in voronoi cells)
# PQ: compare them very fast by dividing them into chunks
ivfpq_index = build_ivfpq_index(embeddings, nlist=100, m=96)
save_index(ivfpq_index, output_folder, 'faiss_ivfpq')