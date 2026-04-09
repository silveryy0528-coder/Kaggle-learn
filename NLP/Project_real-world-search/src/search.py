#%%
import json
import os
import time
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


def search(index, query_vector, index_name, k=3):
    faiss.normalize_L2(query_vector)

    start = time.time()
    scores, indices = index.search(query_vector, k)
    end = time.time()

    print(f"Search time of {index_name}: {1e3*(end - start):.2f} ms")

    return scores, indices


def print_results(chunk_file, queries, scores, indices):
    with open(chunk_file, 'r') as f:
        chunks = json.load(f)

    print(f'Number of chunks are: {len(chunks)}')
    assert(len(queries) == scores.shape[0])
    top_k = scores.shape[-1]

    for query, index, score in zip(queries, indices, scores):
        print(f'For query "{query}", top {top_k} matches are:')
        texts = []
        for i in range(top_k):
            texts.append(chunks[index[i]])
        df = pd.DataFrame(texts, columns=['chunk_id', 'doc_id', 'text'])
        df['score'] = score
        print(df)


def load_faiss(index_folder, index_name):
    faiss_index = faiss.read_index(os.path.join(index_folder, index_name))
    print(f'Number of indices in {index_name} is {faiss_index.ntotal}')
    return faiss_index


#%%
index_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\real-world-search\models'
chunk_file = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\real-world-search\data\processed\chunks.json"

model_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name, device='cuda')

#%%
queries = [
    'What is the latest news on World Cup?'
]
query_vectors = embedder.encode(
    queries,
    convert_to_numpy=True,
    normalize_embeddings=True,
    device='cuda'
)

#%%
index_name = 'faiss_ivfpq'
faiss_index = load_faiss(index_folder, index_name=f'{index_name}.index')
if (
    index_name == 'faiss_ivf'
    or index_name == 'faiss_ivfpq'
):
    faiss_index.nprobe = 10

scores, indices = search(
    faiss_index,
    query_vectors,
    index_name=index_name,
    k=3
)
print_results(chunk_file, queries, scores, indices)