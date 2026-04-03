#%%
import json
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


def load_queries(root_dir):
    file_path = os.path.join(root_dir, './data/queries.json')
    with open(file_path, 'r') as f:
        queries = json.load(f)
    return queries


def load_documents(root_dir):
    file_path = os.path.join(root_dir, './data/documents.json')
    with open(file_path, 'r') as f:
        documents = json.load(f)
    return documents


def find_top_k_results(query_vector, doc_vectors, top_k=3):
    similarity_scores = cosine_similarity(query_vector, doc_vectors).flatten()
    top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]

    return top_k_indices, similarity_scores