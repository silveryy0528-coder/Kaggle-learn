#%%
import sys
import importlib
sys.path.append(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\semantic-search\src')

import pandas as pd
import numpy as np

import load_data
importlib.reload(load_data)

from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main(query, documents, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode([doc['text'] for doc in documents])
    query_embedding = model.encode([query])

    similarity_scores = cosine_similarity(query_embedding, doc_embeddings).flatten()
    top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]

    return top_k_indices, similarity_scores[top_k_indices]
