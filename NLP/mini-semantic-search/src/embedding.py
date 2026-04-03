#%%
import sys
import importlib
sys.path.append(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\semantic-search\src')

import utils
importlib.reload(utils)

from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model_name = 'all-MiniLM-L6-v2'


def vectorize(documents):
    embedder = SentenceTransformer(model_name)
    doc_embeddings = embedder.encode([doc['text'] for doc in documents])

    return embedder, doc_embeddings


def search(query, embedder, doc_embeddings, top_k=3):
    query_embedding = embedder.encode([query])
    top_k_indices, similarity_scores = utils.find_top_k_results(
        query_embedding, doc_embeddings, top_k)

    return top_k_indices, similarity_scores[top_k_indices]
