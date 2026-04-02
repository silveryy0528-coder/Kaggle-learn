#%%
import sys
import importlib
sys.path.append(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\semantic-search\src')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import load_data
importlib.reload(load_data)


def main(query, documents, top_k=3):
    documents_df = pd.json_normalize(documents)
    documents_df = documents_df.rename(columns={"metadata.topic": "topic"})

    vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
    doc_vectors = vectorizer.fit_transform(documents_df['text'])

    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, doc_vectors).flatten()
    top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]

    return top_k_indices, similarity_scores[top_k_indices]
