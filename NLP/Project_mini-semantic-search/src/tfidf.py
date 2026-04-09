#%%
import sys
import importlib
sys.path.append(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\semantic-search\src')

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import utils
importlib.reload(utils)


def vectorize(documents):
    documents_df = pd.json_normalize(documents)
    documents_df = documents_df.rename(columns={"metadata.topic": "topic"})

    vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
    doc_vectors = vectorizer.fit_transform(documents_df['text'])

    return vectorizer, doc_vectors


def search(query, vectorizer, doc_vectors, top_k=3):
    query_vector = vectorizer.transform([query])
    top_k_indices, similarity_scores = utils.find_top_k_results(
        query_vector, doc_vectors, top_k)

    return top_k_indices, similarity_scores[top_k_indices]
