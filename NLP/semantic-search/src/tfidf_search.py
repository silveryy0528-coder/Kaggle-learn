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

    vectorizer = TfidfVectorizer(stop_words='english')
    doc_vectors = vectorizer.fit_transform(documents_df['text'])

    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, doc_vectors).flatten()
    top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]

    return top_k_indices, similarity_scores[top_k_indices]


root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\semantic-search'

# Load documents
documents = load_data.load_documents(root_dir)
queries = load_data.load_queries(root_dir)

for i, query in enumerate(queries):
    top_k_indices, similarity_scores = main(query['query'], documents)
    print(f"\nQuery {i + 1}: {query['query']} - Hand-picked IDs: {query['relevant_ids']}")
    print("-" * 20)
    # print(f"\nQuery: {query['query']} - Hand-picked IDs: {query['relevant_ids']}")
    # print("-" * 20)
    # for idx in top_k_indices:
    #     print(f"Doc {idx + 1}: {documents[idx]['text']}",
    #           f"- Topic: {documents[idx]['metadata']['topic']}",
    #           f"- Similarity: {similarity_scores[i]:.2f}")