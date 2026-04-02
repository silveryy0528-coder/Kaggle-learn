#%%
import sys
import importlib
sys.path.append(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\semantic-search\src')

import pandas as pd
import numpy as np

import load_data, embedding, tfidf
importlib.reload(load_data)


if __name__ == "__main__":
    root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\semantic-search'

    # Load documents
    documents = load_data.load_documents(root_dir)
    queries = load_data.load_queries(root_dir)

    for i, query in enumerate(queries):
        top_k_indices, similarity_scores = tfidf.main(query['query'], documents)
        print(f"\nQuery {i + 1}: {query['query']} - Hand-picked IDs: {query['relevant_ids']}")
        print("-" * 20)
        for idx in top_k_indices:
            print(f"Doc {idx + 1}: {documents[idx]['text']}",
                f"- Topic: {documents[idx]['metadata']['topic']}",
                f"- Similarity: {similarity_scores[top_k_indices.tolist().index(idx)]:.2f}")
