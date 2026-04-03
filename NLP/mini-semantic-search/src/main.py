#%%
import sys
import time
import importlib
sys.path.append(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\mini-semantic-search\src')

import utils
importlib.reload(utils)
import tfidf as main


if __name__ == "__main__":
    root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\mini-semantic-search'

    # Load documents
    documents = utils.load_documents(root_dir)
    queries = utils.load_queries(root_dir)

    vectorizer, doc_vectors = main.vectorize(documents)

    start = time.time()
    for i, query in enumerate(queries):
        top_k_indices, similarity_scores = main.search(
            query['query'], vectorizer, doc_vectors)
        print(f"\nQuery {i + 1}: {query['query']} - Hand-picked IDs: {query['relevant_ids']}")
        print("-" * 20)
        for idx in top_k_indices:
            print(f"Doc {idx + 1}: {documents[idx]['text']}",
                f"- Topic: {documents[idx]['metadata']['topic']}",
                f"- Similarity: {similarity_scores[top_k_indices.tolist().index(idx)]:.2f}")
    end = time.time()
    print(f"\nTotal search time: {end - start:.2f} seconds")