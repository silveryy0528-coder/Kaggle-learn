#%%
import faiss
import sys
from collections import Counter
from itertools import chain
import json
import pickle
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')
from core.embedding import embed_text, load_embedder


def retrieve(query, chunks, embedder, k=3):
    query_emb = embed_text(embedder, [query])
    faiss.normalize_L2(query_emb)
    scores, indices = faiss_index.search(query_emb, k)
    return indices[0].tolist()


def reciprocal_rank(retrieved, relevant):
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1 / rank
    return 0


def recall_at_k(retrieved, relevant):
    return int(any(cid in relevant for cid in retrieved))


def show_top_k(retrieved, k=3):
    flat = list(chain.from_iterable(retrieved))
    counter = Counter(flat)
    topk = counter.most_common(k)
    print(f"\nTop {k} most frequently retrieved chunks:")
    for cid, count in topk:
        print(f"  Chunk {cid}: {count} times")


def evaluate(qa_data, chunks, embedder, k=3):
    rr_scores = []
    recall_scores = []
    retrieved_chunks = []

    for i, item in enumerate(qa_data, start=1):
        print(f"\nProcessing item {i}: {item['question']}")
        question = item['question']
        relevant = item['relevant_chunk_ids']
        retrieved = retrieve(question, chunks, embedder, k)

        print(f"Retrieved: {retrieved}, Relevant: {relevant}")
        rr_scores.append(reciprocal_rank(retrieved, relevant))
        recall_scores.append(recall_at_k(retrieved, relevant))
        retrieved_chunks.append(retrieved)

    mrr = sum(rr_scores) / len(rr_scores)
    recall = sum(recall_scores) / len(recall_scores)
    show_top_k(retrieved_chunks, k)
    
    return mrr, recall


#%%
chunk_file = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\processed\chunks.pkl'
with open(chunk_file, 'rb') as f:
    chunks = pickle.load(f)

qa_file = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\eval\relevance_labels.json"
with open(qa_file, 'r') as f:
    qa_data = json.load(f)

faiss_index_file = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\index\faiss.index"
faiss_index = faiss.read_index(faiss_index_file)

embedder = load_embedder()
mrr, recall = evaluate(qa_data, chunks, embedder, k=5)

print(f"MRR@5: {mrr:.4f}")
print(f"Recall@5: {recall:.4f}")

#%%
for i in [130, 0, 34]:
        print(f'Chunk {i}\n: {chunks[i].text}')