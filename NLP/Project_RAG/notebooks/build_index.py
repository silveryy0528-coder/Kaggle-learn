#%%
import sys
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')
from src.chunking import chunk_text, ChunkingSentencesConfig
from src.ingestion import read_pdf_file
from src.embedding import embed_text, load_embedder
from src.retrieval import build_faiss_index, FaissFlatConfig
import pickle
import numpy as np
import faiss
import os

#%%
pdf_file = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\raw\CV_YanGuo.pdf"
data_folder = os.path.dirname(pdf_file)

texts = read_pdf_file(pdf_file)
chunks = chunk_text(texts, settings=ChunkingSentencesConfig())
print(len(chunks))

embedder = load_embedder()
embeddings = embed_text(embedder, chunks, device='cuda')
print(embeddings.shape)

index = build_faiss_index(embeddings, settings=FaissFlatConfig())
print(index.ntotal)

with open(os.path.join(data_folder.replace('raw', 'processed'), 'chunks.pkl'), 'wb') as f:
    pickle.dump(chunks, f)

np.save(os.path.join(data_folder.replace('raw', 'index'), 'embeddings.npy'), embeddings)

faiss.write_index(index, os.path.join(data_folder.replace('raw', 'index'), 'faiss.index'))
