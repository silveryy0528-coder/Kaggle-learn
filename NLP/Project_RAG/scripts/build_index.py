#%%
import sys
import glob
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')
from core.ingestion import chunk_multiple_documents, ChunkingSentenceConfig
from core.embedding import embed_text, load_embedder
from core.retrieval import build_faiss_index, FaissFlatConfig
import pickle
import numpy as np
import faiss
import os


#%%
pdf_files = glob.glob(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\raw\*.pdf')
data_folder = os.path.dirname(pdf_files[0])

chunk_settings = ChunkingSentenceConfig(chunk_size=400, chunk_overlap=50)
chunks = chunk_multiple_documents(pdf_files, chunk_settings)
print(f"Total chunks created: {len(chunks)}")

#%%
embedder = load_embedder()
texts = [c.text for c in chunks]
embeddings = embed_text(
    embedder,
    texts,
    device='cuda')
faiss.normalize_L2(embeddings)

index = build_faiss_index(embeddings, settings=FaissFlatConfig())
print(f"FAISS index built with {index.ntotal} vectors")

#%%
chunk_file = os.path.join(data_folder.replace('raw', 'processed'), 'chunks.pkl')
with open(chunk_file, 'wb') as f:
    pickle.dump(chunks, f)

embedding_file = os.path.join(data_folder.replace('raw', 'index'), 'embeddings.npy')
np.save(embedding_file, embeddings)

index_file = os.path.join(data_folder.replace('raw', 'index'), 'faiss.index')
faiss.write_index(index, index_file)
