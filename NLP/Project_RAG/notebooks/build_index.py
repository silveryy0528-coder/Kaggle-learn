#%%
import sys
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')
from src.chunking import chunk_text_with_metadata, ChunkingSentencesConfig
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

raw_text = read_pdf_file(pdf_file)
chunks = chunk_text_with_metadata(
    raw_text,
    source=os.path.basename(pdf_file),
    settings=ChunkingSentencesConfig())

#%%
embedder = load_embedder()
texts = [c['text'] for c in chunks]
embeddings = embed_text(
    embedder,
    texts,
    device='cuda')
print(embeddings.shape)

index = build_faiss_index(embeddings, settings=FaissFlatConfig())
print(index.ntotal)

chunk_file = os.path.join(data_folder.replace('raw', 'processed'), 'chunks.pkl')
with open(chunk_file, 'wb') as f:
    pickle.dump(chunks, f)

embedding_file = os.path.join(data_folder.replace('raw', 'index'), 'embeddings.npy')
np.save(embedding_file, embeddings)

index_file = os.path.join(data_folder.replace('raw', 'index'))
faiss.write_index(index, index_file, 'faiss.index')
