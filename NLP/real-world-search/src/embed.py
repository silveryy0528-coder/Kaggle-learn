#%%
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os


def load_chunks(file_path):
    with open(file_path, 'r') as f:
        chunks = json.load(f)
    return chunks


chunk_file = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\real-world-search\data\processed\chunks.json'
chunks = load_chunks(chunk_file)

df = pd.DataFrame(chunks)
texts = df['text'].to_list()

#%%
model_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name, device='cuda')

batch_size = 512
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = embedder.encode(
        batch_texts,
        device='cuda',
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    all_embeddings.append(batch_embeddings)
    np.save(
        os.path.join(os.path.dirname(chunk_file), f'embeddings_batch_{i//batch_size}.npy'),
        batch_embeddings
    )
