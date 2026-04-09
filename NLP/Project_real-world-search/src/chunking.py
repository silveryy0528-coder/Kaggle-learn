#%%
from datasets import load_from_disk
import json
import os


def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


def sample_docs_by_label(ds, num_samples_per_label=2000):
    sampled_docs = []

    for label_id in range(4):
        label_subset = ds.filter(lambda x: x['label'] == label_id)
        sampled = label_subset.shuffle(seed=42).select(
            range(num_samples_per_label))
        sampled_docs.extend(sampled)

    return sampled_docs


#%%
data_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\real-world-search\data'
ds = load_from_disk(os.path.join(data_folder, "raw/ag_news"))
label_names = ds['train'].features['label'].names
sampled_docs = sample_docs_by_label(ds['train'], num_samples_per_label=5000)

chunk_id = 0
chunk_list = []

for doc_id, doc in enumerate(sampled_docs):
    text = doc['text']
    chunks = chunk_text(text, chunk_size=100)
    for chunk in chunks:
        chunk_list.append({
            'chunk_id': chunk_id,
            'text': chunk,
            'doc_id': doc_id,
            'label': doc['label'],
            'label_name': label_names[doc['label']]
        })
        chunk_id += 1


with open(os.path.join(data_folder, "processed/chunks.json"), 'w') as f:
    json.dump(chunk_list, f, indent=4)

stats = {
    'total_docs': len(sampled_docs),
    'total_chunks': len(chunk_list),
    "chunks_per_label": {label_names[i]: sum(1 for chunk in chunk_list if chunk['label'] == i) for i in range(4)}
}

with open(os.path.join(data_folder, "processed/chunk_stats.json"), 'w') as f:
    json.dump(stats, f, indent=4)
