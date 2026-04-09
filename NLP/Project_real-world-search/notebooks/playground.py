#%%
from datasets import load_from_disk
import pandas as pd

#%%
ds = load_from_disk(r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\real-world-search\data\raw\ag_news")

df = pd.DataFrame(ds["train"])

text = df['text'][0]

#%%
sampled_docs = []

for label_id in range(4):
    label_subset = ds['train'].filter(lambda x: x['label'] == label_id)
    sampled = label_subset.shuffle(seed=42).select(range(2000))
    sampled_docs.extend(sampled)

df = pd.DataFrame(sampled_docs)
df['label'].value_counts()