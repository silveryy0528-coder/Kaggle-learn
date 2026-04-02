#%%
import json
import os


def load_queries(root_dir):
    file_path = os.path.join(root_dir, './data/queries.json')
    with open(file_path, 'r') as f:
        queries = json.load(f)
    return queries


def load_documents(root_dir):
    file_path = os.path.join(root_dir, './data/documents.json')
    with open(file_path, 'r') as f:
        documents = json.load(f)
    return documents



# if __name__ == "__main__":
#     root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\semantic-search'
#     documents = load_documents(root_dir)
#     queries = load_queries(root_dir)
#     print(f"Loaded {len(documents)} documents.")
#     print(f"Loaded {len(queries)} queries.")