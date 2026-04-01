'''
Week 1 Goal
- Load and use transformer models without copying blindly
- Extract embeddings and understand what they represent
- Compare classical ML vs transformers (this is key for DS roles)
- Feel comfortable modifying pipelines

Session 1: 2h → Transformer basics (hands-on)
Session 2: 3h → Embeddings
Session 3: 3h → Classification comparison
Session 4: 2h → Reflection + small experiments
'''


# #%% Session 1: Break the hugging face abstraction
# import torch
# import warnings
# warnings.filterwarnings("ignore")
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# sentences = [
#     'Weather gets better today.',
#     'I will go to the supermarket later.',
#     'Hopefully I can find some vegetables there for my guinea pigs.']

# # -----------------------------------
# # Task 1: Load & inspect a model
# # -----------------------------------
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
# print(tokenized_inputs['input_ids'].shape)  # (batch_size, sequence_length)

# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# with torch.no_grad():
#     outputs = model(**tokenized_inputs)
#     probabilities = F.softmax(outputs.logits, dim=1)
#     indices = torch.argmax(outputs.logits, dim=1)
#     labels = [model.config.id2label[ind] for ind in indices.tolist()]
#     for i, (label, prob) in enumerate(zip(labels, probabilities)):
#         ind = indices[i].numpy()
#         print(f"Label: {label}, Probability: {prob[ind]:.4f}")

# base_model = AutoModel.from_pretrained(model_name)
# with torch.no_grad():
#     outputs = base_model(**tokenized_inputs)
#     last_hidden_state = outputs.last_hidden_state
#     print(f"Last hidden state shape: {last_hidden_state.shape}")

# # -----------------------------------
# # Task 2: Compare with pipeline
# # -----------------------------------
# from transformers import pipeline

# classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
# results = classifier(sentences)
# for result in results:
#     print(result)

# #%% Session 2: Embeddings
# from sentence_transformers import SentenceTransformer

# model_name = "all-MiniLM-L6-v2"
# model = SentenceTransformer(model_name)

# # -----------------------------------
# # Task 1: Extract embeddings
# # -----------------------------------
# sentences = [
#     'Weather gets better today.',
#     'The weather is improving today.',
#     'I will go to the supermarket later.',
#     'Hopefully I can find some vegetables there for my guinea pigs.']

# embeddings = model.encode(sentences)
# print(embeddings.shape)  # (batch_size, embedding_dim)

# # -----------------------------------
# # Task 2: Similarity comparison
# # -----------------------------------
# from sklearn.metrics.pairwise import cosine_similarity

# similar_sentences = [
#     'Weather gets better today.',
#     'The weather is improving today.']

# related_sentences = [
#     'I will go to the supermarket later.',
#     'Grocery store has fresh vegetables for my guinea pigs.']

# unrelated_sentences = [
#     'I will go to the supermarket later.',
#     'The stock market is volatile today.']

# similar_embeddings = model.encode(similar_sentences)
# related_embeddings = model.encode(related_sentences)
# unrelated_embeddings = model.encode(unrelated_sentences)

# # model.similarity() computes cosine similarity between two sets of embeddings
# # If a matrix is passed, it computes pairwise cosine similarity between all rows
# similarity_similar = model.similarity(similar_embeddings[0], similar_embeddings[1])
# similarity_related = model.similarity(related_embeddings[0], related_embeddings[1])
# similarity_unrelated = model.similarity(unrelated_embeddings[0], unrelated_embeddings[1])

# print(f"Similarity (similar sentences): {similarity_similar.numpy()[0,0]:.4f}\n",
#       f"Similarity (related sentences): {similarity_related.numpy()[0,0]:.4f}\n",
#       f"Similarity (unrelated sentences): {similarity_unrelated.numpy()[0,0]:.4f}")

# # --------------------------------------------------
# # Task 3: Small experiment with sentence rewording
# # --------------------------------------------------
# sentences = ['I like dogs.', 'I don\'t like dogs.', 'I hate dogs.']
# embeddings = model.encode(sentences)
# similarity = model.similarity(embeddings, embeddings)
# print(similarity.numpy())


# #%% Session 3: Classification comparison
# import os
# from concurrent.futures import ThreadPoolExecutor
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# from pathlib import Path
# import pandas as pd
# import re
# import string
# import time


# def read_imdb_split(split_dir):
#     def _read_file(path_label):
#         path, label = path_label
#         return path.read_text(encoding='utf-8'), label

#     split_dir = Path(split_dir)
#     tasks = []

#     for label_dir in ['pos', 'neg']:
#         label = 0 if label_dir == 'neg' else 1
#         for path in (split_dir / label_dir).iterdir():
#             tasks.append((path, label))

#     with ThreadPoolExecutor() as executor:
#         results = list(executor.map(_read_file, tasks))

#     texts, labels = zip(*results)
#     return list(texts), list(labels)


# def clean_text(text):
#     '''
#     1. Lowercasing
#        Reduces vocab size and improves statistical reliability.
#     2. Remove text in sqaure brackets
#        Reduces noise and vocal size, can be skipped in modern NLP.
#     3. Remove links
#        Remove or replace by URL because links rarely matter individually.
#     4. Remove punctuation
#        Simplifies tokenization and reduces noise. It may be kept for sentiment tasks.
#     5. Remove words with nummbers
#        Numbers often add noise. May be kept for fraud detection and financial NLP.
#     '''
#     text = str(text).lower()
#     text = re.sub('\[.*?\]', '', text)
#     text = re.sub('https?://\S+|www\.\S+', '', text)
#     text = re.sub('<.*?>+', '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\n', '', text)
#     text = re.sub('\w*\d\w*', '', text)
#     return text


# root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\data\aclImdb'

# texts, labels = read_imdb_split(os.path.join(root_dir, 'train'))
# df = pd.DataFrame({'text': texts, 'label': labels})

# ######### Classical ML pipeline: TF-IDF + Logistic Regression
# X_train, X_val, y_train, y_val = train_test_split(
#     df['text'], df['label'], test_size=0.2, random_state=42)

# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=10000)
# classifier = LogisticRegression(max_iter=1000)

# pipe = Pipeline(steps=[
#     ('tfidf', vectorizer),
#     ('clf', classifier)
# ])

# start = time.time()
# pipe.fit(X_train, y_train)
# end = time.time()

# y_pred = pipe.predict(X_val)

# print(f"Training time of classical ML pipeline: {end - start:.4f} seconds")
# print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
# print(classification_report(y_val, y_pred))

# ######### Feature-based transformer pipeline: Extract embeddings + Logistic Regression
# from sentence_transformers import SentenceTransformer

# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# X_train, X_val, y_train, y_val = train_test_split(
#     df['text'], df['label'], test_size=0.2, random_state=42)

# start = time.time()
# X_train_emb = embedder.encode(X_train.tolist(), batch_size=32, show_progress_bar=True)
# X_val_emb = embedder.encode(X_val.tolist(), batch_size=32, show_progress_bar=True)

# classifier = LogisticRegression(max_iter=1000)
# classifier.fit(X_train_emb, y_train)
# end = time.time()

# y_pred = classifier.predict(X_val_emb)
# print(f"Training time of feature-based transformer pipeline: {end - start:.4f} seconds")
# print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
# print(classification_report(y_val, y_pred))

# ######## Fine-tuning transformer pipeline: DistilBERT
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
# from torch.utils.data import Dataset, DataLoader
# import torch

# class IMDbDataset(Dataset):
#     def __init__(self, encodings, labels):
#         super().__init__()
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, index):
#         # Keys in the encodings: input_ids, attention_mask
#         item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[index])
#         return item

#     def __len__(self):
#         return len(self.labels)


# model_name = 'distilbert-base-uncased'
# max_length = 256
# batch_size = 16
# lr = 2e-5
# num_epochs = 3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# X_train, X_val, y_train, y_val = train_test_split(
#     df['text'], df['label'], test_size=0.2, random_state=42)

# tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
# train_encodings = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=max_length)
# val_encodings = tokenizer(X_val.tolist(), padding=True, truncation=True, max_length=max_length)

# train_dataset = IMDbDataset(train_encodings, y_train.tolist())
# val_dataset = IMDbDataset(val_encodings, y_val.tolist())

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
# model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# model.train()
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")
#     for batch in train_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)

#         optimizer.zero_grad()

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

# model.eval()
# all_preds = []
# with torch.no_grad():
#     for batch in val_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         preds = torch.argmax(logits, dim=1).cpu().numpy()
#         all_preds.extend(preds)

# print(f"Accuracy: {accuracy_score(y_val, all_preds):.4f}")
# print(classification_report(y_val, all_preds))


#%% Session 4: Think like a practitioner
# -----------------------------------
# Task 1: Load & inspect a model
# -----------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
import pandas as pd

#%%
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#%%
short_sentences = [
    'movie good',
    'movie bad']

weird_sentences = [
    'movie good',
    'movie not so gut',
    'movie is good and bad',
    'movie is good but bad']

long_sentences = [
    'Go to the supermarkets' * 100 + 'movie good',
    'Go to the supermarkets' * 100 + 'movie bad',
]

tokenized_inputs = tokenizer(long_sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
outputs = model(**tokenized_inputs)

probabilities = F.softmax(outputs.logits, dim=1)
labels = [model.config.id2label[ind] for ind in torch.argmax(outputs.logits, dim=1).tolist()]

df = pd.DataFrame({
    'text': long_sentences,
    'label': labels,
    'probability': probabilities.max(dim=1).values.tolist()
})

print(df)

embedded_texts = embedder.encode(long_sentences)
similarity_matrix = embedder.similarity(embedded_texts, embedded_texts)
print("Similarity matrix:\n", similarity_matrix.numpy())

#%%
# ----------------------------------------------------
# Task 2: Normalize embeddings and compare similarity
# ----------------------------------------------------
import numpy as np
from sentence_transformers import SentenceTransformer

sentences = [
    'Short sentence',
    'This is a long sentence with more stop words and punctuations!!!'
]

model_name = "paraphrase-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

embeddings = embedder.encode(sentences, normalize_embeddings=False)
similarity = embedder.similarity(embeddings, embeddings)
print("Similarity before normalization:\n", similarity.numpy())

embeddings = embedder.encode(sentences, normalize_embeddings=True)
similarity = embedder.similarity(embeddings, embeddings)
print("Similarity after normalization:\n", similarity.numpy())

#%%
# ----------------------------------------------------
# Task 3: Change pooling in sentence transformer
# ----------------------------------------------------
import numpy as np
from sentence_transformers import SentenceTransformer, models

word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')
sentences = ['I like dogs.', 'I don\'t like dogs.', 'I hate dogs.']

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_cls_token=False,
    pooling_mode_mean_tokens=True,
    pooling_mode_max_tokens=False)

model_mean = SentenceTransformer(modules=[word_embedding_model, pooling_model])

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_cls_token=True,
    pooling_mode_mean_tokens=False,
    pooling_mode_max_tokens=False)

model_cls = SentenceTransformer(modules=[word_embedding_model, pooling_model])