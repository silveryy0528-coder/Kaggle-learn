#%%
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification)
'''
# Large Movie Review Dataset
# https://ai.stanford.edu/~amaas/data/sentiment/

1. Prepare dataset
2. Load pretrained tokenizer, call it with dataset -> encoding
3. Build PyTorch dataset with encodings
4. Load pretrained model
5. Use native PyTorch training pipeline
'''

def read_file(path_label):
    path, label = path_label
    return path.read_text(encoding='utf-8'), label


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    tasks = []

    for label_dir in ['pos', 'neg']:
        label = 0 if label_dir == 'neg' else 1
        for path in (split_dir / label_dir).iterdir():
            tasks.append((path, label))

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(read_file, tasks))

    texts, labels = zip(*results)
    return list(texts), list(labels)


class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        super().__init__()
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        # Keys in the encodings: input_ids, attention_mask
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


#%%
model_name = 'distilbert-base-uncased'
root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\data\aclImdb'
train_texts, train_labels = read_imdb_split(os.path.join(root_dir, 'train'))

#%%
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, padding=True, truncation=True)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 5
model = DistilBertForSequenceClassification.from_pretrained(model_name)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

train_losses = np.full(num_epochs, np.nan)
train_accs = np.full(num_epochs, np.nan)
val_losses = np.full(num_epochs, np.nan)
val_accs = np.full(num_epochs, np.nan)

for epoch in range(num_epochs):
    # ----- Training -----
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for batch_id, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)

        loss = outputs[0]
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.shape[0]
        preds = torch.argmax(outputs.logits, dim=1)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = running_corrects / total

    train_losses[epoch] = train_loss
    train_accs[epoch] = train_acc

    # ----- Validation -----
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs[0]

            val_loss += loss.item() * input_ids.shape[0]
            preds = torch.argmax(outputs.logits, dim=1)
            val_corrects += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_corrects / val_total

    val_losses[epoch] = val_loss
    val_accs[epoch] = val_acc

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Learning rate: {optimizer.param_groups[0]['lr']}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

output_folder = 'outputs'
tokenizer.save_pretrained(output_folder)
model.save_pretrained(output_folder)

#%%
x = range(1, num_epochs + 1)
fig = plt.figure()
fig.add_subplot(2, 1, 1)
plt.plot(x, train_losses, label='training loss')
plt.plot(x, val_losses, label='val loss')
plt.xlabel('epochs')
plt.legend()
fig.add_subplot(2, 1, 2)
plt.plot(x, train_accs, label='training accuracy')
plt.plot(x, val_accs, label='val accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

#%%
model_folder = 'outputs'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_folder)
model = DistilBertForSequenceClassification.from_pretrained(model_folder)

root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\data\aclImdb'
test_texts, test_labels = read_imdb_split(os.path.join(root_dir, 'test'))
test_encodings = tokenizer(test_texts, padding=True, truncation=True)
test_dataset = IMDbDataset(test_encodings, test_labels)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
model.to(device)
model.eval()

test_loss = 0.
test_corrects = 0
test_total = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs[0]

        test_loss += loss.item() * input_ids.shape[0]
        preds = torch.argmax(outputs.logits, dim=1)
        test_corrects += (preds == labels).sum().item()
        test_total += labels.size(0)

    test_loss /= test_total
    test_acc = test_corrects / test_total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, ")