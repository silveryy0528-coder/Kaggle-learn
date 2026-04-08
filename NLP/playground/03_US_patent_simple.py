#%%
import os
import numpy as np
import torch
import pandas as pd
from torch.optim import AdamW
from torchmetrics.regression import PearsonCorrCoef
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer, DataCollatorWithPadding


#%%
def tokenize_function(x):
    return tokenizer(
        x['input'],
        truncation=True,
        max_length=256
)


def corr(x, y):
    '''Return the correlation coefficient between two vectors'''
    return np.corrcoef(x, y)[0][1]


def corr_dict(eval_pred):
    return {'pearson': corr(*eval_pred)}


batch_size = 64
epochs = 5
lr = 8e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\data\us-patent-phrase-to-phrase-matching'
train_df = pd.read_csv(os.path.join(root_dir, 'train.csv'))
train_df['input'] = (
    'TEXT1: ' + train_df.context +
    '; TEXT2: ' + train_df.target +
    '; ANC1: ' + train_df.anchor
)

dataset = Dataset.from_pandas(train_df)

model_nm = 'microsoft/deberta-v3-small'
tokenizer = AutoTokenizer.from_pretrained(model_nm)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(
    ['id', 'anchor', 'target', 'context', 'input']
)
tokenized_dataset = tokenized_dataset.rename_column('score', 'labels')

data_dict = tokenized_dataset.train_test_split(0.25, seed=42)

#%%
test_df = pd.read_csv(os.path.join(root_dir, 'test.csv'))
test_df['input'] = (
    'TEXT1: ' + test_df.context +
    '; TEXT2: ' + test_df.target +
    '; ANC1: ' + test_df.anchor
)
test_dataset = Dataset.from_pandas(test_df)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = tokenized_test_dataset.remove_columns(
    ['id', 'anchor', 'target', 'context', 'input']
)

#%%
model = AutoModelForSequenceClassification.from_pretrained(
    model_nm, num_labels=1
)
model = model.float()
model.to(device)

data_collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(
    data_dict['train'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)
val_loader = DataLoader(
    data_dict['test'],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
)

optimizer = AdamW(model.parameters(), lr=lr)
pearson_metric = PearsonCorrCoef().to(device)

train_losses = np.full(epochs, np.nan)
train_corrs = np.full(epochs, np.nan)
val_losses = np.full(epochs, np.nan)
val_corrs = np.full(epochs, np.nan)

for epoch in range(epochs):
    model.train()

    pearson_metric.reset()
    running_loss = 0.0

    for batch_id, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.shape[0]

        preds = outputs.logits.squeeze(-1)
        targets = labels.squeeze(-1)
        pearson_metric.update(preds, targets)

    epoch_corr = pearson_metric.compute().item()
    epoch_loss = running_loss / len(train_loader.dataset)

    train_losses[epoch] = epoch_loss
    train_corrs[epoch] = epoch_corr

    model.eval()
    val_loss = 0.0
    pearson_metric.reset()

    with torch.no_grad():
        for batch_id, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)

            loss = outputs.loss
            val_loss += loss.item() * input_ids.shape[0]
            preds = outputs.logits.squeeze(-1)
            targets = labels.squeeze(-1)
            pearson_metric.update(preds, targets)

    val_corr = pearson_metric.compute().item()
    val_loss = val_loss / len(val_loader.dataset)

    val_losses[epoch] = val_loss
    val_corrs[epoch] = val_corr

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {epoch_loss:.4f}, Train Corr: {epoch_corr:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Corr: {val_corr:.4f}")


#%%
args = TrainingArguments(
    'outputs',
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    fp16=True,
    eval_strategy='epoch',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to='none'
)
'''
The classification task is treated as a regression task with single number output.
You could treat it as 5 classes, but regression is better when:
    1. The labels have an ordinal meaning
    2. The distance between values matters
    3. Correlation is the evaluation metric
'''
model = AutoModelForSequenceClassification.from_pretrained(
    model_nm, num_labels=1
)
model = model.float()
data_collator = DataCollatorWithPadding(tokenizer)
trainer = Trainer(
    model,
    args,
    train_dataset=data_dict['train'],
    eval_dataset=data_dict['test'],
    compute_metrics=corr_dict,
    data_collator=data_collator
)
trainer.train()

#%%
preds = trainer.predict(tokenized_test_dataset).predictions.astype(float)
preds