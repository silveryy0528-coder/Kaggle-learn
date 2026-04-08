#%%
import os
import pandas as pd
import warnings
import numpy as np
import transformers
import logging
import torch
import datasets
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import StratifiedGroupKFold


warnings.simplefilter('ignore')
logging.disable(logging.WARNING)


def tok_func(x):
    return tokenizer(x["inputs"], truncation=True, add_special_tokens=True)


def corr(eval_pred):
    return {'pearson': np.corrcoef(*eval_pred)[0][1]}


lr = 8e-5
bs = 128
wd = 0.1
epochs = 4


def get_dds(df):
    ds = Dataset.from_pandas(df).rename_column('score', 'label')
    inps = "anchor", "target", "context"
    tok_ds = ds.map(
        tok_func,
        batched=True,
        remove_columns=inps + ('inputs', 'id'))
    return DatasetDict({
        "train":tok_ds.select(trn_indices),
        "test": tok_ds.select(val_indices)})


def get_trainer(dds):
    args = TrainingArguments(
        'outputs',
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        fp16=True,
        eval_strategy="epoch",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs,
        weight_decay=wd,
        report_to='none'
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
    model.float()
    return Trainer(
        model,
        args,
        train_dataset=dds['train'],
        eval_dataset=dds['test'],
        compute_metrics=corr,
        data_collator=data_collator
    )


#%%
root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\data\us-patent-phrase-to-phrase-matching'
train_df = pd.read_csv(os.path.join(root_dir, 'train.csv'))
eval_df = pd.read_csv(os.path.join(root_dir, 'test.csv'))

#%%
model_nm = 'microsoft/deberta-v3-small'

# Pretrained models assume that text is tokenized in a particular way; hence,
# the tokenizer must match the model being used.
tokenizer = AutoTokenizer.from_pretrained(model_nm)
train_df['inputs'] = (
    'TEXT1: ' + train_df.context +
    '; TEXT2: ' + train_df.target +
    '; ANC1: ' + train_df.anchor
)
train_df['inputs'] = train_df.inputs.str.lower()

#%%
# Create randomly shuffled list of anchors
anchors = train_df.anchor.unique()
np.random.seed(42)
np.random.shuffle(anchors)

# Take 25% anchors and put the corresponding items in validation set
val_prop = 0.25
val_size = int(len(anchors) * val_prop)
val_anchors = anchors[:val_size]

is_val = np.isin(train_df.anchor, val_anchors)
indices = np.arange(len(train_df))
val_indices = indices[is_val]
trn_indices = indices[~is_val]

dds = get_dds(train_df)

trainer = get_trainer(dds)

#%%
trainer.train()

#%%
n_folds = 4
cv = StratifiedGroupKFold(n_splits=n_folds)
train_df = train_df.sample(frac=1, random_state=42)
scores = (train_df.score * 100).astype(int)
folds = list(cv.split(indices, scores, train_df.anchor))

def get_fold(folds, fold_num):
    trn, val = folds[fold_num]
    ds = Dataset.from_pandas(train_df).rename_column('score', 'label')
    inps = "anchor", "target", "context"
    tok_ds = ds.map(
        tok_func,
        batched=True,
        remove_columns=inps + ('inputs', 'id'))
    return DatasetDict({"train":tok_ds.select(trn), "test": tok_ds.select(val)})

dds = get_fold(folds, 0)
