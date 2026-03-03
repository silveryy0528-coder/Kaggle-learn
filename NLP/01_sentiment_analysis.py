#%%
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

texts = ['I am more complex than I ever wanted to admit.',
         'We sometimes wrap our feelings into words that are not meant for that.',
         'The message I deliver may be interpreted differently when you receive it.']

#%%
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
results = classifier(texts)

for result in results:
    print(result)

#%%
texts_encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

model.to(device)
texts_encodings.to(device)
labels = torch.tensor([1, 0, 0]).to(device)

with torch.no_grad():
    outputs = model(
        **texts_encodings,
        labels=labels
    )
    logits = outputs.logits
    loss = criterion(logits, labels)
    print(texts_encodings['input_ids'].shape[0])
    # predictions = F.softmax(outputs.logits)
    # print(f'Probabilities after softmax:{predictions}')
    # label_ids = torch.argmax(predictions, dim=1)
    # print(f'Labels with highest probabilities:{label_ids}')
    # labels = [model.config.id2label[label_id] for label_id in label_ids.tolist()]
    # print(f'Labels:{labels}')

# output_folder = 'outputs'
# tokenizer.save_pretrained(output_folder)
# model.save_pretrained(output_folder)
