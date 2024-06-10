import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset
import datasets
import json

model_name = 'monologg/koelectra-base-v3-discriminator'
epochs = 10
batch_size = 32
learning_rate = 5e-5
max_length = 128
warmup_ratio = 0.1
log_interval = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataset load
klue_nli = load_dataset('klue', 'nli') 
# train : 24998, validation : 3000
# features: ['guid', 'source', 'premise', 'hypothesis', 'label'],
print("Dataset loaded!")
train_data = klue_nli['train']
test_data = klue_nli['validation'][:300]
test_data = datasets.Dataset.from_dict(test_data)

class NliDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.label = dataset['label']
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item = self.tokenizer(item['premise'], item['hypothesis'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        item['labels'] = torch.tensor(self.label[idx])
        item = {key: val.squeeze(0) for key, val in item.items()}
        return item

    def __len__(self):
        return len(self.dataset)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

test_dataset = NliDataset(test_data, tokenizer)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset))

# inference 
model.eval()
model.to(device)
preds = []
labels = []
for step, data in enumerate(test_dataloader):
    item = {key: val.to(device) for key, val in data.items()}
    outputs = model(**item)
    preds.extend(outputs.logits.argmax(dim=-1).tolist())
print(preds)

# save
json_arr = json.dumps(preds)
with open('huggingface.json', 'w') as f:
    f.write(json_arr)
