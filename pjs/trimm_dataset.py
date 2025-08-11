import json
import os
from datasets import load_dataset
from loguru import logger
from pjs.dataset import Data
from datasets import load_dataset


### DATASET STATS (LMentry_de)###
data = Data()
data.collect_data("data/LMentry_de")
data.split(0.8)
data.save_dataset_state("data/dataset_splits")

token_counts = data.get_token_count()
logger.info(token_counts)


### REMOVAL ###
ds = load_dataset("bbunzeck/babylm-german")
trimed_ds = {'train': {'text': []}}
removed = 0
for sentence in ds['train']['text']:
    split = sentence.split()
    if removed < token_counts['train_token_count']:
        removed += len(split)
    else:
        trimed_ds['train']['text'].append(sentence)

logger.info(f'{removed} tokens removed')

### SAVE AS TXT ###
output_path = "data/baby_LM/trimmed_babylm_de.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for sentence in trimed_ds['train']['text']:
        f.write(sentence.strip() + "\n")

logger.info(f"Trimmed dataset saved to {output_path} (as .txt)")

### DATASET STATS (LMentry_en)###
data = Data()
data.collect_data("data/LMentry_en")
data.split(0.8)
data.save_dataset_state("data/dataset_splits")

token_counts = data.get_token_count()
logger.info(token_counts)


ds = load_dataset(
    "text", 
    data_files={"train": "data/baby_LM/train_10M/*.train"}
)

### REMOVAL ###

trimed_ds = {'train': {'text': []}}
removed = 0
for sentence in ds['train']['text']:
    split = sentence.split()
    if removed < token_counts['train_token_count']:
        removed += len(split)
    else:
        trimed_ds['train']['text'].append(sentence)

logger.info(f'{removed} tokens removed')

### SAVE AS TXT ###
output_path = "data/baby_LM/trimmed_babylm_en.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for sentence in trimed_ds['train']['text']:
        f.write(sentence.strip() + "\n")

logger.info(f"Trimmed dataset saved to {output_path} (as .txt)")