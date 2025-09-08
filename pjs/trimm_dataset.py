import json
import os
from datasets import load_dataset
from loguru import logger
from pjs.dataset import Data
from datasets import load_dataset

MAX_TOKENS = 10000000

### DATASET STATS (LMentry_de)###
data = Data()
data.collect_data("data/LMentry_de")
data.split(0.8)
data.save_dataset_state("data/dataset_splits")

LMentry_de_token_count = data.get_token_count()
logger.info(LMentry_de_token_count)


### REMOVAL ###
ds = load_dataset("bbunzeck/babylm-german")
# Neues gekürztes Dataset
trimmed_ds = {'train': {'text': []}}
current_tokens = 0

# Zeilen durchgehen und sammeln, bis Token-Limit erreicht ist
for sentence in ds['train']['text']:
    split = sentence.split()
    token_count = len(split)
    if current_tokens + token_count > MAX_TOKENS:
        break
    trimmed_ds['train']['text'].append(sentence)
    current_tokens += token_count

trimmed_babylm_de_token_count = current_tokens
logger.info(f'{trimmed_babylm_de_token_count} tokens behalten (Limit: {MAX_TOKENS})')

# Als TXT speichern
output_path = "data/baby_LM/trimmed_babylm_de.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for sentence in trimmed_ds['train']['text']:
        f.write(sentence.strip() + "\n")

logger.info(f"Trimmed dataset saved to {output_path} (as .txt)")

### DATASET STATS (LMentry_en)###
data = Data()
data.collect_data("data/LMentry_en")
data.split(0.8)
data.save_dataset_state("data/dataset_splits")

LMentry_en_token_count = data.get_token_count()
logger.info(LMentry_en_token_count)


ds = load_dataset(
    "text", 
    data_files={"train": "data/baby_LM/train_10M/*.train"}
)

### REMOVAL ###
trimmed_ds = {'train': {'text': []}}
current_tokens = 0

for sentence in ds['train']['text']:
    split = sentence.split()
    token_count = len(split)
    if current_tokens + token_count > MAX_TOKENS:
        break
    trimmed_ds['train']['text'].append(sentence)
    current_tokens += token_count

trimmed_babylm_en_token_count = current_tokens
logger.info(f'{trimmed_babylm_en_token_count} tokens behalten (Limit: {MAX_TOKENS})')

output_path = "data/baby_LM/trimmed_babylm_en.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for sentence in trimmed_ds['train']['text']:
        f.write(sentence.strip() + "\n")

logger.info(f"Trimmed dataset saved to {output_path} (as .txt)")

### Stats ###

counts = {
    "LMentry_de_token_count": LMentry_de_token_count,
    "LMentry_en_token_count": LMentry_en_token_count,
    "trimmed_babylm_de_token_count": trimmed_babylm_de_token_count,
    "trimmed_babylm_en_token_count": trimmed_babylm_en_token_count
}

output_path = "data/stats/stats.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(counts, f, ensure_ascii=False, indent=4)

logger.info(f"Stats saved to {output_path}")