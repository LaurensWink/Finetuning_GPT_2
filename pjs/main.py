import os
from loguru import logger
import torch
from pjs.dataset import Data
from transformers import PreTrainedTokenizerFast

from pjs.eval import evaluate
from pjs.test import test_model, test_model_outlines
from pjs.train import finetune_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device loaded: {device}')

data = Data()
data.load_split("data/dataset_splits/LMentry_de")

### DATASET STATS ###
token_counts = data.get_token_count()
logger.info(f'The dataset contains {token_counts} tokens')

### CHAR BASE-MODEL DE ###
BASE_MODEL_NAME_DE = "data/models/baby_lm_de_char/model/final"
tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/char_tokenizer")
split_data = data.split_data

for task in split_data:
    test_data = split_data[task]['test']
    test_model_outlines(BASE_MODEL_NAME_DE, tokenizer, test_data, 'data/outputs_outlines/base_model_de', task.split('.')[0], True)

# ### CHAR MODEL FINETUNIG AND TESTS DE ###
# tokenised_dict = data.get_tokenised_dict(tokenizer)
# merged_data = data.merge_tokenised_dict(tokenised_dict, tokenizer)

# finetune_model(BASE_MODEL_NAME, tokenizer, merged_data, "data/models/full_data_train_de_char", 1000, 50, 3, device)
               
# for key in tokenised_dict:
#   finetune_model(BASE_MODEL_NAME, tokenizer, tokenised_dict[key], f"data/models/task_data_train_de_char/{key}", 1000, 10, 3, device)

FULL_DATA_FINETUNED_MODEL_PATH_DE = "data/models/full_data_train_de_char/checkpoint-17100"

for task in split_data:
    test_data = split_data[task]['test']
    test_model(FULL_DATA_FINETUNED_MODEL_PATH_DE, tokenizer, test_data, 'data/outputs_raw/full_data_train_de_char', task.split('.')[0], 25, True, device)

TASK_DATA_FINETUNED_MODEL_DIR_DE = "data/models/task_data_train_de_char"

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_DE):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_DE, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")
        test_data = split_data[str(subfolder)]['test']
        test_model(checkpoint_path, tokenizer, test_data, 'data/outputs_raw/task_data_train_de_char', str(subfolder).split('.')[0], 25, True, device)


for task in split_data:
    test_data = split_data[task]['test']
    test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH_DE, tokenizer, test_data, 'data/outputs_outlines/full_data_train_de_char', task.split('.')[0], True)

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_DE):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_DE, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900") 
        test_data = split_data[str(subfolder)]['test']
        test_model_outlines(checkpoint_path, tokenizer, test_data, 'data/outputs_outlines/task_data_train_de_char', str(subfolder).split('.')[0], True)

### CHAR BASE-MODEL EN ###
BASE_MODEL_NAME_EN = "data/models/baby_lm_en_char/model/final"
tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/char_tokenizer")
split_data = data.split_data

for task in split_data:
    test_data = split_data[task]['test']
    test_model_outlines(BASE_MODEL_NAME_EN, tokenizer, test_data, 'data/outputs_outlines/base_model_en', task.split('.')[0], True)

# ### CHAR MODEL FINETUNIG AND TESTS EN ###
# tokenised_dict = data.get_tokenised_dict(tokenizer)
# merged_data = data.merge_tokenised_dict(tokenised_dict, tokenizer)

# finetune_model(BASE_MODEL_NAME, tokenizer, merged_data, "data/models/full_data_train_en_char", 1000, 50, 3, device)
               
# for key in tokenised_dict:
#   finetune_model(BASE_MODEL_NAME, tokenizer, tokenised_dict[key], f"data/models/task_data_train_en_char/{key}", 1000, 10, 3, device)

FULL_DATA_FINETUNED_MODEL_PATH_EN = "data/models/full_data_train_en_char/checkpoint-17100"

for task in split_data:
    test_data = split_data[task]['test']
    test_model(FULL_DATA_FINETUNED_MODEL_PATH_DE, tokenizer, test_data, 'data/outputs_raw/full_data_train_en_char', task.split('.')[0], 25, True, device)

TASK_DATA_FINETUNED_MODEL_DIR_EN = "data/models/task_data_train_en_char"

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_EN):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_EN, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")
        test_data = split_data[str(subfolder)]['test']
        test_model(checkpoint_path, tokenizer, test_data, 'data/outputs_raw/task_data_train', str(subfolder).split('.')[0], 25, True, device)


for task in split_data:
    test_data = split_data[task]['test']
    test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH_DE, tokenizer, test_data, 'data/outputs_outlines/full_data_train_en_char', task.split('.')[0], True)

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_EN):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_EN, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900") 
        test_data = split_data[str(subfolder)]['test']
        test_model_outlines(checkpoint_path, tokenizer, test_data, 'data/outputs_outlines/task_data_train_en_char', str(subfolder).split('.')[0], True)

###BPE BASE-MODEL EN###
BASE_BPE_MODEL_NAME_EN ='data/models/baby_lm_en_bpe'
tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/bpe_en_tokenizer")
split_data = data.split_data

for task in split_data:
    test_data = split_data[task]['test']
    test_model_outlines(BASE_BPE_MODEL_NAME_EN, tokenizer, test_data, 'data/outputs_outlines/base_model_en_bpe', task.split('.')[0], False)

###BPE MODEL FINETUNIG AND TESTS EN###
tokenised_dict = data.get_tokenised_dict(tokenizer)
merged_data = data.merge_tokenised_dict(tokenised_dict, tokenizer)

finetune_model(BASE_BPE_MODEL_NAME_EN, tokenizer, merged_data, "data/models/full_data_train_en_bpe", 500, 50, 3, device)
               
for key in tokenised_dict:
  finetune_model(BASE_BPE_MODEL_NAME_EN, tokenizer, tokenised_dict[key], f"data/models/task_data_train_en_bpe/{key}", 100, 10, 3, device)

FULL_DATA_FINETUNED_MODEL_PATH_BPE_EN = "data/models/full_data_train_en_bpe/checkpoint-17100"

for task in split_data:
    test_data = split_data[task]['test']
    test_model(FULL_DATA_FINETUNED_MODEL_PATH_BPE_EN, tokenizer, test_data, 'data/outputs_raw/full_data_train_en_bpe', task.split('.')[0], 5, False, device)

TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN = "data/models/task_data_train_en_bpe"

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")
        test_data = split_data[str(subfolder)]['test']
        test_model(checkpoint_path, tokenizer, test_data, 'data/outputs_raw/task_data_train_en_bpe', str(subfolder).split('.')[0], 5, False, device)

for task in split_data:
    test_data = split_data[task]['test']
    test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH_BPE_EN, tokenizer, test_data, 'data/outputs_outlines/full_data_train_en_bpe', task.split('.')[0], False)

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")
        test_data = split_data[str(subfolder)]['test']
        test_model_outlines(checkpoint_path, tokenizer, test_data, 'data/outputs_outlines/task_data_train_en_bpe', str(subfolder).split('.')[0], False)

###BPE BASE-MODEL DE###
BASE_BPE_MODEL_NAME_DE ='data/models/baby_lm_de_bpe'
tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/bpe_de_tokenizer")
split_data = data.split_data

for task in split_data:
    test_data = split_data[task]['test']
    test_model_outlines(BASE_BPE_MODEL_NAME_DE, tokenizer, test_data, 'data/outputs_outlines/base_model_de_bpe', task.split('.')[0], False)

###BPE MODEL FINETUNIG AND TESTS DE###
tokenised_dict = data.get_tokenised_dict(tokenizer)
merged_data = data.merge_tokenised_dict(tokenised_dict, tokenizer)

finetune_model(BASE_BPE_MODEL_NAME_DE, tokenizer, merged_data, "data/models/full_data_train_de_bpe", 500, 50, 3, device)
               
for key in tokenised_dict:
  finetune_model(BASE_BPE_MODEL_NAME_DE, tokenizer, tokenised_dict[key], f"data/models/task_data_train_de_bpe/{key}", 100, 10, 3, device)

FULL_DATA_FINETUNED_MODEL_PATH_BPE_DE = "data/models/full_data_train_de_bpe/checkpoint-17100"

for task in split_data:
    test_data = split_data[task]['test']
    test_model(FULL_DATA_FINETUNED_MODEL_PATH_BPE_DE, tokenizer, test_data, 'data/outputs_raw/full_data_train_de_bpe', task.split('.')[0], 5, False, device)

TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE = "data/models/task_data_train_de_bpe"

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")
        test_data = split_data[str(subfolder)]['test']
        test_model(checkpoint_path, tokenizer, test_data, 'data/outputs_raw/task_data_train_de_bpe', str(subfolder).split('.')[0], 5, False, device)

for task in split_data:
    test_data = split_data[task]['test']
    test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH_BPE_DE, tokenizer, test_data, 'data/outputs_outlines/full_data_train_de_bpe', task.split('.')[0], False)

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")
        test_data = split_data[str(subfolder)]['test']
        test_model_outlines(checkpoint_path, tokenizer, test_data, 'data/outputs_outlines/task_data_train_de_bpe', str(subfolder).split('.')[0], False)

###EVALUATION###
OUTPUT_OUTLINES_DIR = "data/outputs_outlines"
evaluate(OUTPUT_OUTLINES_DIR , "outlines_data_results")