import os
from loguru import logger
import torch
from pjs.dataset import Data
from transformers import AutoTokenizer, GPT2LMHeadModel

from pjs.test import test_model, test_model_outlines
from pjs.train import finetune_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device loaded: {device}')

data = Data()
# data.collect_data("data/LMentry")
# data.split(0.8)
# data.save_dataset_state("data/dataset_splits")
data.load_split("data/dataset_splits/LMentry_split")

##CHAR MODEL FINETUNIG AND TESTS###
BASE_MODEL_NAME = "phonemetransformers/GPT2-85M-CHAR-TXT"
tokenizer = AutoTokenizer.from_pretrained('phonemetransformers/babble-tokenizers', subfolder='BABYLM-TOKENIZER-CHAR-TXT')

split_data = data.split_data
tokenised_dict = data.get_tokenised_dict(tokenizer)
merged_data = data.merge_tokenised_dict(tokenised_dict, tokenizer)

# finetune_model(BASE_MODEL_NAME, tokenizer, merged_data, "data/models/full_data_train", 500, 50, 3, device)
               
# for key in tokenised_dict:
#   finetune_model(BASE_MODEL_NAME, tokenizer, tokenised_dict[key], f"data/models/task_data_train/{key}", 100, 10, 3, device)

FULL_DATA_FINETUNED_MODEL_PATH = "data/models/full_data_train/checkpoint-21600"

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model(FULL_DATA_FINETUNED_MODEL_PATH, tokenizer, test_data, 'data/outputs_raw/full_data_train', task.split('.')[0], 25, True, device)

TASK_DATA_FINETUNED_MODEL_DIR = "data/models/task_data_train"

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")
#         test_data = split_data[str(subfolder)]['test']
#         test_model(checkpoint_path, tokenizer, test_data, 'data/outputs_raw/task_data_train', str(subfolder).split('.')[0], 25, True, device)


# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH, tokenizer, test_data, 'data/outputs_outlines/full_data_train', task.split('.')[0], True)

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "checkpoint-900") 
#         test_data = split_data[str(subfolder)]['test']
#         test_model_outlines(checkpoint_path, tokenizer, test_data, 'data/outputs_outlines/task_data_train', str(subfolder).split('.')[0], True)

##BPE MODEL FINETUNIG AND TESTS###
BASE_BPE_MODEL_NAME = "phonemetransformers/GPT2-85M-BPE-TXT"
tokenizer = AutoTokenizer.from_pretrained('phonemetransformers/babble-tokenizers', subfolder='BABYLM-TOKENIZER-BPE-TXT')
split_data = data.split_data
tokenised_dict = data.get_tokenised_dict(tokenizer)
merged_data = data.merge_tokenised_dict(tokenised_dict, tokenizer)

finetune_model(BASE_BPE_MODEL_NAME, tokenizer, merged_data, "data/models/full_data_train_BPE", 500, 50, 3, device)
               
for key in tokenised_dict:
  finetune_model(BASE_BPE_MODEL_NAME, tokenizer, tokenised_dict[key], f"data/models/task_data_train_BPE/{key}", 100, 10, 3, device)

FULL_DATA_FINETUNED_MODEL_PATH = "data/models/full_data_train_BPE/checkpoint-21600"

for task in split_data:
    test_data = split_data[task]['test']
    test_model(FULL_DATA_FINETUNED_MODEL_PATH, tokenizer, test_data, 'data/outputs_raw/full_data_train_BPE', task.split('.')[0], 5, False, device)

TASK_DATA_FINETUNED_MODEL_DIR = "data/models/task_data_train_BPE"

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")
        test_data = split_data[str(subfolder)]['test']
        test_model(checkpoint_path, tokenizer, test_data, 'data/outputs_raw/task_data_train_BPE', str(subfolder).split('.')[0], 5, False, device)

for task in split_data:
    test_data = split_data[task]['test']
    test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH, tokenizer, test_data, 'data/outputs_outlines/full_data_train_BPE', task.split('.')[0], False)

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")
        test_data = split_data[str(subfolder)]['test']
        test_model_outlines(checkpoint_path, tokenizer, test_data, 'data/outputs_outlines/task_data_train_BPE', str(subfolder).split('.')[0], False)