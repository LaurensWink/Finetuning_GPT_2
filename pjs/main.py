import os
from loguru import logger
import torch
from pjs.dataset import Data
from transformers import AutoTokenizer, GPT2LMHeadModel

from pjs.test import test_model
from pjs.train import finetune_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device loaded: {device}')

data = Data()
# data.collect_data("data/LMentry")
# data.split(0.8)
# data.save_dataset_state("data/dataset_splits")
data.load_split("data/dataset_splits/LMentry_split")

###CHAR MODEL FINETUNIG AND TESTS###
# BASE_MODEL_NAME = "phonemetransformers/GPT2-85M-CHAR-TXT"
# model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_NAME).to(device)
# tokenizer = AutoTokenizer.from_pretrained('phonemetransformers/babble-tokenizers', subfolder='BABYLM-TOKENIZER-CHAR-TXT')

# split_data = data.split_data
# tokenised_dict = data.get_tokenised_dict(tokenizer)
# merged_data = data.merge_tokenised_dict(tokenised_dict, tokenizer)

# finetune_model(model, tokenizer, merged_data, "data/models/full_data_train", 250, 100, 3)
               
# for key in tokenised_dict:
#   model = model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_NAME).to(device)
#   finetune_model(model, tokenizer, tokenised_dict[key], f"data/models/task_data_train/{key}", 100, 10, 3)

# FULL_DATA_FINETUNED_MODEL_PATH = "data/models/full_data_train/checkpoint-22320"
# model = GPT2LMHeadModel.from_pretrained(FULL_DATA_FINETUNED_MODEL_PATH)

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model(model, tokenizer, test_data, 'data/outputs_raw/full_data_train', task.split('.')[0], 25, True)

# TASK_DATA_FINETUNED_MODEL_DIR = "data/models/task_data_train"

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "checkpoint-900") # der letzte homophones checkpoint wurde manuell von checkpoint-720 zu checkpoint-900 umbenannt (der task hat weniger daten als die anderen)
#         model = GPT2LMHeadModel.from_pretrained(checkpoint_path).to(device)
#         test_data = split_data[str(subfolder)]['test']
#         test_model(model, tokenizer, test_data, 'data/outputs_raw/task_data_train', str(subfolder).split('.')[0], 25, True)


###BPE MODEL FINETUNIG AND TESTS###
# BASE_BPE_MODEL_NAME = "phonemetransformers/GPT2-85M-BPE-TXT"
# model = GPT2LMHeadModel.from_pretrained(BASE_BPE_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained('phonemetransformers/babble-tokenizers', subfolder='BABYLM-TOKENIZER-BPE-TXT')
split_data = data.split_data
# tokenised_dict = data.get_tokenised_dict(tokenizer)
# merged_data = data.merge_tokenised_dict(tokenised_dict, tokenizer)

# finetune_model(model, tokenizer, merged_data, "data/models/full_data_train_BPE", 250, 100, 3)
               
# for key in tokenised_dict:
#   model = model = GPT2LMHeadModel.from_pretrained(BASE_BPE_MODEL_NAME).to(device)
#   finetune_model(model, tokenizer, tokenised_dict[key], f"data/models/task_data_train_BPE/{key}", 100, 10, 3)

FULL_DATA_FINETUNED_MODEL_PATH = "data/models/full_data_train_BPE/checkpoint-22320"
model = GPT2LMHeadModel.from_pretrained(FULL_DATA_FINETUNED_MODEL_PATH)

for task in split_data:
    test_data = split_data[task]['test']
    test_model(model, tokenizer, test_data, 'data/outputs_raw/full_data_train_BPE', task.split('.')[0], 5, False)

TASK_DATA_FINETUNED_MODEL_DIR = "data/models/task_data_train_BPE"

for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR):
        subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR, subfolder)
        checkpoint_path = os.path.join(subfolder_path, "checkpoint-900")# der letzte homophones checkpoint wurde manuell von checkpoint-720 zu checkpoint-900 umbenannt (der task hat weniger daten als die anderen)
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path).to(device)
        test_data = split_data[str(subfolder)]['test']
        test_model(model, tokenizer, test_data, 'data/outputs_raw/task_data_train_BPE', str(subfolder).split('.')[0], 5, False)