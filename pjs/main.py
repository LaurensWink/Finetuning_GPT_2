import os
from loguru import logger
from pjs.mblimp import Language, mblimp
import torch
from pjs.dataset import Data
from transformers import LlamaTokenizerFast

from transformers import AutoTokenizer, AutoModelForCausalLM

from pjs.eval import eval_mblimp, evaluate
from pjs.test import test_model, test_model_outlines
from pjs.train import finetune_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device loaded: {device}')

data = Data()
data.load_split("data/dataset_splits/LMentry_de")
split_data = data.split_data

### DATASET STATS DE ###
token_counts = data.get_token_count()
logger.info(f'The dataset contains {token_counts} tokens')

### BASELINE MODEL DE ###
# BASELINE_MODEL_NAME_DE_01 = "LSX-UniWue/LLaMmlein_1B"
# TOKENIZER_BASELINE_DE_01 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_01)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_DE_01, TOKENIZER_BASELINE_DE_01, test_data, 'data/outputs_outlines/baseline_LLaMmlein_1B_de', task.split('.')[0], False)

# BASELINE_MODEL_NAME_DE_02 = "meta-llama/Llama-3.2-1B"
# TOKENIZER_BASELINE_DE_02 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_02)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_DE_02, TOKENIZER_BASELINE_DE_02, test_data, 'data/outputs_outlines/baseline_Llama-3.2-1B_de', task.split('.')[0], False)

# BASELINE_MODEL_NAME_DE_03 = "meta-llama/Llama-3.2-1B-Instruct"
# TOKENIZER_BASELINE_DE_03 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_03)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_DE_03, TOKENIZER_BASELINE_DE_03, test_data, 'data/outputs_outlines/baseline_Llama-3.2-1B-Instruct_de', task.split('.')[0], False)

# BASELINE_MODEL_NAME_DE_04 = "HuggingFaceTB/SmolLM2-135M"
# TOKENIZER_BASELINE_DE_04 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_04)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_DE_04, TOKENIZER_BASELINE_DE_04, test_data, 'data/outputs_outlines/baseline_SmolLM2-135M_de', task.split('.')[0], False)

# BASELINE_MODEL_NAME_DE_05 = "HuggingFaceTB/SmolLM2-135M-Instruct"
# TOKENIZER_BASELINE_DE_05 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_05)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_DE_05, TOKENIZER_BASELINE_DE_05, test_data, 'data/outputs_outlines/baseline_SmolLM2-135M-Instruct_de', task.split('.')[0], False)

# BASELINE_MODEL_NAME_DE_06 = "HuggingFaceTB/SmolLM2-360M"
# TOKENIZER_BASELINE_DE_06 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_06)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_DE_06, TOKENIZER_BASELINE_DE_06, test_data, 'data/outputs_outlines/baseline_SmolLM2-360M_de', task.split('.')[0], False)

# BASELINE_MODEL_NAME_DE_07 = "HuggingFaceTB/SmolLM2-360M-Instruct"
# TOKENIZER_BASELINE_DE_07 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_07)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_DE_07, TOKENIZER_BASELINE_DE_07, test_data, 'data/outputs_outlines/baseline_SmolLM2-360M-Instruct_de', task.split('.')[0], False)

# BASELINE_MODEL_NAME_DE_08 = "HuggingFaceTB/SmolLM2-1.7B"
# TOKENIZER_BASELINE_DE_08 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_08)
# for task in split_data: 
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_DE_08, TOKENIZER_BASELINE_DE_08, test_data, 'data/outputs_outlines/baseline_SmolLM2-1.7B_de', task.split('.')[0], False)

# BASELINE_MODEL_NAME_DE_09 = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# TOKENIZER_BASELINE_DE_09 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_09)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_DE_09, TOKENIZER_BASELINE_DE_09, test_data, 'data/outputs_outlines/baseline_SmolLM2-1.7B-Instruct_de', task.split('.')[0], False)

BASELINE_MODEL_NAME_DE_10 = "meta-llama/Llama-3.2-3B"
TOKENIZER_BASELINE_DE_10 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_10)

# for task in split_data:
#         test_data = split_data[task]['test']
#         test_model_outlines(BASELINE_MODEL_NAME_DE_10, TOKENIZER_BASELINE_DE_10, test_data, 'data/outputs_outlines/baseline_Llama-3.2-3B_de', task.split('.')[0], False)

BASELINE_MODEL_NAME_DE_11 = "meta-llama/Llama-3.2-3B-Instruct"
TOKENIZER_BASELINE_DE_11 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_DE_11)
# for task in split_data:
#         test_data = split_data[task]['test']
#         test_model_outlines(BASELINE_MODEL_NAME_DE_11, TOKENIZER_BASELINE_DE_11, test_data, 'data/outputs_outlines/baseline_Llama-3.2-3B-Instruct_de', task.split('.')[0], False)

### CHAR BASE-MODEL DE ###
# BASE_CHAR_MODEL_NAME_DE = "data/models/baby_lm_de_char/model/final"
# TOKENIZER_CHAR = LlamaTokenizerFast.from_pretrained("data/tokenizer/char_tokenizer")
# split_data = data.split_data

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASE_CHAR_MODEL_NAME_DE, TOKENIZER_CHAR, test_data, 'data/outputs_outlines/base_model_de_char', task.split('.')[0], True)

# ## CHAR MODEL FINETUNIG AND TESTS DE ###
# tokenised_dict = data.get_tokenised_dict(TOKENIZER_CHAR)
# merged_data = data.merge_tokenised_dict(tokenised_dict, TOKENIZER_CHAR)

# finetune_model(BASE_CHAR_MODEL_NAME_DE, TOKENIZER_CHAR , merged_data, "data/models/full_data_train_de_char", 1000, 50, 3, device)
               
# for key in tokenised_dict:
#   finetune_model(BASE_CHAR_MODEL_NAME_DE, TOKENIZER_CHAR , tokenised_dict[key], f"data/models/task_data_train_de_char/{key}", 1000, 10, 3, device)

# FULL_DATA_FINETUNED_MODEL_PATH_CHAR_DE = "data/models/full_data_train_de_char/final"

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model(FULL_DATA_FINETUNED_MODEL_PATH_CHAR_DE, TOKENIZER_CHAR, test_data, 'data/outputs_raw/full_data_train_de_char', task.split('.')[0], 25, True, device)

# TASK_DATA_FINETUNED_MODEL_DIR_DE = "data/models/task_data_train_de_char"

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_DE):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_DE, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "final")
#         test_data = split_data[str(subfolder)]['test']
#         test_model(checkpoint_path, TOKENIZER_CHAR, test_data, 'data/outputs_raw/task_data_train_de_char', str(subfolder).split('.')[0], 25, True, device)


# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH_CHAR_DE, TOKENIZER_CHAR, test_data, 'data/outputs_outlines/full_data_train_de_char', task.split('.')[0], True)

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_DE):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_DE, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "final") 
#         test_data = split_data[str(subfolder)]['test']
#         test_model_outlines(checkpoint_path, TOKENIZER_CHAR, test_data, 'data/outputs_outlines/task_data_train_de_char', str(subfolder).split('.')[0], True)

# ###BPE BASE-MODEL DE###
# BASE_BPE_MODEL_NAME_DE = 'data/models/baby_lm_de_bpe/model/final'
# TOKENIZER_DE_BPE = LlamaTokenizerFast.from_pretrained("data/tokenizer/bpe_de_tokenizer")
# split_data = data.split_data

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASE_BPE_MODEL_NAME_DE, TOKENIZER_DE_BPE, test_data, 'data/outputs_outlines/base_model_de_bpe', task.split('.')[0], False)

# ###BPE MODEL FINETUNIG AND TESTS DE###
# tokenised_dict = data.get_tokenised_dict(TOKENIZER_DE_BPE)
# merged_data = data.merge_tokenised_dict(tokenised_dict, TOKENIZER_DE_BPE)

# finetune_model(BASE_BPE_MODEL_NAME_DE, TOKENIZER_DE_BPE, merged_data, "data/models/full_data_train_de_bpe", 1000, 50, 3, device)
               
# for key in tokenised_dict:
#   finetune_model(BASE_BPE_MODEL_NAME_DE, TOKENIZER_DE_BPE, tokenised_dict[key], f"data/models/task_data_train_de_bpe/{key}", 1000, 10, 3, device)

# FULL_DATA_FINETUNED_MODEL_PATH_BPE_DE = "data/models/full_data_train_de_bpe/final"

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model(FULL_DATA_FINETUNED_MODEL_PATH_BPE_DE, TOKENIZER_DE_BPE, test_data, 'data/outputs_raw/full_data_train_de_bpe', task.split('.')[0], 5, False, device)

# TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE = "data/models/task_data_train_de_bpe"

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "final")
#         test_data = split_data[str(subfolder)]['test']
#         test_model(checkpoint_path, TOKENIZER_DE_BPE, test_data, 'data/outputs_raw/task_data_train_de_bpe', str(subfolder).split('.')[0], 5, False, device)

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH_BPE_DE, TOKENIZER_DE_BPE, test_data, 'data/outputs_outlines/full_data_train_de_bpe', task.split('.')[0], False)

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_BPE_DE, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "final")
#         test_data = split_data[str(subfolder)]['test']
#         test_model_outlines(checkpoint_path, TOKENIZER_DE_BPE, test_data, 'data/outputs_outlines/task_data_train_de_bpe', str(subfolder).split('.')[0], False)

### DATASET STATS EN ###
data.load_split("data/dataset_splits/LMentry_en")
split_data = data.split_data
token_counts = data.get_token_count()
logger.info(f'The dataset contains {token_counts} tokens')

### BASELINE MODEL EN ###
# BASELINE_MODEL_NAME_EN_01 = "meta-llama/Llama-3.2-1B"
# TOKENIZER_BASELINE_EN_01 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_01)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_EN_01, TOKENIZER_BASELINE_EN_01, test_data, 'data/outputs_outlines/baseline_Llama-3.2-1B_en', task.split('.')[0], False)

# BASELINE_MODEL_NAME_EN_02 = "meta-llama/Llama-3.2-1B-Instruct"
# TOKENIZER_BASELINE_EN_02 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_02)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_EN_02, TOKENIZER_BASELINE_EN_02, test_data, 'data/outputs_outlines/baseline_Llama-3.2-1B-Instruct_en', task.split('.')[0], False)

# BASELINE_MODEL_NAME_EN_03 = "HuggingFaceTB/SmolLM2-135M"
# TOKENIZER_BASELINE_EN_03 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_03)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_EN_03, TOKENIZER_BASELINE_EN_03, test_data, 'data/outputs_outlines/baseline_SmolLM2-135M_en', task.split('.')[0], False)

# BASELINE_MODEL_NAME_EN_04 = "HuggingFaceTB/SmolLM2-135M-Instruct"
# TOKENIZER_BASELINE_EN_04 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_04)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_EN_04, TOKENIZER_BASELINE_EN_04, test_data, 'data/outputs_outlines/baseline_SmolLM2-135M-Instruct_en', task.split('.')[0], False)

# BASELINE_MODEL_NAME_EN_05 = "HuggingFaceTB/SmolLM2-360M"
# TOKENIZER_BASELINE_EN_05 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_05)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_EN_05, TOKENIZER_BASELINE_EN_05, test_data, 'data/outputs_outlines/baseline_SmolLM2-360M_en', task.split('.')[0], False)

# BASELINE_MODEL_NAME_EN_06 = "HuggingFaceTB/SmolLM2-360M-Instruct"
# TOKENIZER_BASELINE_EN_06 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_06)
# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASELINE_MODEL_NAME_EN_06, TOKENIZER_BASELINE_EN_06, test_data, 'data/outputs_outlines/baseline_SmolLM2-360M-Instruct_en', task.split('.')[0], False)

# BASELINE_MODEL_NAME_EN_07 = "HuggingFaceTB/SmolLM2-1.7B"
# TOKENIZER_BASELINE_EN_07 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_07)
# for task in split_data:
#         test_data = split_data[task]['test']
#         test_model_outlines(BASELINE_MODEL_NAME_EN_07, TOKENIZER_BASELINE_EN_07, test_data, 'data/outputs_outlines/baseline_SmolLM2-1.7B_en', task.split('.')[0], False)




# BASELINE_MODEL_NAME_EN_08 = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# TOKENIZER_BASELINE_EN_08 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_08)
# for task in split_data:
#         test_data = split_data[task]['test']
#         test_model_outlines(BASELINE_MODEL_NAME_EN_08, TOKENIZER_BASELINE_EN_08, test_data, 'data/outputs_outlines/baseline_SmolLM2-1.7B-Instruct_en', task.split('.')[0], False)

BASELINE_MODEL_NAME_EN_09 = "meta-llama/Llama-3.2-3B"
TOKENIZER_BASELINE_EN_09 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_09)
# for task in split_data:
#         test_data = split_data[task]['test']
#         test_model_outlines(BASELINE_MODEL_NAME_EN_09, TOKENIZER_BASELINE_EN_09, test_data, 'data/outputs_outlines/baseline_Llama-3.2-3B_en', task.split('.')[0], False)

BASELINE_MODEL_NAME_EN_10 = "meta-llama/Llama-3.2-3B-Instruct"
TOKENIZER_BASELINE_EN_10 = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME_EN_10)
# for task in split_data:
#         test_data = split_data[task]['test']
#         test_model_outlines(BASELINE_MODEL_NAME_EN_10, TOKENIZER_BASELINE_EN_10, test_data, 'data/outputs_outlines/baseline_Llama-3.2-3B-Instruct_en', task.split('.')[0], False)




### CHAR BASE-MODEL EN ###
# BASE_CHAR_MODEL_NAME_EN = "data/models/baby_lm_en_char/model/final"

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASE_CHAR_MODEL_NAME_EN, TOKENIZER_CHAR, test_data, 'data/outputs_outlines/base_model_en_char', task.split('.')[0], True)

# ### CHAR MODEL FINETUNIG AND TESTS EN ###
# tokenised_dict = data.get_tokenised_dict(TOKENIZER_CHAR)
# merged_data = data.merge_tokenised_dict(tokenised_dict, TOKENIZER_CHAR)

# finetune_model(BASE_CHAR_MODEL_NAME_EN, TOKENIZER_CHAR, merged_data, "data/models/full_data_train_en_char", 1000, 50, 3, device)
               
# for key in tokenised_dict:
#   finetune_model(BASE_CHAR_MODEL_NAME_EN, TOKENIZER_CHAR, tokenised_dict[key], f"data/models/task_data_train_en_char/{key}", 1000, 10, 3, device)

# FULL_DATA_FINETUNED_MODEL_PATH_CHAR_EN = "data/models/full_data_train_en_char/final"

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model(FULL_DATA_FINETUNED_MODEL_PATH_CHAR_EN, TOKENIZER_CHAR, test_data, 'data/outputs_raw/full_data_train_en_char', task.split('.')[0], 25, True, device)

# TASK_DATA_FINETUNED_MODEL_DIR_EN = "data/models/task_data_train_en_char"

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_EN):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_EN, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "final")
#         test_data = split_data[str(subfolder)]['test']
#         test_model(checkpoint_path, TOKENIZER_CHAR, test_data, 'data/outputs_raw/task_data_train_en_char', str(subfolder).split('.')[0], 25, True, device)

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH_CHAR_EN, TOKENIZER_CHAR, test_data, 'data/outputs_outlines/full_data_train_en_char', task.split('.')[0], True)

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_EN):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_EN, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "final") 
#         test_data = split_data[str(subfolder)]['test']
#         test_model_outlines(checkpoint_path, TOKENIZER_CHAR, test_data, 'data/outputs_outlines/task_data_train_en_char', str(subfolder).split('.')[0], True)

# ###BPE BASE-MODEL EN###
# BASE_BPE_MODEL_NAME_EN ='data/models/baby_lm_en_bpe/model/final'
# TOKENIZER_EN_BPE = LlamaTokenizerFast.from_pretrained("data/tokenizer/bpe_en_tokenizer")

# split_data = data.split_data

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(BASE_BPE_MODEL_NAME_EN, TOKENIZER_EN_BPE, test_data, 'data/outputs_outlines/base_model_en_bpe', task.split('.')[0], False)

# ###BPE MODEL FINETUNIG AND TESTS EN###
# tokenised_dict = data.get_tokenised_dict(TOKENIZER_EN_BPE)
# merged_data = data.merge_tokenised_dict(tokenised_dict, TOKENIZER_EN_BPE)

# finetune_model(BASE_BPE_MODEL_NAME_EN, TOKENIZER_EN_BPE, merged_data, "data/models/full_data_train_en_bpe", 1000, 50, 3, device)
               
# for key in tokenised_dict:
#   finetune_model(BASE_BPE_MODEL_NAME_EN, TOKENIZER_EN_BPE, tokenised_dict[key], f"data/models/task_data_train_en_bpe/{key}", 1000, 10, 3, device)

# FULL_DATA_FINETUNED_MODEL_PATH_BPE_EN = "data/models/full_data_train_en_bpe/final"

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model(FULL_DATA_FINETUNED_MODEL_PATH_BPE_EN, TOKENIZER_EN_BPE, test_data, 'data/outputs_raw/full_data_train_en_bpe', task.split('.')[0], 5, False, device)

# TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN = "data/models/task_data_train_en_bpe"

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "final")
#         test_data = split_data[str(subfolder)]['test']
#         test_model(checkpoint_path, TOKENIZER_EN_BPE, test_data, 'data/outputs_raw/task_data_train_en_bpe', str(subfolder).split('.')[0], 5, False, device)

# for task in split_data:
#     test_data = split_data[task]['test']
#     test_model_outlines(FULL_DATA_FINETUNED_MODEL_PATH_BPE_EN, TOKENIZER_EN_BPE, test_data, 'data/outputs_outlines/full_data_train_en_bpe', task.split('.')[0], False)

# for subfolder in os.listdir(TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN):
#         subfolder_path = os.path.join(TASK_DATA_FINETUNED_MODEL_DIR_BPE_EN, subfolder)
#         checkpoint_path = os.path.join(subfolder_path, "final")
#         test_data = split_data[str(subfolder)]['test']
#         test_model_outlines(checkpoint_path, TOKENIZER_EN_BPE, test_data, 'data/outputs_outlines/task_data_train_en_bpe', str(subfolder).split('.')[0], False)

###EVALUATION###

# mblimp(BASE_BPE_MODEL_NAME_DE, TOKENIZER_DE_BPE, Language.DE, device, "baby_lm_de_bpe")
# mblimp(BASE_CHAR_MODEL_NAME_DE, TOKENIZER_CHAR, Language.DE, device, "baby_lm_de_char")
# mblimp(FULL_DATA_FINETUNED_MODEL_PATH_BPE_DE, TOKENIZER_DE_BPE, Language.DE, device, "full_data_train_de_bpe")
# mblimp(FULL_DATA_FINETUNED_MODEL_PATH_CHAR_DE, TOKENIZER_CHAR, Language.DE, device, "full_data_train_de_char")
# mblimp(BASELINE_MODEL_NAME_DE_01, TOKENIZER_BASELINE_DE_01, Language.DE, device, "baseline_LLaMmlein_1B_de")
# mblimp(BASELINE_MODEL_NAME_DE_02, TOKENIZER_BASELINE_DE_02, Language.DE, device, "baseline_Llama-3.2-1B_de")
# mblimp(BASELINE_MODEL_NAME_DE_03, TOKENIZER_BASELINE_DE_03, Language.DE, device, "baseline_Llama-3.2-1B-Instruct_de")
# mblimp(BASELINE_MODEL_NAME_DE_04, TOKENIZER_BASELINE_DE_04, Language.DE, device, "baseline_SmolLM2-135M_de")
# mblimp(BASELINE_MODEL_NAME_DE_05, TOKENIZER_BASELINE_DE_05, Language.DE, device, "baseline_SmolLM2-135M-Instruct_de")
# mblimp(BASELINE_MODEL_NAME_DE_06, TOKENIZER_BASELINE_DE_06, Language.DE, device, "baseline_SmolLM2-360M_de")
# mblimp(BASELINE_MODEL_NAME_DE_07, TOKENIZER_BASELINE_DE_07, Language.DE, device, "baseline_SmolLM2-360M-Instruct_de")
# mblimp(BASELINE_MODEL_NAME_DE_08, TOKENIZER_BASELINE_DE_07, Language.DE, device, "baseline_SmolLM2-1.7B_de")
# mblimp(BASELINE_MODEL_NAME_DE_09, TOKENIZER_BASELINE_DE_08, Language.DE, device, "baseline_SmolLM2-1.7B-Instruct_de")

mblimp(BASELINE_MODEL_NAME_DE_10, TOKENIZER_BASELINE_DE_10, Language.DE, device, "baseline_Llama-3.2-3B_de")
mblimp(BASELINE_MODEL_NAME_DE_11, TOKENIZER_BASELINE_DE_11, Language.DE, device, "baseline_Llama-3.2-3B-Instruct_de")

# mblimp(BASE_BPE_MODEL_NAME_EN, TOKENIZER_EN_BPE, Language.EN, device, "baby_lm_en_bpe")
# mblimp(BASE_CHAR_MODEL_NAME_EN, TOKENIZER_CHAR, Language.EN, device, "baby_lm_en_char")
# mblimp(FULL_DATA_FINETUNED_MODEL_PATH_BPE_EN, TOKENIZER_EN_BPE, Language.EN, device, "full_data_train_en_bpe")
# mblimp(FULL_DATA_FINETUNED_MODEL_PATH_CHAR_EN, TOKENIZER_CHAR, Language.EN, device, "full_data_train_en_char")
# mblimp(BASELINE_MODEL_NAME_EN_01, TOKENIZER_BASELINE_EN_01, Language.EN, device, "baseline_Llama-3.2-1B_en")
# mblimp(BASELINE_MODEL_NAME_EN_02, TOKENIZER_BASELINE_EN_02, Language.EN, device, "baseline_Llama-3.2-1B-Instruct_en")
# mblimp(BASELINE_MODEL_NAME_EN_03, TOKENIZER_BASELINE_EN_03, Language.EN, device, "baseline_SmolLM2-135M_en")
# mblimp(BASELINE_MODEL_NAME_EN_04, TOKENIZER_BASELINE_EN_04, Language.EN, device, "baseline_SmolLM2-135M-Instruct_en")
# mblimp(BASELINE_MODEL_NAME_EN_05, TOKENIZER_BASELINE_EN_05, Language.EN, device, "baseline_SmolLM2-360M_en")
# mblimp(BASELINE_MODEL_NAME_EN_06, TOKENIZER_BASELINE_EN_06, Language.EN, device, "baseline_SmolLM2-360M-Instruct_en")
# mblimp(BASELINE_MODEL_NAME_EN_07, TOKENIZER_BASELINE_EN_07, Language.EN, device, "baseline_SmolLM2-1.7B_en")
# mblimp(BASELINE_MODEL_NAME_EN_08, TOKENIZER_BASELINE_EN_08, Language.EN, device, "baseline_SmolLM2-1.7B-Instruct_en")


mblimp(BASELINE_MODEL_NAME_EN_09, TOKENIZER_BASELINE_EN_09, Language.EN, device, "baseline_Llama-3.2-3B_en")
mblimp(BASELINE_MODEL_NAME_EN_10, TOKENIZER_BASELINE_EN_10, Language.EN, device, "baseline_Llama-3.2-3B-Instruct_en")



MBLIMP_DIR = "data/mblimp"
eval_mblimp(MBLIMP_DIR)
OUTPUT_OUTLINES_DIR = "data/outputs_outlines"
evaluate(OUTPUT_OUTLINES_DIR)