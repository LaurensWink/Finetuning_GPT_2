import os
import torch
from datasets import load_dataset, DatasetDict
import pandas as pd
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast, 
    set_seed, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    LlamaForCausalLM, 
    LlamaConfig)


def pretrain(training_files: list[str], tokenizer, output_dir, device):
    # --- DATASET ---
    raw_datasets = load_dataset('text', data_files=training_files)
    split_datasets = raw_datasets['train'].train_test_split(test_size=0.05, seed=42)
    final_datasets = DatasetDict({'train': split_datasets['train'], 'validation': split_datasets['test']})
    # --- TOKENIZER ---
    context_length = 64

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True)
        
        input_batch = [input_ids for length, input_ids in zip(outputs["length"], outputs["input_ids"])]
        return {"input_ids": input_batch}

    tokenized_datasets = final_datasets.map(tokenize, 
                                            batched=True, 
                                            remove_columns=final_datasets["train"].column_names)
    # --- MODEL CONFIG ---
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,         
        num_hidden_layers=12,          
        intermediate_size=3072,            
        num_attention_heads=12,           
        max_position_embeddings=256,       
        bos_token_id=tokenizer.convert_tokens_to_ids("UTT_BOUNDARY"),
        eos_token_id=tokenizer.convert_tokens_to_ids("UTT_BOUNDARY"),
        pad_token_id=tokenizer.convert_tokens_to_ids("PAD")
    )

    set_seed(42)
    model = LlamaForCausalLM(config)

    # DEVICE INFO
    model.to(device)
    print(f"Training on device: {device}")
    print(f"Model parameters: {model.num_parameters()}")

    # --- TRAINING ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f'{output_dir}/model',
        overwrite_output_dir=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        num_train_epochs=5,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        learning_rate=3e-4,
        logging_steps=10,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    trainer.train()

    # --- LOGGING ---
    df = pd.DataFrame(trainer.state.log_history)
    os.makedirs(f'{output_dir}/logs', exist_ok=True)
    df.to_csv(f'{output_dir}/logs/losses.csv')  

    # --- FINAL SAVE ---
    trainer.save_model(f'{output_dir}/model/final')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_CHAR = AutoTokenizer.from_pretrained("data/tokenizer/char_tokenizer")
# TOKENIZER_BPE_DE = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/bpe_de_tokenizer")
# TOKENIZER_BPE_EN = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/bpe_en_tokenizer")

LM_CHAR_DE = 'data/models/baby_lm_de_char'
LM_CHAR_EN = 'data/models/baby_lm_en_char'
LM_BPE_DE = 'data/models/baby_lm_de_bpe'
LM_BPE_EN = 'data/models/baby_lm_en_bpe'
DATA_DE = ["data/baby_LM/trimmed_babylm_de.txt"]
DATA_EN = ["data/baby_LM/trimmed_babylm_en.txt"]

# pretrain(DATA_DE, TOKENIZER_CHAR, LM_CHAR_DE, device)
pretrain(DATA_EN, TOKENIZER_CHAR, LM_CHAR_EN, device)
# pretrain(DATA_DE, TOKENIZER_BPE_DE, LM_BPE_DE, device)
# pretrain(DATA_EN, TOKENIZER_BPE_EN, LM_BPE_EN, device)