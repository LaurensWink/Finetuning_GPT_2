from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from pjs.dataset import Data
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from transformers import pipeline

data = Data()
df = pd.DataFrame(data.files[Path("data\word_not_containing.json")])
print(df)
dataset = Dataset.from_pandas(df[['input']])


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Nutzt das EOS-Token als Padding


def tokenize_function(examples):
    # Tokenize die Eingabe
    inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
    # Labels für kausales Sprachmodell (Zieltext ist die gleiche Eingabe wie der Input, aber als Label)
    inputs["labels"] = inputs["input_ids"].copy()  # Labels sind die gleichen wie input_ids
    return inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)


model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    evaluation_strategy="no", 
    save_strategy="epoch",  
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=None,
    tokenizer=tokenizer,
)



print(torch.cuda.is_available())

trainer.train()

model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

generator = pipeline("text-generation", model="./gpt2-finetuned")
print(generator("Write a word that doesn't contain the letter", max_length=30))
