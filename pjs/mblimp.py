import csv
import os
import pandas as pd
from datasets import load_dataset
from enum import Enum
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

class Language(Enum):
    DE = "deu"
    EN = "eng"

def calculate_perplexity(model, tokenizer, sentence: str, device) -> float:
    encodings = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def mblimp(model_name: str, tokenizer, language: Language, device, file_name):
    ds = load_dataset("jumelet/multiblimp", language.value)
    output_metrics = []
    model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    output_file = f'data/mblimp/{file_name}.csv'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for sen, wrong_sen in zip(ds["train"]["sen"], ds["train"]["wrong_sen"]):
        sen_ppl = calculate_perplexity(model, tokenizer, sen, device)
        wrong_sen_ppl = calculate_perplexity(model, tokenizer, wrong_sen, device)

        score = 1 if sen_ppl < wrong_sen_ppl else 0
        output_metrics.append({
            "Sen": sen,
            "Sen_ppl": sen_ppl,
            "Wrong_Sen": wrong_sen,
            "Wrong_Sen_ppl": wrong_sen_ppl,
            "Score": score
        })

    results_df = pd.DataFrame(output_metrics)
    results_df.to_csv(output_file, index=False)