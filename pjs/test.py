import csv
import os
import re
from transformers import LlamaForCausalLM
import outlines
import unicodedata
from tokenizers.normalizers import Lowercase, Sequence, Replace, NFC, Strip

normalizer = Sequence([
        NFC(),
        Lowercase(),
        Strip(),
    ])

def test_model(model_name, tokenizer, test_data, output_dir, file_name, max_new_tokens, char, device):
    model = LlamaForCausalLM.from_pretrained(model_name).to(device)

    for index, row in test_data.iterrows():
        input = row["input"] + tokenizer.eos_token

        inputs = tokenizer(
        input,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=200,
        return_attention_mask=True
        ).to(device)
        
        outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
        )
        
        output_text_raw = tokenizer.decode(outputs[0])
        output_split = [part for part in output_text_raw.split(tokenizer.eos_token) if part]
        output_text = output_split[-1]
        if char:
            output_text = output_text.replace(" ", "")
        output_text = output_text.replace("W", " ")

        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f'{file_name}.csv')

        file_exists = os.path.isfile(csv_path)

        expected = normalizer.normalize_str(str(row["output"]))

        with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Input", "Options", "Expected", "Output (raw)", "Predicted"])
            writer.writerow([row["input"],row["options"], expected, output_text_raw, output_text])

def test_model_outlines(model_name, tokenizer, test_data, output_dir, file_name, char):
    model = outlines.models.transformers(model_name)
    for index, row in test_data.iterrows():
        input = f'{tokenizer.eos_token} {row["input"]} {tokenizer.eos_token}'
        options = [normalizer.normalize_str(str(opt)) for opt in row["options"]]
        generator = outlines.generate.choice(model, options)
        expected = normalizer.normalize_str(str(row["output"]))
        output_text = generator(input)
        if char:
            output_text = output_text.replace(" ", "")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f'{file_name}.csv')
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Input", "Options", "Expected", "Predicted",])
            writer.writerow([row["input"], options, expected, output_text])