import csv
import os
import re
from transformers import GPT2LMHeadModel
import outlines


def test_model(model_name, tokenizer, test_data, output_dir, file_name, max_new_tokens, char, device):
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
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
        
        output_text = tokenizer.decode(outputs[0])
        # because of tokenizer.eos_token == tokenizer.bos_token has to be index == 2 (tokenizer.bos_token input tokenizer.eos_token output tokenizer.eos_token)
        output_text = output_text.split(tokenizer.eos_token)[2]
        output_text = output_text.replace(tokenizer.eos_token, "")
        if char:
            output_text = output_text.replace(" ", "")
        output_text = output_text.replace("W", " ")

        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f'{file_name}.csv')

        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Input", "Options", "Expected", "Predicted"])
            writer.writerow([row["input"],row["options"], row["output"], output_text])

def test_model_outlines(model_name, tokenizer, test_data, output_dir, file_name, char):
    model = outlines.models.transformers(model_name)
    for index, row in test_data.iterrows():
        input = row["input"] + tokenizer.eos_token
        options = row["options"]
        generator = outlines.generate.choice(model, options)
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
            writer.writerow([row["input"], row["options"], row["output"], output_text])