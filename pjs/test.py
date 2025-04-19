import csv
import os


def test_model(model, tokenizer, test_data, output_dir, file_name, max_new_tokens, char):
    for index, row in test_data.iterrows():
        input = row["input"] + tokenizer.eos_token

        inputs = tokenizer(
        input,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=200,
        return_attention_mask=True
        )
        
        outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
        )
        
        output_text = tokenizer.decode(outputs[0])
        output_text = output_text.split("UTT_BOUNDARY")[2]
        output_text = output_text.replace("UTT_BOUNDARY", "")
        if char:
            output_text = output_text.replace(" ", "")
        output_text = output_text.replace("W", " ")

        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f'{file_name}.csv')

        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Input", "Expected", "Predicted"])
            writer.writerow([row["input"], row["output"], output_text])

def test_model_outlines(model, tokenizer, test_data, output_dir, file_name):
    for index, row in test_data.iterrows():
        input = row["input"] + tokenizer.eos_token
        print(row['metadata'])