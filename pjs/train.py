
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

def finetune_model(model, tokenizer, input_data, output_dir):
    class CustomDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                key: val[idx] for key, val in self.encodings.items()
            }
        
    input_encodings = tokenizer(input_data['input'].tolist(), return_tensors="pt", padding=True, truncation=True)
    label_encodings = tokenizer(input_data['output'].tolist(), return_tensors="pt", padding=True, truncation=True)

    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": label_encodings["input_ids"],
    }

    dataset = CustomDataset(encodings)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()