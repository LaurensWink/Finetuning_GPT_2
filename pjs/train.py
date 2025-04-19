
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

def finetune_model(model, tokenizer, encodings, output_dir, save_steps, save_limit, epochs):
    class CustomDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                key: val[idx] for key, val in self.encodings.items()
            }

    dataset = CustomDataset(encodings)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        num_train_epochs=epochs,
        save_steps=save_steps,
        save_total_limit=save_limit,
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