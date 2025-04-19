import json
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import torch
from transformers.tokenization_utils_base import BatchEncoding



class Data():
    """Class to creat and administer datasets"""
    files: dict[str, pd.DataFrame] = {}
    split_data: dict[str, pd.DataFrame] = {}
    
    def collect_data(self, data_path: str) -> None:
        """Reads all JSON files, extracts the 'examples' key, stores them as DataFrames in self.files and creates an path-enum"""
        pathlist = Path(data_path).glob('**/*.json')
        for path in pathlist:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                
                if 'examples' in json_data:
                    examples = json_data['examples']
                    
                    data = [{
                        "input": example['input'].replace('\n', ' '),
                        "output": example['metadata'].get('answer', None),
                        "options": self.extract_choices(example['metadata'])
                    }
                    for example in examples.values()]

                    data_frame = pd.DataFrame(data)
                    
                    self.files[path.name] = data_frame
                    logger.info(f"Successfully loaded 'examples' from: {path}")
                else:
                    logger.warning(f"'examples' key not found in {path}")
            
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON in {path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error with {path}: {e}")


    def extract_choices(self, metadata):
        if "n1" in metadata and "n2" in metadata:
            return [str(metadata["n1"]), str(metadata["n2"])]
        
        elif "word1" in metadata and "word2" in metadata:
            return [metadata["word1"], metadata["word2"]]
        
        elif "answer" in metadata and "distractor" in metadata:
            return [metadata["answer"], metadata["distractor"]]
        
        elif "answer" in metadata and "distractors" in metadata:
            return [metadata["answer"]] + metadata["distractors"]
        
        elif "sentence" in metadata and "answer" in metadata:
            words = metadata["sentence"].split()
            if metadata["answer"] in words:
                return words 
        
        elif "word" in metadata and "answer" in metadata:
            return list(metadata["word"])

        logger.warning(f"No valid choice found!")
        return []

    def split(self, train_size: float) -> None:
        """Method to split the Data in train and test. The train_size parameter determines the percentage of samples used for traing"""
        self.split_data = {}

        for path, df in self.files.items():
            try:
                train_df, test_df = train_test_split(df, train_size=train_size, shuffle=True, random_state=42)

                self.split_data[path] = {
                    "train": train_df.reset_index(drop=True),
                    "test": test_df.reset_index(drop=True)
                }

                logger.info(f"Split for {path}: {len(train_df)} train / {len(test_df)} test rows")

            except Exception as e:
                logger.error(f"Error splitting data for {path}: {e}")

    def save_dataset_state(self, directory: str) -> None:
        """
        Saves the current split datasets (train/test) as JSON files
        into a new timestamped subdirectory inside the given base directory.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = os.path.join(directory, f"split_{timestamp}")
        os.makedirs(target_dir, exist_ok=True)

        if not self.split_data:
            logger.warning("No split data found. Did you call .split() or .load_split() first?")
            return

        for path, split in self.split_data.items():
            base_filename = Path(path).stem

            for split_name, df in split.items():
                filename = f"{base_filename}_{split_name}.json"
                save_path = os.path.join(target_dir, filename)

                try:
                    records = df.to_dict(orient="records")
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(records, f, ensure_ascii=False, indent=2)

                    logger.info(f"Saved {split_name} set to {save_path}")

                except Exception as e:
                    logger.error(f"Failed to save {split_name} set for {path}: {e}")
    
    def load_split(self, directory: str) -> None:
        split_data = {}

        try:
  
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
 
                    path = os.path.join(directory, filename)
                    base_name = "_".join(filename.split("_")[:-1]) + ".json"
                    split_type = filename.split("_")[-1].replace(".json", "")

                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    df = pd.DataFrame(data)

                    if base_name not in split_data:
                        split_data[base_name] = {}

                    split_data[base_name][split_type] = df

                    logger.info(f"loaded: {base_name}")

        except Exception as e:
            logger.error(f"Error while loading splits: {e}")

        self.split_data = split_data


    def get_tokenised_dict(self, tokenizer) -> dict:
        tokenised_dict = {}

        for key in self.split_data:
            inputs = self.split_data[key]['train']['input'].astype(str).tolist()
            outputs = self.split_data[key]['train']['output'].astype(str).tolist()
            combined_texts = [inp + tokenizer.eos_token + out for inp, out in zip(inputs, outputs)]
            input_lens = [len(tokenizer(inp + tokenizer.eos_token)["input_ids"]) for inp in inputs]
            encodings = tokenizer(combined_texts, return_tensors="pt", padding=True, truncation=True)

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            labels = input_ids.clone()
            for i, input_len in enumerate(input_lens):
                labels[i, :input_len] = -100

            tokenised_dict[key] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        return tokenised_dict
    

    def merge_tokenised_dict(self, tokenised_dict, tokenizer) -> dict:

        def pad_to_max_length(tensors, pad_token_id=-100):
            max_len = max(t.size(1) for t in tensors)
            padded = []
            for t in tensors:
                pad_len = max_len - t.size(1)
                if pad_len > 0:
                    padding = torch.full((t.size(0), pad_len), pad_token_id, dtype=t.dtype)
                    t = torch.cat([t, padding], dim=1)
                padded.append(t)
            return padded

        input_ids = [v["input_ids"] for v in tokenised_dict.values()]
        attention_masks = [v["attention_mask"] for v in tokenised_dict.values()]
        labels = [v["labels"] for v in tokenised_dict.values()]

        input_ids = pad_to_max_length(input_ids, tokenizer.pad_token_id)
        attention_masks = pad_to_max_length(attention_masks, 0)
        labels = pad_to_max_length(labels, -100)

        return {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_masks, dim=0),
            "labels": torch.cat(labels, dim=0),
        }
    
