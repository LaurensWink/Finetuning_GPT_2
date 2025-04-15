import json
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from transformers.tokenization_utils_base import BatchEncoding



class Data():
    """Class to creat and administer datasets"""
    files: dict[str, pd.DataFrame] = {}
    split_data: dict[str, pd.DataFrame] = {}

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.collect_data()
    
    def collect_data(self) -> None:
        """Reads all JSON files, extracts the 'examples' key, stores them as DataFrames in self.files and creates an path-enum"""
        pathlist = Path(self.data_path).glob('**/*.json')
        for path in pathlist:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                
                if 'examples' in json_data:
                    examples = json_data['examples']
                    
                    data = [{
                        "input": example['input'].replace('\n', ' '),
                        "output": example['metadata'].get('answer', None)
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
            logger.warning("No split data found. Did you call .split() first?")
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

    def get_tokenised_train_split(self, tokenizer) -> dict[str, tuple[BatchEncoding, BatchEncoding]]:
        tokenised_dict = {}
        for df_key in self.split_data:
            input_train_encodings = tokenizer(self.split_data[df_key]['train']['input'].astype(str).tolist(), return_tensors="pt", padding=True, truncation=True)
            label_train_encodings = tokenizer(self.split_data[df_key]['train']['output'].astype(str).tolist(), return_tensors="pt", padding=True, truncation=True)

            tokenised_dict[df_key] = (input_train_encodings, label_train_encodings)

        return tokenised_dict