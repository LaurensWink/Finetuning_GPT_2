import json
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from data_paths import create_dataset_enum



class Dataset():
    """Class to creat and administer datasets"""
    files: dict = {}
    split: dict = {}
    paths: enumerate

    def __init__(self) -> None:
        self.collect_data()
    
    def collect_data(self) -> None:
        """Reads all JSON files, extracts the 'examples' key, stores them as DataFrames in self.files and creates an path-enum"""
        pathlist = Path("data").glob('**/*.json')
        self.paths = create_dataset_enum("data")
        for path in pathlist:
            try:
                # JSON-Datei als Dictionary einlesen
                with open(path, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                
                # Zugriff auf den 'examples'-Key
                if 'examples' in json_data:
                    examples = json_data['examples']
                    
                    # Konvertiere Beispiele in eine Liste von Dictionaries
                    example_rows = []
                    for key, value in examples.items():
                        row = {"id": key, **value}  # ID hinzufügen und Rest entpacken
                        example_rows.append(row)
                    
                    # Liste der Dictionaries in DataFrame umwandeln
                    data_frame = pd.DataFrame(example_rows)
                    
                    self.files[path] = data_frame
                    logger.info(f"Successfully loaded 'examples' from: {path}")
                else:
                    logger.warning(f"'examples' key not found in {path}")
            
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON in {path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error with {path}: {e}")

    def print_data(self, path: Path) -> None:
        """Method to print specific loaded data"""
        if path in self.files:
            print(self.files[path])
        else:
            logger.error("path not found")

    def print_paths(self) -> None:
        """Method to print all paths"""
        for path in self.paths:
            logger.info(f"{path.name}: {path.value}")

    def split_data(self, train_size: float) -> None:
        """Method to split the Data in train and test. The train_size parameter determines the percentage of samples used for traing"""
        for file in self.files:
            logger.debug(file)