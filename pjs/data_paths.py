from enum import Enum
from pathlib import Path

def create_dataset_enum(directory: str):
    """Dynamically creates an Enum with all JSON file paths in the specified directory."""
    paths = {
        path.stem.upper(): path for path in Path(directory).glob("*.json")
    }
    return Enum("DatasetPaths", paths)