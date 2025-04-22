from loguru import logger
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(dir_path, file_name):
    dir_path = Path(dir_path) 
    output_metrics = []

    if not dir_path.exists():
        logger.error(f"{dir_path} does not exisit.")
        return

    for subfolder in dir_path.iterdir():
        if not subfolder.is_dir():
            continue

        for file in subfolder.iterdir():
            if file.suffix != ".csv":
                logger.warning(f"{file.name} is no csv file.")
                continue

            df = pd.read_csv(file)

            expected = df["Expected"]
            predicted = df["Predicted"]

            accuracy = accuracy_score(expected, predicted)
            precision = precision_score(expected, predicted, average='macro', zero_division=0)
            recall = recall_score(expected, predicted, average='macro', zero_division=0)
            f1_macro = f1_score(expected, predicted, average='macro', zero_division=0)
            f1_weighted = f1_score(expected, predicted, average='weighted', zero_division=0)

            output_metrics.append({
                "Subfolder": subfolder.name,
                "File": file.name,
                "Accuracy": accuracy,
                "Precision (macro)": precision,
                "Recall (macro)": recall,
                "F1 Score (macro)": f1_macro,
                "F1 Score (weighted)": f1_weighted
            })

    results_df = pd.DataFrame(output_metrics)

    output_file = Path(f"data/results/{file_name}.csv")

    if output_file.exists():
        logger.warning(f"'{output_file}' does already exist, it will be overwritten.")

    results_df.to_csv(output_file, index=False)
    logger.info(f"{output_file} created.")
