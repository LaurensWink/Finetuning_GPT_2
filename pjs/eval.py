import os
from loguru import logger
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import unicodedata

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

            # Replace NaN -> no answer was generated
            dummy_label = "__MISSING__"
            df["Predicted"] = df["Predicted"].fillna(dummy_label)

            expected = df["Expected"].astype(str).str.strip()
            predicted = df["Predicted"].astype(str).str.strip()

            df["Correct"] = (expected == predicted).map({True: "richtig", False: "falsch"})

            y_true = ["richtig"] * len(df) 
            y_pred = df["Correct"]

            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred, pos_label="richtig")

            output_metrics.append({
                "Subfolder": subfolder.name,
                "File": file.name,
                "Accuracy": accuracy,
                "Recall (macro)": recall,
            })

    results_df = pd.DataFrame(output_metrics)
    output_dir = "data/results/"
    output_file = Path(f"{output_dir}{file_name}.csv")
    os.makedirs(output_dir, exist_ok=True)
    if output_file.exists():
        logger.warning(f"'{output_file}' does already exist, it will be overwritten.")

    results_df.to_csv(output_file, index=False)
    logger.info(f"{output_file} created.")

def eval_mblimp(dir_path, file_name="mblimp_results.csv"):
    dir_path = Path(dir_path) 
    output_metrics = []

    if not dir_path.exists():
        logger.error(f"{dir_path} does not exisit.")
        return

    for file in dir_path.iterdir():
        print(file)
        df = pd.read_csv(file)
        score = df["Score"].astype(int)

        y_true = [1] * len(df) 
        y_pred = score

        accuracy = accuracy_score(y_true, y_pred)

        output_metrics.append({
            "Model": file.name,
            "MBlimp_Accuracy": accuracy,
        })

    results_df = pd.DataFrame(output_metrics)
    output_dir = "data/results/"
    output_file = Path(f"{output_dir}{file_name}.csv")
    os.makedirs(os.path.dirname(output_dir) or ".", exist_ok=True)
    if output_file.exists():
        logger.warning(f"'{output_file}' does already exist, it will be overwritten.")
    results_df.to_csv(output_file, index=False)
    logger.info(f"{output_file} created.")