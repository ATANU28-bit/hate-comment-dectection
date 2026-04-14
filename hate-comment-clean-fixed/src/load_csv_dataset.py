"""Load the labeled_data.csv dataset from archive and convert to Hugging Face format."""
import os
import pandas as pd
from datasets import Dataset


def load_csv_dataset(csv_path: str = "data/archive/labeled_data.csv"):
    """Load CSV and convert to Hugging Face Dataset."""
    df = pd.read_csv(csv_path)
    # Select only text and label columns
    df = df[['tweet', 'class']].copy()
    df.columns = ['text', 'label']
    # Create HF Dataset
    ds = Dataset.from_pandas(df)
    return ds


if __name__ == "__main__":
    ds = load_csv_dataset()
    print("Loaded dataset:", ds)
    print("Columns:", ds.column_names)
    print("Number of samples:", len(ds))
    print("First example:", ds[0])
