"""Load a dataset saved with `DatasetDict.save_to_disk` and export splits to CSV/JSONL.

Usage:
    python src/export_saved_dataset.py --input_dir data/hatexplain --out_dir data/processed

Output:
    data/processed/<split>.csv
    data/processed/<split>.jsonl
"""
import argparse
import os

from datasets import load_from_disk


def export_dataset(input_dir: str = "data/hatexplain", out_dir: str = "data/processed"):
    print(f"Loading dataset from {input_dir}...")
    ds = load_from_disk(input_dir)
    os.makedirs(out_dir, exist_ok=True)
    for split, d in ds.items():
        print(f"Exporting split '{split}' with {len(d)} rows")
        df = d.to_pandas()
        csv_path = os.path.join(out_dir, f"{split}.csv")
        jsonl_path = os.path.join(out_dir, f"{split}.jsonl")
        df.to_csv(csv_path, index=False)
        df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        print(f"  -> {csv_path}\n  -> {jsonl_path}")
    print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/hatexplain")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()
    export_dataset(args.input_dir, args.out_dir)
