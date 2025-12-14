"""Download the HateXplain dataset via Hugging Face `datasets` and save to disk.

Saves the entire DatasetDict to `data/hatexplain/` by default.

Usage:
    python src/fetch_and_save_dataset.py --output_dir data/hatexplain
"""
import argparse
import os

from datasets import load_dataset


def main(output_dir: str = "data/hatexplain"):
    print(f"Loading HateXplain...")
    ds = load_dataset("hatexplain")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving dataset to {output_dir} (this may take a moment)...")
    ds.save_to_disk(output_dir)
    print("Done. Dataset saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/hatexplain")
    args = parser.parse_args()
    main(args.output_dir)
