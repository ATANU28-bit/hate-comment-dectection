from datasets import load_dataset


def load_hatexplain(split="train"):
    """Load HateXplain dataset via Hugging Face datasets.

    Returns a Dataset object for the requested split.
    """
    ds = load_dataset("hatexplain")
    if split not in ds:
        raise ValueError(f"Split {split} not found. Available splits: {list(ds.keys())}")
    return ds[split]


def inspect_dataset(ds):
    """Print simple info about a HF dataset object."""
    print("Columns:", ds.column_names)
    try:
        print("Features:", ds.features)
    except Exception:
        pass
    print("Number of rows:", len(ds))
    # show sample
    print(ds[:3])


if __name__ == "__main__":
    d = load_hatexplain("train")
    inspect_dataset(d)
