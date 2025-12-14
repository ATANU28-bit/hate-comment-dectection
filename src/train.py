import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from preprocess import clean_text, detect_text_column


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


def prepare_dataset(dataset_name="hatexplain", text_column=None, label_column=None, split="train", csv_path=None):
    # If CSV path provided, load from CSV
    if csv_path and os.path.exists(csv_path):
        print(f"Loading from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        # Auto-detect text and label columns
        if text_column is None:
            text_column = detect_text_column(df.columns.tolist())
        if label_column is None:
            # Try to find label column
            candidates = ['class', 'label', 'target']
            for c in candidates:
                if c in df.columns:
                    label_column = c
                    break
            if label_column is None:
                label_column = df.columns[-1]
        
        df = df[[text_column, label_column]].copy()
        df.columns = ['text', 'label']
        dataset = Dataset.from_pandas(df)
    else:
        # Load from Hugging Face datasets
        ds = load_dataset(dataset_name)
        # pick split
        if split not in ds:
            raise ValueError(f"Split {split} not found in dataset. Available: {list(ds.keys())}")
        dataset = ds[split]
        cols = dataset.column_names
        if text_column is None:
            text_column = detect_text_column(cols)
        if label_column is None:
            # heuristics for label column
            if "label" in cols:
                label_column = "label"
            elif "labels" in cols:
                label_column = "labels"
            else:
                # pick last column as label (fallback)
                label_column = cols[-1]
    
    # clean text
    def _clean(example):
        example['text'] = clean_text(example['text'])
        return example

    dataset = dataset.map(_clean)
    return dataset, 'text', 'label'


def tokenize_and_encode(dataset, tokenizer, text_column, label_column, max_length=128):
    def tokenize_fn(examples):
        texts = examples[text_column]
        tokenized = tokenizer(texts, truncation=True, max_length=max_length)
        # ensure label present
        if label_column in examples:
            tokenized["labels"] = examples[label_column]
        return tokenized

    return dataset.map(tokenize_fn, batched=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="hatexplain")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV file instead of HF dataset")
    parser.add_argument("--output_dir", type=str, default="models/distilbert-hatexplain")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    # prepare dataset
    print("Loading dataset...")
    dataset, text_col, label_col = prepare_dataset(args.dataset_name, split=args.split, csv_path=args.csv_path)
    print("Columns:", dataset.column_names)

    # some datasets provide multi-class labels; infer num_labels
    # try to find the number of label classes from the dataset features
    num_labels = 2
    try:
        # attempt to inspect features
        features = dataset.features
        if label_col in features and hasattr(features[label_col], "num_classes"):
            num_labels = features[label_col].num_classes
        else:
            # fallback: infer from unique labels in a subset
            labels_unique = list(set(dataset["label"])) if "label" in dataset.column_names else None
            if labels_unique:
                num_labels = len(labels_unique)
    except Exception:
        pass

    print(f"Using text column: {text_col} and label column: {label_col} (num_labels={num_labels})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # tokenization
    tokenized = tokenize_and_encode(dataset, tokenizer, text_col, label_col, max_length=args.max_length)

    # split into train/validation if needed
    if "train" in dataset.split and "validation" not in dataset.split:
        # attempt quick split
        try:
            tokenized = tokenized.train_test_split(test_size=0.1)
            train_dataset = tokenized["train"]
            eval_dataset = tokenized["test"]
        except Exception:
            train_dataset = tokenized
            eval_dataset = None
    else:
        train_dataset = tokenized
        eval_dataset = None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        logging_steps=50,
        load_best_model_at_end=True if eval_dataset is not None else False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
