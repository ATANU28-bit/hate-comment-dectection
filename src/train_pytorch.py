"""PyTorch-only training script for hate comment detection.

Supports loading from:
1. Preprocessed splits (data/processed/train.csv, etc.) - recommended
2. Raw CSV (data/archive/labeled_data.csv) - applies basic cleaning

Usage:
    # Train on preprocessed data (recommended):
    python src/train_pytorch.py --use_preprocessed
    
    # Or train on raw CSV with basic cleaning:
    python src/train_pytorch.py --csv_path "data/archive/labeled_data.csv"
"""
import os
try:
    import hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass

import argparse

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


def load_preprocessed_splits(data_dir="data/processed"):
    """Load pre-split and preprocessed data from CSV files."""
    print(f"Loading preprocessed splits from {data_dir}...")
    
    splits = {}
    for split in ['train', 'validation', 'test']:
        csv_path = os.path.join(data_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}. Run preprocessing first: python src/preprocess_dataset.py --input_csv data/archive/labeled_data.csv")
        
        df = pd.read_csv(csv_path)
        ds = Dataset.from_pandas(df)
        splits[split] = ds
        print(f"  {split}: {len(ds)} samples")
    
    return DatasetDict(splits)


def load_csv_and_split(csv_path, train_size=0.8, val_size=0.1):
    """Load raw CSV with basic text cleaning, then split into train/val/test."""
    print(f"Loading raw CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Select text and label columns
    df = df[['tweet', 'class']].copy()
    df.columns = ['text', 'label']
    
    # Apply basic text cleaning
    import re
    def basic_clean(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    df['text'] = df['text'].apply(basic_clean)
    
    # Create HF Dataset
    ds = Dataset.from_pandas(df)
    
    # Split into train/val/test
    n = len(ds)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    train_ds = ds.select(range(train_end))
    val_ds = ds.select(range(train_end, val_end))
    test_ds = ds.select(range(val_end, n))
    
    return DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds
    })


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_preprocessed", action='store_true', default=True, help="Use preprocessed data from data/processed/")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to raw CSV file (if not using preprocessed)")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with preprocessed splits")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="models/hate-detection-balanced")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()
    
    # Load data (preprocessed by default, or raw CSV)
    if args.use_preprocessed:
        ds = load_preprocessed_splits(args.data_dir)
    else:
        if args.csv_path is None:
            raise ValueError("Either --use_preprocessed or --csv_path must be provided")
        ds = load_csv_and_split(args.csv_path)
    print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")
    print(f"Label distribution: {ds['train']['label'].count(0)}, {ds['train']['label'].count(1)}, {ds['train']['label'].count(2)}")
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    
    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=args.max_length)
    
    tokenized_ds = ds.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Train
    # Use basic TrainingArguments to maintain compatibility across transformers versions
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=100,
        fp16=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    results = trainer.evaluate(eval_dataset=tokenized_ds['test'])
    print(f"Test results: {results}")


if __name__ == "__main__":
    main()
