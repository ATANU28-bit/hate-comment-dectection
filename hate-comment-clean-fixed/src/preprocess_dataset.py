"""
Advanced preprocessing for hate comment detection dataset.

Features:
- Remove special characters, URLs, mentions
- Handle contractions (e.g., "don't" -> "do not")
- Remove/normalize emojis
- Normalize repeated characters (e.g., "haaate" -> "hate")
- Remove stopwords (optional)
- Handle class imbalance via oversampling/undersampling
- Save processed splits to data/processed/

Usage:
    python src/preprocess_dataset.py --input_csv data/archive/labeled_data.csv \
        --output_dir data/processed --strategy balance --balance_method oversample
"""
import argparse
import os
import re
from collections import Counter

import pandas as pd
import numpy as np
from datasets import Dataset


# Contractions mapping
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Common stopwords to remove (optional)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "will", "with", "i", "me", "my", "we", "you",
}


def expand_contractions(text):
    """Expand contractions like don't -> do not."""
    pattern = re.compile(r'\b(' + '|'.join(CONTRACTIONS.keys()) + r')\b')
    return pattern.sub(lambda x: CONTRACTIONS[x.group(0)], text)


def remove_emoji(text):
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def normalize_repeated_chars(text, threshold=2):
    """Normalize repeated characters (e.g., 'haaaaate' -> 'hate')."""
    return re.sub(r'(.)\1{' + str(threshold - 1) + ',}', r'\1\1', text)


def remove_special_chars(text):
    """Remove special characters but keep basic punctuation."""
    # Keep alphanumerics, spaces, and basic punctuation (.,!?)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text


def remove_stopwords(text, use_stopwords=False):
    """Remove common stopwords (optional)."""
    if not use_stopwords:
        return text
    words = text.split()
    words = [w for w in words if w.lower() not in STOPWORDS]
    return ' '.join(words)


def preprocess_text(text, remove_sw=False):
    """Apply all preprocessing steps."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove @mentions and #hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove emojis
    text = remove_emoji(text)
    
    # Expand contractions
    text = expand_contractions(text)
    
    # Normalize repeated characters
    text = normalize_repeated_chars(text)
    
    # Remove special characters
    text = remove_special_chars(text)
    
    # Remove stopwords (optional)
    text = remove_stopwords(text, use_stopwords=remove_sw)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def balance_classes(df, method='oversample'):
    """
    Balance class distribution.
    
    Args:
        df: DataFrame with 'label' column
        method: 'oversample' (oversample minority), 'undersample' (undersample majority), or 'none'
    
    Returns:
        Balanced DataFrame
    """
    if method == 'none':
        return df
    
    label_counts = df['label'].value_counts()
    print(f"Original class distribution:\n{label_counts}\n")
    
    if method == 'oversample':
        # Oversample minority classes to match majority
        max_count = label_counts.max()
        dfs = []
        for label in label_counts.index:
            label_df = df[df['label'] == label]
            if len(label_df) < max_count:
                # Oversample with replacement
                oversampled = label_df.sample(n=max_count, replace=True, random_state=42)
                dfs.append(oversampled)
            else:
                dfs.append(label_df)
        df = pd.concat(dfs, ignore_index=True)
    
    elif method == 'undersample':
        # Undersample majority classes to match minority
        min_count = label_counts.min()
        dfs = []
        for label in label_counts.index:
            label_df = df[df['label'] == label]
            if len(label_df) > min_count:
                # Undersample
                undersampled = label_df.sample(n=min_count, replace=False, random_state=42)
                dfs.append(undersampled)
            else:
                dfs.append(label_df)
        df = pd.concat(dfs, ignore_index=True)
    
    print(f"Balanced class distribution ({method}):\n{df['label'].value_counts()}\n")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--strategy", type=str, choices=['balance', 'no_balance'], default='balance')
    parser.add_argument("--balance_method", type=str, choices=['oversample', 'undersample', 'none'], default='oversample')
    parser.add_argument("--remove_stopwords", action='store_true', help="Remove stopwords")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train split fraction")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation split fraction")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    df = df[['tweet', 'class']].copy()
    df.columns = ['text', 'label']
    
    print(f"Loaded {len(df)} samples\n")
    
    # Preprocess text
    print("Preprocessing text...")
    df['text'] = df['text'].apply(lambda x: preprocess_text(x, remove_sw=args.remove_stopwords))
    print("Text preprocessing complete\n")
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    print(f"After removing empty texts: {len(df)} samples\n")
    
    # Balance classes if requested
    if args.strategy == 'balance':
        df = balance_classes(df, method=args.balance_method)
    else:
        print(f"Original class distribution:\n{df['label'].value_counts()}\n")
    
    # Split into train/val/test
    print(f"Splitting data (train={args.train_size}, val={args.val_size})...")
    n = len(df)
    train_end = int(n * args.train_size)
    val_end = train_end + int(n * args.val_size)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n")
    
    # Save splits
    os.makedirs(args.output_dir, exist_ok=True)
    
    for split, split_df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        csv_path = os.path.join(args.output_dir, f'{split}.csv')
        jsonl_path = os.path.join(args.output_dir, f'{split}.jsonl')
        
        split_df.to_csv(csv_path, index=False)
        split_df.to_json(jsonl_path, orient='records', lines=True)
        
        print(f"Saved {split}: {csv_path}, {jsonl_path}")
    
    print(f"\n✓ Preprocessing complete. Processed data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
