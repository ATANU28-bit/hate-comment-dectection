import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercasing, removing URLs/usernames, extra whitespace."""
    if text is None:
        return ""
    text = text.lower()
    # remove urls
    text = re.sub(r"https?://\S+", "", text)
    # remove usernames (twitter style)
    text = re.sub(r"@\w+", "", text)
    # remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text: str) -> list:
    """Preprocess text: clean, tokenize, and remove stopwords."""
    text = clean_text(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

def detect_text_column(column_names):
    """Heuristic to pick a text column from dataset column names."""
    candidates = [
        "text",
        "comment_text",
        "comment",
        "post",
        "sentence",
        "body",
        "tweet",
    ]
    for c in candidates:
        if c in column_names:
            return c
    # otherwise pick the first string-like column name
    return column_names[0]

if __name__ == "__main__":
    s = "Hello World! Visit https://example.com @user"
    print("Cleaned Text:", clean_text(s))
    print("Tokens:", preprocess_text(s))
