import re


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
    print(clean_text(s))
