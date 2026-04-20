from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "clean_messages.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "featured_messages.csv"

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
CURRENCY_PATTERN = re.compile(r"[$£€₹]")
URGENT_WORDS = {
    "free", "win", "winner", "claim", "urgent", "offer", "limited",
    "prize", "cash", "bonus", "click", "call", "guaranteed", "selected"
}

def count_digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text)

def count_uppercase(text: str) -> int:
    return sum(ch.isupper() for ch in text)

def count_alpha(text: str) -> int:
    return sum(ch.isalpha() for ch in text)

def count_special_chars(text: str) -> int:
    return sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)

def count_urgent_words(text: str) -> int:
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return sum(1 for w in words if w in URGENT_WORDS)

df = pd.read_csv(INPUT_PATH)

df["raw_text"] = df["raw_text"].astype(str)

df["char_count"] = df["raw_text"].str.len()
df["word_count_fe"] = df["raw_text"].str.split().str.len()
df["digit_count"] = df["raw_text"].apply(count_digits)
df["special_char_count"] = df["raw_text"].apply(count_special_chars)

df["has_url"] = df["raw_text"].apply(lambda x: int(bool(URL_PATTERN.search(x))))
df["has_phone"] = df["raw_text"].apply(lambda x: int(bool(PHONE_PATTERN.search(x))))
df["has_currency"] = df["raw_text"].apply(lambda x: int(bool(CURRENCY_PATTERN.search(x))))

df["uppercase_count"] = df["raw_text"].apply(count_uppercase)
df["alpha_count"] = df["raw_text"].apply(count_alpha)
df["uppercase_ratio"] = df.apply(
    lambda row: round(row["uppercase_count"] / row["alpha_count"], 4) if row["alpha_count"] > 0 else 0.0,
    axis=1
)

df["urgency_word_count"] = df["raw_text"].apply(count_urgent_words)

df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature engineering complete")
print("Saved to:", OUTPUT_PATH)
print("\nShape:", df.shape)

preview_cols = [
    "message_id", "source_dataset", "label", "char_count", "word_count_fe",
    "digit_count", "special_char_count", "has_url", "has_phone",
    "has_currency", "uppercase_ratio", "urgency_word_count"
]
print("\nPreview:")
print(df[preview_cols].head(10).to_string(index=False))

print("\nFeature summaries:")
summary_cols = [
    "char_count", "word_count_fe", "digit_count", "special_char_count",
    "has_url", "has_phone", "has_currency", "uppercase_ratio", "urgency_word_count"
]
print(df[summary_cols].describe().to_string())