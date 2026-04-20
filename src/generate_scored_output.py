from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from text_utils import clean_and_tokenize  # needed so joblib can resolve tokenizer

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "processed" / "featured_messages.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model_linear_svm.joblib"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "scored_messages.csv"

df = pd.read_csv(INPUT_PATH)

# Match the same split logic used in model_comparison.py
df = df[["message_id", "source_dataset", "source_platform_group", "label", "raw_text"]].copy()
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X = df["raw_text"]
y = df["label_num"]

X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X,
    y,
    df.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

test_df = df.loc[test_idx].copy().reset_index(drop=True)

model = joblib.load(MODEL_PATH)

test_df["predicted_label_num"] = model.predict(test_df["raw_text"])
test_df["spam_probability"] = model.predict_proba(test_df["raw_text"])[:, 1]

test_df["actual_label"] = test_df["label"]
test_df["predicted_label"] = test_df["predicted_label_num"].map({0: "ham", 1: "spam"})
test_df["is_correct"] = (test_df["actual_label"] == test_df["predicted_label"]).astype(int)


def get_error_type(row):
    actual = row["actual_label"]
    pred = row["predicted_label"]

    if actual == "spam" and pred == "spam":
        return "TP"
    if actual == "ham" and pred == "ham":
        return "TN"
    if actual == "ham" and pred == "spam":
        return "FP"
    return "FN"


test_df["error_type"] = test_df.apply(get_error_type, axis=1)

output_cols = [
    "message_id",
    "source_dataset",
    "source_platform_group",
    "actual_label",
    "predicted_label",
    "spam_probability",
    "is_correct",
    "error_type",
    "raw_text",
]

scored_df = test_df[output_cols].copy()
scored_df.to_csv(OUTPUT_PATH, index=False)

print("\nScored output created")
print("Saved to:", OUTPUT_PATH)
print("\nShape:", scored_df.shape)

print("\nError type counts:")
print(scored_df["error_type"].value_counts().to_string())

print("\nPreview:")
print(scored_df.head(10).to_string(index=False))