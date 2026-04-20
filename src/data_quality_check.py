from pathlib import Path
from datetime import datetime
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "merged_raw_messages.csv"
OUTPUT_REPORT_PATH = BASE_DIR / "reports" / "data_quality_report.csv"
OUTPUT_CLEAN_PATH = BASE_DIR / "data" / "processed" / "clean_messages.csv"

df = pd.read_csv(INPUT_PATH)

# -----------------------------
# Basic validation flags
# -----------------------------
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["raw_text"] = df["raw_text"].astype(str).str.strip()

df["is_null_text"] = df["raw_text"].isna() | (df["raw_text"] == "")
df["is_null_label"] = df["label"].isna() | (df["label"] == "")
df["is_invalid_label"] = ~df["label"].isin(["ham", "spam"])
df["is_duplicate_text"] = df.duplicated(subset=["raw_text"], keep="first")
df["text_length"] = df["raw_text"].fillna("").str.len()
df["word_count"] = df["raw_text"].fillna("").str.split().str.len()
df["is_too_short"] = df["word_count"] < 2

# -----------------------------
# Report summary
# -----------------------------
report = {
    "report_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_rows": len(df),
    "null_text_rows": int(df["is_null_text"].sum()),
    "null_label_rows": int(df["is_null_label"].sum()),
    "invalid_label_rows": int(df["is_invalid_label"].sum()),
    "duplicate_text_rows": int(df["is_duplicate_text"].sum()),
    "too_short_rows": int(df["is_too_short"].sum()),
    "ham_rows": int((df["label"] == "ham").sum()),
    "spam_rows": int((df["label"] == "spam").sum()),
    "spam_ratio": round((df["label"] == "spam").mean(), 4),
    "avg_text_length": round(df["text_length"].mean(), 2),
    "avg_word_count": round(df["word_count"].mean(), 2),
}

report_df = pd.DataFrame([report])

# -----------------------------
# Source-wise summary
# -----------------------------
source_summary = (
    df.groupby("source_dataset")
      .agg(
          total_rows=("message_id", "count"),
          ham_rows=("label", lambda x: (x == "ham").sum()),
          spam_rows=("label", lambda x: (x == "spam").sum()),
          duplicate_text_rows=("is_duplicate_text", "sum"),
          avg_text_length=("text_length", "mean"),
          avg_word_count=("word_count", "mean"),
      )
      .reset_index()
)

# -----------------------------
# Clean dataset for next stages
# -----------------------------
clean_df = df[
    (~df["is_null_text"]) &
    (~df["is_null_label"]) &
    (~df["is_invalid_label"])
].copy()

# Keep duplicates for now? No — remove exact duplicates
clean_df = clean_df[~clean_df["is_duplicate_text"]].copy()

# Save outputs
report_df.to_csv(OUTPUT_REPORT_PATH, index=False)
clean_df.to_csv(OUTPUT_CLEAN_PATH, index=False)

print("\nData Quality Report")
print(report_df.to_string(index=False))

print("\nSource-wise Summary")
print(source_summary.to_string(index=False))

print("\nClean dataset saved to:")
print(OUTPUT_CLEAN_PATH)

print("\nFinal clean shape:", clean_df.shape)