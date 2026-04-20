from pathlib import Path
from datetime import datetime
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

sms_path = RAW_DIR / "uci_sms_real.csv"
telegram_path = RAW_DIR / "telegram_raw.csv"

# -----------------------------
# Load SMS dataset
# -----------------------------
sms = pd.read_csv(sms_path)

sms = sms.rename(columns={
    "Label": "label",
    "EmailText": "raw_text"
})

sms["source_dataset"] = "uci_sms"
sms["source_platform_group"] = "sms"

# -----------------------------
# Load Telegram dataset
# -----------------------------
telegram = pd.read_csv(telegram_path)

telegram = telegram.rename(columns={
    "text_type": "label",
    "text": "raw_text"
})

telegram["source_dataset"] = "telegram_spam_ham"
telegram["source_platform_group"] = "chat_message"

# -----------------------------
# Keep only unified columns
# -----------------------------
common_cols = ["source_dataset", "source_platform_group", "label", "raw_text"]

sms = sms[common_cols].copy()
telegram = telegram[common_cols].copy()

# -----------------------------
# Merge both datasets
# -----------------------------
merged = pd.concat([sms, telegram], ignore_index=True)

# Standardize text and labels
merged["label"] = merged["label"].astype(str).str.strip().str.lower()
merged["raw_text"] = merged["raw_text"].astype(str).str.strip()

# Replace blank strings with NA
merged.loc[merged["raw_text"] == "", "raw_text"] = pd.NA
merged.loc[merged["label"] == "", "label"] = pd.NA

# Add message_id
merged.insert(0, "message_id", [f"msg_{i:06d}" for i in range(1, len(merged) + 1)])

# Add ingest timestamp
merged["ingested_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save merged file
output_path = PROCESSED_DIR / "merged_raw_messages.csv"
merged.to_csv(output_path, index=False)

# Print summary
print("\nMerged file created successfully")
print("Saved to:", output_path)
print("\nShape:", merged.shape)
print("\nColumns:", merged.columns.tolist())

print("\nRows by source:")
print(merged["source_dataset"].value_counts(dropna=False))

print("\nRows by label:")
print(merged["label"].value_counts(dropna=False))

print("\nNull counts:")
print(merged.isnull().sum())