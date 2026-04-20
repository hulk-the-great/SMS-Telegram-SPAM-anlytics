from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURED_PATH = BASE_DIR / "data" / "processed" / "featured_messages.csv"
SCORED_PATH = BASE_DIR / "data" / "processed" / "scored_messages.csv"
MODEL_RESULTS_PATH = BASE_DIR / "reports" / "model_comparison_results.csv"

OUTPUT_DIR = BASE_DIR / "dashboard" / "exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

featured_df = pd.read_csv(FEATURED_PATH)
scored_df = pd.read_csv(SCORED_PATH)
model_results_df = pd.read_csv(MODEL_RESULTS_PATH)

# 1. Main scored detail table for dashboarding
dashboard_scored = scored_df.merge(
    featured_df,
    on=["message_id", "source_dataset", "source_platform_group", "raw_text"],
    how="left"
)

dashboard_scored.to_csv(OUTPUT_DIR / "dashboard_scored_messages.csv", index=False)

# 2. Source summary
source_summary = (
    featured_df.groupby("source_dataset")
    .agg(
        total_messages=("message_id", "count"),
        spam_messages=("label", lambda x: (x == "spam").sum()),
        ham_messages=("label", lambda x: (x == "ham").sum()),
        avg_char_count=("char_count", "mean"),
        avg_word_count=("word_count_fe", "mean"),
        pct_has_url=("has_url", "mean"),
        pct_has_phone=("has_phone", "mean"),
        pct_has_currency=("has_currency", "mean"),
        avg_urgency_word_count=("urgency_word_count", "mean"),
    )
    .reset_index()
)

source_summary["spam_rate_pct"] = (
    100 * source_summary["spam_messages"] / source_summary["total_messages"]
).round(2)

for col in ["pct_has_url", "pct_has_phone", "pct_has_currency"]:
    source_summary[col] = (100 * source_summary[col]).round(2)

source_summary["avg_char_count"] = source_summary["avg_char_count"].round(2)
source_summary["avg_word_count"] = source_summary["avg_word_count"].round(2)
source_summary["avg_urgency_word_count"] = source_summary["avg_urgency_word_count"].round(2)

source_summary.to_csv(OUTPUT_DIR / "source_summary.csv", index=False)

# 3. Label summary
label_summary = (
    featured_df.groupby(["source_dataset", "label"])
    .agg(
        total_messages=("message_id", "count"),
        avg_char_count=("char_count", "mean"),
        avg_word_count=("word_count_fe", "mean"),
        avg_digit_count=("digit_count", "mean"),
        avg_special_char_count=("special_char_count", "mean"),
        pct_has_url=("has_url", "mean"),
        pct_has_phone=("has_phone", "mean"),
        pct_has_currency=("has_currency", "mean"),
        avg_uppercase_ratio=("uppercase_ratio", "mean"),
        avg_urgency_word_count=("urgency_word_count", "mean"),
    )
    .reset_index()
)

for col in ["pct_has_url", "pct_has_phone", "pct_has_currency", "avg_uppercase_ratio"]:
    label_summary[col] = (100 * label_summary[col]).round(2)

for col in [
    "avg_char_count",
    "avg_word_count",
    "avg_digit_count",
    "avg_special_char_count",
    "avg_urgency_word_count",
]:
    label_summary[col] = label_summary[col].round(2)

label_summary.to_csv(OUTPUT_DIR / "label_summary.csv", index=False)

# 4. Error summary
error_summary = (
    scored_df.groupby(["source_dataset", "error_type"])
    .size()
    .reset_index(name="count")
)

error_summary.to_csv(OUTPUT_DIR / "error_summary.csv", index=False)

# 5. Probability bands for dashboard histograms
scored_df["probability_band"] = pd.cut(
    scored_df["spam_probability"],
    bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    include_lowest=True
).astype(str)

probability_summary = (
    scored_df.groupby(["source_dataset", "actual_label", "probability_band"])
    .size()
    .reset_index(name="count")
)

probability_summary.to_csv(OUTPUT_DIR / "probability_summary.csv", index=False)

# 6. Model comparison export
model_results_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)

print("\nDashboard exports created in:")
print(OUTPUT_DIR)

print("\nFiles created:")
for path in sorted(OUTPUT_DIR.glob("*.csv")):
    print("-", path.name)