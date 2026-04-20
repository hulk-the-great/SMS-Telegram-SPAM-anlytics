from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = BASE_DIR / "data" / "spam_analytics.db"
SCHEMA_PATH = BASE_DIR / "sql" / "schema.sql"

RAW_MESSAGES_PATH = BASE_DIR / "data" / "processed" / "merged_raw_messages.csv"
CLEAN_MESSAGES_PATH = BASE_DIR / "data" / "processed" / "clean_messages.csv"
DQ_REPORT_PATH = BASE_DIR / "reports" / "data_quality_report.csv"

# Read source files
raw_df = pd.read_csv(RAW_MESSAGES_PATH)
clean_df = pd.read_csv(CLEAN_MESSAGES_PATH)
dq_df = pd.read_csv(DQ_REPORT_PATH)

# Create database and apply schema
conn = sqlite3.connect(DB_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema_sql = f.read()

conn.executescript(schema_sql)

# Load tables
raw_df.to_sql("raw_messages", conn, if_exists="append", index=False)
clean_df.to_sql("clean_messages", conn, if_exists="append", index=False)
dq_df.to_sql("data_quality_report", conn, if_exists="append", index=False)

# Quick verification
cursor = conn.cursor()

for table in ["raw_messages", "clean_messages", "data_quality_report"]:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"{table}: {count} rows")

conn.close()

print(f"\nSQLite database created successfully at:\n{DB_PATH}")