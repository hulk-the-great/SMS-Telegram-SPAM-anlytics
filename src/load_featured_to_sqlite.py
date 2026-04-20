from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = BASE_DIR / "data" / "spam_analytics.db"
SCHEMA_PATH = BASE_DIR / "sql" / "add_featured_table.sql"
FEATURED_PATH = BASE_DIR / "data" / "processed" / "featured_messages.csv"

# Read featured file
featured_df = pd.read_csv(FEATURED_PATH)

# Connect to database
conn = sqlite3.connect(DB_PATH)

# Create featured_messages table
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema_sql = f.read()

conn.executescript(schema_sql)

# Load data
featured_df.to_sql("featured_messages", conn, if_exists="append", index=False)

# Verify row count
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM featured_messages")
count = cursor.fetchone()[0]

print(f"featured_messages: {count} rows")
print(f"\nLoaded successfully into:\n{DB_PATH}")

conn.close()