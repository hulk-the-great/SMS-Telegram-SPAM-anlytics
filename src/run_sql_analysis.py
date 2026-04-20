from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "spam_analytics.db"
SQL_PATH = BASE_DIR / "sql" / "analysis_queries.sql"
OUTPUT_DIR = BASE_DIR / "reports" / "sql_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

query_names = [
    "01_total_messages_by_source",
    "02_spam_vs_ham_by_source",
    "03_overall_spam_rate",
    "04_avg_text_length_by_source_and_label",
    "05_duplicate_messages_by_source",
    "06_short_message_distribution",
]

def load_queries(sql_text: str):
    parts = sql_text.split(";")
    queries = []
    for part in parts:
        lines = []
        for line in part.splitlines():
            stripped = line.strip()
            if not stripped.startswith("--"):
                lines.append(line)
        cleaned = "\n".join(lines).strip()
        if cleaned:
            queries.append(cleaned)
    return queries

with open(SQL_PATH, "r", encoding="utf-8") as f:
    sql_text = f.read()

queries = load_queries(sql_text)

if len(queries) != len(query_names):
    raise ValueError(
        f"Expected {len(query_names)} queries, but found {len(queries)} in analysis_queries.sql"
    )

conn = sqlite3.connect(DB_PATH)

for name, query in zip(query_names, queries):
    df = pd.read_sql_query(query, conn)
    output_path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path.name}")
    print(df.head(10).to_string(index=False))

conn.close()

print(f"\nAll query outputs saved in:\n{OUTPUT_DIR}")