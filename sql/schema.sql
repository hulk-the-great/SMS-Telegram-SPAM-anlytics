DROP TABLE IF EXISTS raw_messages;
DROP TABLE IF EXISTS clean_messages;
DROP TABLE IF EXISTS data_quality_report;

CREATE TABLE raw_messages (
    message_id TEXT PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    source_platform_group TEXT NOT NULL,
    label TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE clean_messages (
    message_id TEXT PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    source_platform_group TEXT NOT NULL,
    label TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    is_null_text INTEGER NOT NULL,
    is_null_label INTEGER NOT NULL,
    is_invalid_label INTEGER NOT NULL,
    is_duplicate_text INTEGER NOT NULL,
    text_length INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    is_too_short INTEGER NOT NULL
);

CREATE TABLE data_quality_report (
    report_timestamp TEXT NOT NULL,
    total_rows INTEGER NOT NULL,
    null_text_rows INTEGER NOT NULL,
    null_label_rows INTEGER NOT NULL,
    invalid_label_rows INTEGER NOT NULL,
    duplicate_text_rows INTEGER NOT NULL,
    too_short_rows INTEGER NOT NULL,
    ham_rows INTEGER NOT NULL,
    spam_rows INTEGER NOT NULL,
    spam_ratio REAL NOT NULL,
    avg_text_length REAL NOT NULL,
    avg_word_count REAL NOT NULL
);