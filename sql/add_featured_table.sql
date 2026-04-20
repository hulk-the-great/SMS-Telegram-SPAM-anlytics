DROP TABLE IF EXISTS featured_messages;

CREATE TABLE featured_messages (
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
    is_too_short INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    word_count_fe INTEGER NOT NULL,
    digit_count INTEGER NOT NULL,
    special_char_count INTEGER NOT NULL,
    has_url INTEGER NOT NULL,
    has_phone INTEGER NOT NULL,
    has_currency INTEGER NOT NULL,
    uppercase_count INTEGER NOT NULL,
    alpha_count INTEGER NOT NULL,
    uppercase_ratio REAL NOT NULL,
    urgency_word_count INTEGER NOT NULL
);