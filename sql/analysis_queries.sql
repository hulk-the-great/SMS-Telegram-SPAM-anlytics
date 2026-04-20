-- 1. Total messages by source
SELECT
    source_dataset,
    COUNT(*) AS total_messages
FROM clean_messages
GROUP BY source_dataset;

-- 2. Spam vs ham by source
SELECT
    source_dataset,
    label,
    COUNT(*) AS message_count
FROM clean_messages
GROUP BY source_dataset, label
ORDER BY source_dataset, label;

-- 3. Overall spam rate
SELECT
    ROUND(
        100.0 * SUM(CASE WHEN label = 'spam' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS spam_rate_percent
FROM clean_messages;

-- 4. Average text length by source and label
SELECT
    source_dataset,
    label,
    ROUND(AVG(text_length), 2) AS avg_text_length,
    ROUND(AVG(word_count), 2) AS avg_word_count
FROM clean_messages
GROUP BY source_dataset, label
ORDER BY source_dataset, label;

-- 5. Duplicate messages by source
SELECT
    source_dataset,
    SUM(duplicate_count - 1) AS duplicate_rows
FROM (
    SELECT
        source_dataset,
        raw_text,
        COUNT(*) AS duplicate_count
    FROM raw_messages
    GROUP BY source_dataset, raw_text
    HAVING COUNT(*) > 1
) t
GROUP BY source_dataset;

-- 6. Short-message distribution
SELECT
    source_dataset,
    label,
    SUM(CASE WHEN is_too_short = 1 THEN 1 ELSE 0 END) AS too_short_count,
    COUNT(*) AS total_count
FROM clean_messages
GROUP BY source_dataset, label
ORDER BY source_dataset, label;