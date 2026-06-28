-- ABC Technologies Customer Support AI System
-- Task 7: SQLite Memory Schema
-- This file documents the database structure used for conversation memory.

CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   TEXT    NOT NULL,
    customer_name TEXT,
    role          TEXT    NOT NULL,
    content       TEXT    NOT NULL,
    intent        TEXT,
    timestamp     TEXT    NOT NULL
);

-- Index for fast customer lookup
CREATE INDEX IF NOT EXISTS idx_customer_id ON conversations(customer_id);
CREATE INDEX IF NOT EXISTS idx_timestamp   ON conversations(timestamp);