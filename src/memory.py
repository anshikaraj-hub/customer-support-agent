# Task 7: SQLite Memory
# Stores and retrieves customer conversation history using a local SQLite database.

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional

# Database will be created in the root project folder
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory.db")


def init_db():
    """Create the conversations table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id   TEXT NOT NULL,
            customer_name TEXT,
            role          TEXT NOT NULL,
            content       TEXT NOT NULL,
            intent        TEXT,
            timestamp     TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_message(
    customer_id: str,
    role: str,
    content: str,
    customer_name: Optional[str] = None,
    intent: Optional[str] = None
):
    """Save a single message (user or assistant) to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (customer_id, customer_name, role, content, intent, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (customer_id, customer_name, role, content, intent, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_conversation_history(customer_id: str, limit: int = 20) -> List[Dict]:
    """Retrieve the last N messages for a given customer."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, intent, timestamp
        FROM conversations
        WHERE customer_id = ?
        ORDER BY timestamp ASC
        LIMIT ?
    """, (customer_id, limit))
    rows = cursor.fetchall()
    conn.close()

    history = [
        {"role": row[0], "content": row[1], "intent": row[2], "timestamp": row[3]}
        for row in rows
    ]
    return history


def get_customer_name(customer_id: str) -> Optional[str]:
    """Look up the stored name for a customer."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT customer_name FROM conversations
        WHERE customer_id = ? AND customer_name IS NOT NULL
        ORDER BY timestamp DESC LIMIT 1
    """, (customer_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def format_history_for_prompt(history: List[Dict]) -> str:
    """Convert history list into a readable string for the LLM prompt."""
    if not history:
        return "No previous conversation history found."

    lines = []
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Support Agent"
        intent_tag = f" [Intent: {msg['intent']}]" if msg.get("intent") else ""
        lines.append(f"{role}{intent_tag}: {msg['content']}")
    return "\n".join(lines)


# Initialise the database as soon as this file is imported
init_db()