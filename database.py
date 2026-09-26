import sqlite3
from datetime import datetime
from typing import List, Dict, Any

DB_NAME = "companion_vault.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # User journal and mood logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS journal_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'local_user',
                entry_text TEXT,
                primary_emotion TEXT,
                sentiment_score REAL,
                stress_level INTEGER
            )
        """)
        
        # Conversation history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'local_user',
                role TEXT,
                message TEXT,
                detected_emotion TEXT
            )
        """)
        conn.commit()

def log_journal_entry(entry_text: str, emotion: str, score: float, stress: int, user_id: str = "local_user"):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO journal_logs (user_id, entry_text, primary_emotion, sentiment_score, stress_level)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, entry_text, emotion, score, stress))
        conn.commit()

def log_message(role: str, message: str, emotion: str = "Neutral", user_id: str = "local_user"):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversation_history (user_id, role, message, detected_emotion)
            VALUES (?, ?, ?, ?)
        """, (user_id, role, message, emotion))
        conn.commit()

def fetch_recent_messages(limit: int = 6, user_id: str = "local_user") -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, message FROM conversation_history
            WHERE user_id = ?
            ORDER BY id DESC LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

def fetch_mood_analytics(user_id: str = "local_user") -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, primary_emotion, sentiment_score, stress_level, entry_text
            FROM journal_logs
            WHERE user_id = ?
            ORDER BY timestamp ASC
        """, (user_id,))
        return [dict(r) for r in cursor.fetchall()]