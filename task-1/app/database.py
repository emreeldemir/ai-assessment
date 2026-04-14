"""
SQLite persistence for predictions and feedback.
Schema is intentionally minimal.
"""

import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.environ.get("DB_PATH", "/app/data/mnist.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT    NOT NULL,
                digit         INTEGER NOT NULL,
                confidence    REAL    NOT NULL,
                image_prefix  TEXT,
                correct_label INTEGER          -- NULL until user provides feedback
            )
        """)
        conn.commit()
    print(f"Database ready at {DB_PATH}")


def log_prediction(digit: int, confidence: float, image_prefix: str) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (created_at, digit, confidence, image_prefix)
            VALUES (?, ?, ?, ?)
            """,
            (datetime.now(timezone.utc).isoformat(), digit, confidence, image_prefix),
        )
        conn.commit()
        return cur.lastrowid


def log_feedback(prediction_id: int, correct_label: int):
    with _conn() as conn:
        conn.execute(
            "UPDATE predictions SET correct_label = ? WHERE id = ?",
            (correct_label, prediction_id),
        )
        conn.commit()


def get_recent_predictions(limit: int = 20) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, digit, confidence, correct_label
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
