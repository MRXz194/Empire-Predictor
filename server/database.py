"""
database.py - SQLite WAL mode database layer
Bảng: rolls, predictions, sessions
"""
import sqlite3
import os
import time
import json

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'empire.db')


def get_db() -> sqlite3.Connection:
    """Get a database connection with WAL mode and row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS rolls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id INTEGER UNIQUE,
            outcome INTEGER NOT NULL,
            color TEXT NOT NULL,
            timestamp INTEGER,
            received_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id INTEGER,
            predicted_color TEXT,
            confidence REAL,
            model_votes TEXT,
            actual_color TEXT,
            correct INTEGER,
            bet_amount REAL,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at INTEGER,
            ended_at INTEGER,
            total_rounds INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            bankroll_start REAL,
            bankroll_end REAL,
            health_score REAL,
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_rolls_round ON rolls(round_id);
        CREATE INDEX IF NOT EXISTS idx_rolls_ts ON rolls(received_at);
        CREATE INDEX IF NOT EXISTS idx_predictions_round ON predictions(round_id);
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialized at {os.path.abspath(DB_PATH)}")


def insert_roll(round_id: int, outcome: int, color: str, timestamp: int = None) -> bool:
    """Insert a single roll. Returns True if inserted, False if duplicate."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO rolls (round_id, outcome, color, timestamp, received_at) VALUES (?, ?, ?, ?, ?)",
            (round_id, outcome, color, timestamp or int(time.time() * 1000), int(time.time() * 1000))
        )
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def insert_prediction(round_id: int, predicted_color: str, confidence: float,
                      model_votes: dict, bet_amount: float = 0):
    """Insert a prediction for a round."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO predictions (round_id, predicted_color, confidence, model_votes, bet_amount, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (round_id, predicted_color, confidence, json.dumps(model_votes), bet_amount, int(time.time() * 1000))
        )
        conn.commit()
    finally:
        conn.close()


def update_prediction_result(round_id: int, actual_color: str):
    """Update prediction with actual result."""
    conn = get_db()
    try:
        conn.execute(
            "UPDATE predictions SET actual_color = ?, correct = (predicted_color = ?) WHERE round_id = ?",
            (actual_color, actual_color, round_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_prediction_by_round(round_id: int) -> dict:
    """Get prediction for a specific round."""
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM predictions WHERE round_id = ?", (round_id,)).fetchone()
        if row:
            d = dict(row)
            d['model_votes'] = json.loads(d['model_votes']) if d['model_votes'] else {}
            return d
        return None
    finally:
        conn.close()


def get_recent_rolls(limit: int = 100) -> list[dict]:
    """Get the most recent N rolls ordered newest first."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT round_id, outcome, color, timestamp, received_at FROM rolls ORDER BY round_id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_rolls_for_training(limit: int = None) -> list[dict]:
    """Get all rolls ordered chronologically (oldest first) for model training."""
    conn = get_db()
    try:
        query = "SELECT round_id, outcome, color FROM rolls ORDER BY round_id ASC"
        if limit:
            query += f" LIMIT {limit}"
        rows = conn.execute(query).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_roll_count() -> int:
    """Get total number of rolls in database."""
    conn = get_db()
    try:
        return conn.execute("SELECT COUNT(*) FROM rolls").fetchone()[0]
    finally:
        conn.close()


def get_prediction_accuracy(last_n: int = 100) -> dict:
    """Get prediction accuracy for the last N predictions."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT COUNT(*) as total, SUM(correct) as wins FROM "
            "(SELECT correct FROM predictions WHERE correct IS NOT NULL ORDER BY id DESC LIMIT ?)",
            (last_n,)
        ).fetchone()
        total = rows[0] or 0
        wins = rows[1] or 0
        return {
            "total": total,
            "wins": wins,
            "accuracy": round(wins / total, 4) if total > 0 else 0
        }
    finally:
        conn.close()


def get_daily_stats() -> list[dict]:
    """Get roll statistics grouped by day."""
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT
                date(received_at / 1000, 'unixepoch') as day,
                COUNT(*) as total,
                SUM(CASE WHEN color = 'T' THEN 1 ELSE 0 END) as t_count,
                SUM(CASE WHEN color = 'CT' THEN 1 ELSE 0 END) as ct_count,
                SUM(CASE WHEN color = 'Bonus' THEN 1 ELSE 0 END) as bonus_count
            FROM rolls
            GROUP BY day
            ORDER BY day DESC
            LIMIT 30
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
