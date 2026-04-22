"""
stats.py - Tầng 7: Reporting / Cross-session Stats
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import get_db


def get_overall_stats() -> dict:
    """Overall statistics across all data."""
    conn = get_db()
    try:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN color='T' THEN 1 ELSE 0 END) as t_count,
                SUM(CASE WHEN color='CT' THEN 1 ELSE 0 END) as ct_count,
                SUM(CASE WHEN color='Bonus' THEN 1 ELSE 0 END) as bonus_count,
                MIN(round_id) as first_round,
                MAX(round_id) as last_round
            FROM rolls
        """).fetchone()

        total = row[0] or 1
        return {
            'total_rounds': row[0],
            't_count': row[1],
            'ct_count': row[2],
            'bonus_count': row[3],
            't_pct': round(row[1] / total, 4),
            'ct_pct': round(row[2] / total, 4),
            'bonus_pct': round(row[3] / total, 4),
            'first_round': row[4],
            'last_round': row[5],
        }
    finally:
        conn.close()


def get_prediction_stats(last_n: int = None) -> dict:
    """Prediction accuracy statistics."""
    conn = get_db()
    try:
        query = """
            SELECT
                COUNT(*) as total,
                SUM(correct) as wins,
                AVG(confidence) as avg_confidence
            FROM predictions
            WHERE correct IS NOT NULL
        """
        if last_n:
            query = f"""
                SELECT COUNT(*) as total, SUM(correct) as wins, AVG(confidence) as avg_confidence
                FROM (SELECT correct, confidence FROM predictions WHERE correct IS NOT NULL
                      ORDER BY id DESC LIMIT {last_n})
            """

        row = conn.execute(query).fetchone()
        total = row[0] or 1
        wins = row[1] or 0

        return {
            'total_predictions': row[0],
            'correct': wins,
            'accuracy': round(wins / total, 4),
            'avg_confidence': round(row[2] or 0, 4),
        }
    finally:
        conn.close()


def get_streak_analysis(last_n: int = 500) -> dict:
    """Analyze streaks in recent data."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT color FROM rolls ORDER BY round_id DESC LIMIT ?", (last_n,)
        ).fetchall()

        if not rows:
            return {'max_streak': 0, 'streaks': {}}

        colors = [r[0] for r in reversed(rows)]

        # Find all streaks
        streaks = {'T': [], 'CT': [], 'Bonus': []}
        current_color = colors[0]
        current_len = 1

        for i in range(1, len(colors)):
            if colors[i] == current_color:
                current_len += 1
            else:
                streaks[current_color].append(current_len)
                current_color = colors[i]
                current_len = 1
        streaks[current_color].append(current_len)

        result = {}
        for color in ['T', 'CT', 'Bonus']:
            s = streaks[color]
            if s:
                result[color] = {
                    'max': max(s),
                    'avg': round(sum(s) / len(s), 2),
                    'count': len(s),
                }
            else:
                result[color] = {'max': 0, 'avg': 0, 'count': 0}

        return {
            'window': last_n,
            'streaks': result,
            'max_streak': max(max(s) if s else 0 for s in streaks.values()),
        }
    finally:
        conn.close()
