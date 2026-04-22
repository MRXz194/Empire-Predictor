"""
bootstrap.py - Import 35 CSV files vào SQLite database
Chạy 1 lần để seed historical data (~125,000 rounds)
"""
import os
import csv
import time
from database import init_db, get_db, get_roll_count


def bootstrap_csv(csv_dir: str):
    """Import all CSV files from cleaned_data/ into rolls table."""
    init_db()

    existing = get_roll_count()
    if existing > 0:
        print(f"[Bootstrap] Database already has {existing} rolls. Skipping.")
        print("[Bootstrap] To re-import, delete empire.db first.")
        return

    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    if not csv_files:
        print(f"[Bootstrap] No CSV files found in {csv_dir}")
        return

    print(f"[Bootstrap] Found {len(csv_files)} CSV files")
    conn = get_db()
    total = 0

    for csv_file in csv_files:
        filepath = os.path.join(csv_dir, csv_file)
        count = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                # Tạo fake timestamp dựa trên round_id (ước tính ~24s mỗi round)
                round_id = int(row['session_id'])
                outcome = int(row['outcome'])
                color = row['color']
                rows.append((round_id, outcome, color, None, None))
                count += 1

            conn.executemany(
                "INSERT OR IGNORE INTO rolls (round_id, outcome, color, timestamp, received_at) "
                "VALUES (?, ?, ?, ?, ?)",
                rows
            )

        total += count
        print(f"  [{csv_file}] {count} rounds")

    conn.commit()
    conn.close()

    final_count = get_roll_count()
    print(f"\n[Bootstrap] Done! Imported {final_count} rounds total.")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, '..', 'cleaned_data')
    bootstrap_csv(csv_dir)
