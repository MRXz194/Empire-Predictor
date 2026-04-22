import re
import csv
import os
from datetime import datetime

def clean_data(input_file, output_dir):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Patterns
    # Header: 2026-03-17 00:00:19 - 2026-03-17 23:59:44
    header_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2} - (\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2}')
    # Round: #12050787 - 5
    round_pattern = re.compile(r'#(\d+)\s*-\s*(\d+)')

    current_date = None
    current_session_rounds = []
    day_sessions = {} # {date: [session1_rounds, session2_rounds, ...]}

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            header_match = header_pattern.search(line)
            if header_match:
                # If we have rounds from a previous session, save them
                if current_session_rounds and current_date:
                    if current_date not in day_sessions:
                        day_sessions[current_date] = []
                    day_sessions[current_date].append(current_session_rounds)
                
                # Start new session
                current_date = header_match.group(1) # Use the start date of the range
                current_session_rounds = []
                continue

            round_match = round_pattern.search(line)
            if round_match:
                session_id = int(round_match.group(1))
                outcome_int = int(round_match.group(2))
                
                if outcome_int == 0:
                    color = 'Bonus'
                elif 1 <= outcome_int <= 7:
                    color = 'T'
                elif 8 <= outcome_int <= 14:
                    color = 'CT'
                else:
                    color = 'Unknown'

                current_session_rounds.append({
                    'session_id': session_id,
                    'outcome': outcome_int,
                    'color': color
                })

    # Add the last session
    if current_session_rounds and current_date:
        if current_date not in day_sessions:
            day_sessions[current_date] = []
        day_sessions[current_date].append(current_session_rounds)

    total_files = 0
    total_rounds = 0

    for date, sessions in day_sessions.items():
        # Consolidate all sessions for the same day
        all_rounds = []
        for s in sessions:
            all_rounds.extend(s)
        
        # Sort chronologically
        all_rounds.sort(key=lambda x: x['session_id'])
        
        output_file = os.path.join(output_dir, f'{date}.csv')
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['session_id', 'outcome', 'color']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rounds)
        
        total_files += 1
        total_rounds += len(all_rounds)

    print(f"Successfully processed {total_rounds} rounds into {total_files} daily files.")
    print(f"Data saved to directory: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'data.txt')
    output_path = os.path.join(current_dir, 'cleaned_data')
    clean_data(input_path, output_path)
