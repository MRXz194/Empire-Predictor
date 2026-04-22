"""
features.py - Feature engineering cho tất cả models
Tạo rich feature vector từ raw roll sequence
"""
import numpy as np
from collections import Counter
import math
import time


COLOR_MAP = {'T': 0, 'CT': 1, 'Bonus': 2}
COLOR_NAMES = ['T', 'CT', 'Bonus']

# Deterministic trigram lookup (27 combinations)
TRIGRAM_MAP = {}
_idx = 0
for _a in ['T', 'CT', 'Bonus']:
    for _b in ['T', 'CT', 'Bonus']:
        for _c in ['T', 'CT', 'Bonus']:
            TRIGRAM_MAP[f"{_a}_{_b}_{_c}"] = _idx
            _idx += 1


def outcome_to_color(outcome: int) -> str:
    if outcome == 0:
        return 'Bonus'
    elif 1 <= outcome <= 7:
        return 'T'
    else:
        return 'CT'


def compute_features(rolls: list[dict], index: int, lookback: int = 20, markov_probs: dict = None) -> dict:
    """
    Tính feature vector cho vị trí `index` trong danh sách rolls.
    rolls: list of {round_id, outcome, color}
    """
    if index < lookback:
        return None

    window = rolls[index - lookback:index]
    colors = [r['color'] for r in window]
    outcomes = [r['outcome'] for r in window]

    features = {}

    # ── Lag features (1-10) ──
    for lag in range(1, min(11, len(colors) + 1)):
        features[f'lag_color_{lag}'] = COLOR_MAP.get(colors[-lag], 0)
        features[f'lag_outcome_{lag}'] = outcomes[-lag]

    # ── Streak ──
    streak_len = 1
    streak_color = colors[-1]
    for i in range(len(colors) - 2, -1, -1):
        if colors[i] == streak_color:
            streak_len += 1
        else:
            break
    features['streak_len'] = streak_len
    features['streak_color'] = COLOR_MAP.get(streak_color, 0)

    # ── Rounds since last Bonus ──
    bonus_since = lookback
    for i in range(len(colors) - 1, -1, -1):
        if colors[i] == 'Bonus':
            bonus_since = len(colors) - 1 - i
            break
    features['bonus_since'] = bonus_since

    # ── Item 19: Time-based & Gap features ──
    prev_round = rolls[index-1] if index > 0 else rolls[index]
    curr_round = rolls[index]
    features['round_gap'] = min(curr_round['round_id'] - prev_round['round_id'], 100)
    
    # Cyclical hour feature
    received_at = curr_round.get('received_at') or (time.time() * 1000)
    ts = received_at / 1000
    import datetime
    dt = datetime.datetime.fromtimestamp(ts)
    features['hour_sin'] = math.sin(2 * math.pi * dt.hour / 24)
    features['hour_cos'] = math.cos(2 * math.pi * dt.hour / 24)

    # ── Frequency in windows ──
    for win_size in [10, 20, 50, 100]:
        start = max(0, index - win_size)
        win_colors_full = [r['color'] for r in rolls[start:index]]
        total = len(win_colors_full) if win_colors_full else 1
        c = Counter(win_colors_full)
        features[f'freq_T_{win_size}'] = c.get('T', 0) / total
        features[f'freq_CT_{win_size}'] = c.get('CT', 0) / total
        features[f'freq_Bonus_{win_size}'] = c.get('Bonus', 0) / total

    # ── Shannon Entropy (of last 20) ──
    c = Counter(colors)
    total = len(colors)
    entropy = 0
    for count in c.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    features['entropy_20'] = entropy

    # ── Deviation from expected (T=7/15, CT=7/15, Bonus=1/15) ──
    expected_t = 7 / 15
    expected_ct = 7 / 15
    features['dev_T'] = features.get('freq_T_20', 0.467) - expected_t
    features['dev_CT'] = features.get('freq_CT_20', 0.467) - expected_ct

    # ── Bigram pattern ──
    if len(colors) >= 2:
        bigram = f"{colors[-2]}_{colors[-1]}"
        bigram_map = {'T_T': 0, 'T_CT': 1, 'T_Bonus': 2,
                      'CT_T': 3, 'CT_CT': 4, 'CT_Bonus': 5,
                      'Bonus_T': 6, 'Bonus_CT': 7, 'Bonus_Bonus': 8}
        features['bigram'] = bigram_map.get(bigram, 0)
    else:
        features['bigram'] = 0

    # ── Trigram pattern (deterministic lookup) ──
    if len(colors) >= 3:
        trigram = f"{colors[-3]}_{colors[-2]}_{colors[-1]}"
        features['trigram_hash'] = TRIGRAM_MAP.get(trigram, 26)
    else:
        features['trigram_hash'] = 0

    # ── Item 13: Markov Features ──
    if markov_probs:
        features['markov_T'] = markov_probs.get('T', 0.46)
        features['markov_CT'] = markov_probs.get('CT', 0.46)
        features['markov_Bonus'] = markov_probs.get('Bonus', 0.08)
    else:
        features['markov_T'] = 0.46
        features['markov_CT'] = 0.46
        features['markov_Bonus'] = 0.08

    return features


def compute_features_array(rolls: list[dict], index: int, lookback: int = 20, markov_probs: dict = None) -> np.ndarray:
    """Return features as numpy array with Min-Max scaling for NN stability."""
    f = compute_features(rolls, index, lookback, markov_probs)
    if f is None:
        return None
        
    f_s = dict(f)
    for k in f_s:
        if k.startswith('lag_color_') or k == 'streak_color':
            f_s[k] = f_s[k] / 2.0
        elif k.startswith('lag_outcome_'):
            f_s[k] = f_s[k] / 14.0
        elif k == 'streak_len':
            f_s[k] = min(f_s[k] / 20.0, 1.0)
        elif k == 'bonus_since':
            f_s[k] = min(f_s[k] / 100.0, 1.0)
        elif k == 'round_gap':
            f_s[k] = min(f_s[k] / 100.0, 1.0)
        elif k == 'bigram':
            f_s[k] = f_s[k] / 8.0
        elif k == 'trigram_hash':
            f_s[k] = f_s[k] / 26.0
            
    # Max entropy for 3 classes is ~1.585
    f_s['entropy_20'] = f_s.get('entropy_20', 0) / 1.585
    
    return np.array(list(f_s.values()), dtype=np.float32)


def get_feature_names(lookback: int = 20) -> list[str]:
    """Get ordered list of feature names (for debugging)."""
    names = []
    for lag in range(1, 11):
        names.append(f'lag_color_{lag}')
        names.append(f'lag_outcome_{lag}')
    names.extend(['streak_len', 'streak_color', 'bonus_since'])
    names.extend(['round_gap', 'hour_sin', 'hour_cos'])
    for ws in [10, 20, 50, 100]:
        names.extend([f'freq_T_{ws}', f'freq_CT_{ws}', f'freq_Bonus_{ws}'])
    names.extend(['entropy_20', 'dev_T', 'dev_CT', 'bigram', 'trigram_hash'])
    names.extend(['markov_T', 'markov_CT', 'markov_Bonus'])
    return names


def prepare_sequences(rolls: list[dict], seq_len: int = 60, markov_model=None) -> tuple:
    """
    Prepare data for LSTM/TFT: sliding window sequences.
    markov_model: If provided, dynamically compute probs per step (SOTA).
    """
    X, y = [], []

    # Pre-compute colors for speed
    all_colors = [r['color'] for r in rolls]

    # ─── OPTIMIZATION: Pre-compute Markov Probs for the whole history ───
    markov_cache = {}
    if markov_model:
        print(f"[Features] Pre-computing Markov probabilities for {len(rolls)} rounds...")
        # Since Markov Chain logic is efficient, we can compute them all in one pass
        # or use the model's internal capability if optimized.
        # But even just caching results of predict() helps.
        for i in range(len(all_colors)):
            # Since MarkovChain only looks back 3 rounds, we only need the tail
            if i > 5:
                markov_cache[i] = markov_model.predict(all_colors[max(0, i-10):i])
    
    # Start from a point where we have enough history for lookbacks
    start_idx = max(seq_len + 100, 100)
    for i in range(start_idx, len(rolls)):
        seq = []
        valid = True
        
        # Check overall round continuity for the target window
        # Allow minor gaps (up to 10) but discard major jumps (e.g. new day)
        if rolls[i]['round_id'] - rolls[i-seq_len]['round_id'] > seq_len + 10:
            continue

        for j in range(i - seq_len, i):
            # For each step in seq, compute markov if model provided
            m_probs = markov_cache.get(j)
            
            feat = compute_features_array(rolls, j, lookback=20, markov_probs=m_probs)
            if feat is None:
                valid = False
                break
            seq.append(feat)

        if not valid:
            continue

        target_color = COLOR_MAP.get(rolls[i]['color'], 0)
        target_onehot = [0, 0, 0]
        target_onehot[target_color] = 1

        X.append(seq)
        y.append(target_onehot)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
