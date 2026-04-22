"""
monte_carlo.py - Tầng 5: Monte Carlo Simulation
10,000 scenarios, bankroll projection, ruin probability
Uses Markov-chain predictions instead of random betting
"""
import random
import numpy as np
from collections import Counter, defaultdict


def _build_transition_table(colors: list[str]) -> dict:
    """Build order-1 Markov transition probabilities."""
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(1, len(colors)):
        transitions[colors[i-1]][colors[i]] += 1
    
    probs = {}
    for state, counts in transitions.items():
        total = sum(counts.values())
        probs[state] = {c: counts.get(c, 0) / total for c in ['T', 'CT', 'Bonus']}
    return probs


def run_monte_carlo(historical_colors: list[str], n_simulations: int = 10000,
                    n_rounds: int = 100, bankroll: float = 100.0,
                    bet_fraction: float = 0.05, confidence_threshold: float = 0.55,
                    ensemble_predict_fn=None) -> dict:
    """
    Monte Carlo simulation using ensemble-based predictions (if provided)
    or Markov-based predictions.
    """
    # Compute distribution from historical data
    c = Counter(historical_colors)
    total = len(historical_colors)
    p_t = c.get('T', 0) / total
    p_ct = c.get('CT', 0) / total
    p_bonus = c.get('Bonus', 0) / total

    colors = ['T', 'CT', 'Bonus']
    weights = [p_t, p_ct, p_bonus]

    # Build Markov transition table (fallback)
    markov_probs = _build_transition_table(historical_colors)

    final_bankrolls = []
    ruin_count = 0
    peak_bankrolls = []
    min_bankrolls = []

    for _ in range(n_simulations):
        br = bankroll
        peak = bankroll
        trough = bankroll
        
        # Simulated sequence for state tracking
        sim_history = historical_colors[-100:]

        for _ in range(n_rounds):
            if br <= 0:
                break

            last_color = sim_history[-1]
            
            # ───── Strategy Logic ─────
            if ensemble_predict_fn:
                # Use actual ensemble logic (Item 10 fix from note 2)
                pred = ensemble_predict_fn(sim_history)
                best_color = pred['color']
                confidence = pred['confidence']
                if confidence >= confidence_threshold:
                    bet_color = best_color
                else:
                    bet_color = None
            else:
                # Fallback to Markov
                probs = markov_probs.get(last_color, {'T': 0.467, 'CT': 0.467, 'Bonus': 0.067})
                t_prob = probs.get('T', 0.467)
                ct_prob = probs.get('CT', 0.467)
                if t_prob >= ct_prob and t_prob >= confidence_threshold * 0.9:
                    bet_color = 'T'
                elif ct_prob > t_prob and ct_prob >= confidence_threshold * 0.9:
                    bet_color = 'CT'
                else:
                    bet_color = None

            # ───── Execution ─────
            actual = random.choices(colors, weights=weights)[0]
            sim_history.append(actual)
            
            if bet_color:
                bet_amount = min(br * bet_fraction, br)
                # Note: Roulette pays 2x on T/CT (Net profit = bet_amount)
                if bet_color == actual:
                    br += bet_amount
                else:
                    br -= bet_amount

            peak = max(peak, br)
            trough = min(trough, br)

        final_bankrolls.append(br)
        peak_bankrolls.append(peak)
        min_bankrolls.append(trough)
        if br <= 0:
            ruin_count += 1

    arr = np.array(final_bankrolls)

    return {
        'n_simulations': n_simulations,
        'n_rounds': n_rounds,
        'initial_bankroll': bankroll,
        'bet_fraction': bet_fraction,
        'distribution': {
            'T': round(p_t, 4),
            'CT': round(p_ct, 4),
            'Bonus': round(p_bonus, 4)
        },
        'results': {
            'mean': round(float(arr.mean()), 2),
            'median': round(float(np.median(arr)), 2),
            'std': round(float(arr.std()), 2),
            'p5': round(float(np.percentile(arr, 5)), 2),
            'p10': round(float(np.percentile(arr, 10)), 2),
            'p25': round(float(np.percentile(arr, 25)), 2),
            'p50': round(float(np.percentile(arr, 50)), 2),
            'p75': round(float(np.percentile(arr, 75)), 2),
            'p90': round(float(np.percentile(arr, 90)), 2),
            'p95': round(float(np.percentile(arr, 95)), 2),
        },
        'ruin_probability': round(ruin_count / n_simulations, 4),
        'profit_probability': round(float((arr > bankroll).mean()), 4),
        'avg_peak': round(float(np.mean(peak_bankrolls)), 2),
        'avg_trough': round(float(np.mean(min_bankrolls)), 2),
        # Histogram data for chart (20 bins)
        'histogram': _histogram(arr, 20),
    }


def _histogram(arr: np.ndarray, n_bins: int = 20) -> dict:
    counts, edges = np.histogram(arr, bins=n_bins)
    return {
        'counts': counts.tolist(),
        'edges': [round(float(e), 2) for e in edges.tolist()],
    }
