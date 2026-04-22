"""
statistical.py - Chi-square test + Frequency oscillation + Autocorrelation
Statistical model that detects distribution deviations
"""
import os
import numpy as np
from collections import Counter
from scipy import stats as sp_stats

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'saved', 'statistical.pkl')


class StatisticalModel:
    def __init__(self):
        self.expected = {'T': 7/15, 'CT': 7/15, 'Bonus': 1/15}
        self.trained = False

    def train(self, colors: list[str]):
        """Compute baseline statistics from historical data."""
        c = Counter(colors)
        total = len(colors)
        self.actual_dist = {k: c.get(k, 0) / total for k in ['T', 'CT', 'Bonus']}

        # Compute autocorrelation of color sequence
        color_nums = [0 if c == 'T' else 1 if c == 'CT' else 2 for c in colors]
        self.acf_values = self._autocorrelation(color_nums, max_lag=10)

        self.trained = True
        print(f"[Statistical] Trained. Actual dist: T={self.actual_dist['T']:.4f} "
              f"CT={self.actual_dist['CT']:.4f} Bonus={self.actual_dist['Bonus']:.4f}")

    def predict(self, recent_colors: list[str], window: int = 50) -> dict:
        """
        Predict based on statistical patterns.
        PRIORITY 1: Autocorrelation (Streak/Ping-pong detection)
        PRIORITY 2: Frequency Deviation (Mean reversion)
        """
        if len(recent_colors) < 10:
            return {'T': 7/15, 'CT': 7/15, 'Bonus': 1/15}

        # Initialize with baseline expected probabilities
        probs = self.expected.copy()
        sample = recent_colors[-window:]
        total = len(sample)
        last_color = recent_colors[-1]

        # ─── PRIORITY 1: Autocorrelation (Lag-1) ───
        if len(sample) >= 20:
            color_nums = [0 if c == 'T' else 1 if c == 'CT' else 2 for c in sample]
            acf = self._autocorrelation(color_nums, max_lag=1)
            
            if len(acf) > 0:
                acf1 = acf[0]
                if acf1 < -0.15:
                    # Strong negative ACF: HIGH PROBABILITY of reversal (Ping-pong)
                    opposite = 'CT' if last_color == 'T' else 'T'
                    probs[opposite] += 0.25
                    probs[last_color] -= 0.15
                    # print(f"[Stat] Ping-pong detected (ACF={acf1:.2f})")
                elif acf1 > 0.15:
                    # Positive autocorrelation: Trend/Streak protection
                    probs[last_color] += 0.20
                    # print(f"[Stat] Streak detected (ACF={acf1:.2f})")

        # ─── PRIORITY 2: Frequency Deviation (Chi-square) ───
        c = Counter(sample)
        obs_arr = np.array([c.get('T', 0), c.get('CT', 0), c.get('Bonus', 0)])
        exp_arr = np.array([self.expected['T'], self.expected['CT'], self.expected['Bonus']]) * total
        exp_arr = np.maximum(exp_arr, 1)
        
        chi2, p_value = sp_stats.chisquare(obs_arr, exp_arr)
        
        # Only apply frequency adjustment if deviation is significant (p < 0.1)
        # and keep it subtle to avoid Gambler's Fallacy dominance.
        if p_value < 0.1:
            adjustment_strength = 0.08
            for color in ['T', 'CT', 'Bonus']:
                observed_freq = c.get(color, 0) / total
                deviation = self.expected[color] - observed_freq
                probs[color] += deviation * adjustment_strength

        # Normalize probabilities to sum to 1.0
        total_p = sum(probs.values())
        return {k: v / total_p for k, v in probs.items()}

    def get_chi_square(self, recent_colors: list[str], window: int = 50) -> dict:
        """Return chi-square test results for the recent window."""
        sample = recent_colors[-window:]
        c = Counter(sample)
        total = len(sample)

        obs = np.array([c.get('T', 0), c.get('CT', 0), c.get('Bonus', 0)])
        exp = np.array([self.expected['T'], self.expected['CT'], self.expected['Bonus']]) * total
        exp = np.maximum(exp, 1)

        chi2, p_value = sp_stats.chisquare(obs, exp)
        return {
            'chi2': float(chi2),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'observed': {k: c.get(k, 0) / total for k in ['T', 'CT', 'Bonus']},
            'expected': dict(self.expected)
        }

    def _autocorrelation(self, series: list, max_lag: int = 10) -> list[float]:
        """Compute autocorrelation for lags 1..max_lag."""
        if len(series) < max_lag + 2:
            return []
        arr = np.array(series, dtype=float)
        mean = arr.mean()
        var = np.var(arr)
        if var == 0:
            return [0.0] * max_lag

        acf = []
        for lag in range(1, max_lag + 1):
            c = np.mean((arr[:-lag] - mean) * (arr[lag:] - mean)) / var
            acf.append(float(c))
        return acf

    def save(self, path: str = None):
        """Save trained model state to disk."""
        import pickle
        path = path or SAVE_PATH
        if not self.trained:
            return
        data = {
            'actual_dist': self.actual_dist,
            'acf_values': self.acf_values,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Statistical] Saved to {path}")

    def load(self, path: str = None) -> bool:
        """Load trained model state from disk."""
        import pickle
        path = path or SAVE_PATH
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.actual_dist = data['actual_dist']
        self.acf_values = data['acf_values']
        self.trained = True
        print(f"[Statistical] Loaded from {path}")
        return True
