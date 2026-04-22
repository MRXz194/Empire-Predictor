"""
online.py - Tầng 6: Continuous Learning
Online learning (river lib) + ADWIN drift detection + auto-retrain trigger
"""
import os
import sys
import pickle
import time

SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'online_model.pkl')


class OnlineLearner:
    """
    Online learning using river library's HoeffdingTreeClassifier.
    Updates after every single round — no batch retraining needed.
    """

    def __init__(self):
        self.model = None
        self.drift_detector = None
        self.n_updates = 0
        self.correct_count = 0
        self.total_count = 0
        self.drift_alerts = []
        self._drift_detected_now = False
        self._initialized = False

    def _init_models(self):
        """Lazy init to avoid import issues at module load time."""
        if self._initialized:
            return
        try:
            from river.tree import HoeffdingTreeClassifier
            from river.drift import ADWIN

            # HoeffdingTree is better for capturing non-linear interactions (Claude Audit Item 10)
            self.model = HoeffdingTreeClassifier(grace_period=100, max_depth=8)
            self.drift_detector = ADWIN(delta=0.002)
            self._initialized = True
            print("[Online] River HoeffdingTreeClassifier initialized (pattern-aware)")
        except ImportError:
            print("[Online] WARNING: river not installed. Online learning disabled.")
            self._initialized = False

    def _build_features(self, recent_colors: list[str], markov_probs: dict = None, regime_info: dict = None) -> dict:
        """Build enriched features including Markov and regime data."""
        if len(recent_colors) < 20:
            return None

        from collections import Counter

        features = {}
        color_map = {'T': 0, 'CT': 1, 'Bonus': 2}

        # Lag features
        for lag in range(1, 11):
            features[f'lag_{lag}'] = color_map.get(recent_colors[-lag], 0)

        # Streak
        streak = 1
        for i in range(len(recent_colors) - 2, max(len(recent_colors) - 20, -1), -1):
            if recent_colors[i] == recent_colors[-1]:
                streak += 1
            else:
                break
        features['streak'] = streak

        # Frequencies
        last_20 = recent_colors[-20:]
        c = Counter(last_20)
        features['freq_t_20'] = c.get('T', 0) / 20
        features['freq_ct_20'] = c.get('CT', 0) / 20
        features['freq_bonus_20'] = c.get('Bonus', 0) / 20

        # Kịch Kim Enrichment: Markov Probs
        if markov_probs:
            features['m_t'] = markov_probs.get('T', 0.33)
            features['m_ct'] = markov_probs.get('CT', 0.33)
            features['m_b'] = markov_probs.get('Bonus', 0.05)

        # Kịch Kim Enrichment: Regime signals
        if regime_info:
            features['switch_rate'] = regime_info.get('switch_rate', 0.5)
            features['is_streak'] = (1.0 if regime_info.get('regime') == 'STREAK' else 0.0)

        return features

    def predict(self, recent_colors: list[str], markov_probs: dict = None, regime_info: dict = None) -> dict:
        """Predict next color using online model."""
        self._init_models()
        if not self._initialized or self.n_updates < 50:
            return {'T': 7/15, 'CT': 7/15, 'Bonus': 1/15}

        features = self._build_features(recent_colors, markov_probs, regime_info)
        if features is None:
            return {'T': 7/15, 'CT': 7/15, 'Bonus': 1/15}

        try:
            probs = self.model.predict_proba_one(features)
            result = {
                'T': probs.get('T', 7/15),
                'CT': probs.get('CT', 7/15),
                'Bonus': probs.get('Bonus', 1/15),
            }
            # Normalize
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
            return result
        except Exception:
            return {'T': 7/15, 'CT': 7/15, 'Bonus': 1/15}

    def update(self, recent_colors: list[str], actual_color: str, markov_probs: dict = None, regime_info: dict = None) -> dict:
        """Learn from one new observation. Returns drift info."""
        self._init_models()
        if not self._initialized:
            return {'updated': False, 'drift': False}

        features = self._build_features(recent_colors, markov_probs, regime_info)
        if features is None:
            return {'updated': False, 'drift': False}

        # Predict before learning (for accuracy tracking)
        try:
            pred_probs = self.model.predict_proba_one(features)
            predicted = max(pred_probs, key=pred_probs.get) if pred_probs else None
        except Exception:
            predicted = None

        # Learn
        try:
            self.model.learn_one(features, actual_color)
            self.n_updates += 1
        except Exception:
            return {'updated': False, 'drift': False}

        # Track accuracy
        correct = 1 if predicted == actual_color else 0
        self.total_count += 1
        self.correct_count += correct

        # Drift detection
        drift_detected = False
        try:
            self.drift_detector.update(1 - correct)  # feed error rate
            if self.drift_detector.drift_detected:
                drift_detected = True
                self._drift_detected_now = True # Latch for one health check
                self.drift_alerts.append({
                    'time': int(time.time()),
                    'n_updates': self.n_updates,
                    'accuracy': round(self.correct_count / max(1, self.total_count), 4)
                })
                print(f"[Online] DRIFT detected at update #{self.n_updates}!")
        except Exception:
            pass

        return {
            'updated': True,
            'n_updates': self.n_updates,
            'drift': drift_detected,
            'accuracy': round(self.correct_count / max(1, self.total_count), 4)
        }

    @property
    def drift_detected(self):
        val = self._drift_detected_now
        self._drift_detected_now = False # Reset on read
        return val

    def get_stats(self) -> dict:
        return {
            'n_updates': self.n_updates,
            'accuracy': round(self.correct_count / max(1, self.total_count), 4) if self.total_count > 0 else 0,
            'total_predictions': self.total_count,
            'correct_predictions': self.correct_count,
            'drift_alerts': self.drift_alerts[-10:],  # last 10
            'initialized': self._initialized,
        }

    def save(self, path: str = None):
        path = path or SAVE_PATH
        if not self._initialized:
            return
        data = {
            'model': self.model,
            'drift_detector': self.drift_detector,
            'n_updates': self.n_updates,
            'correct_count': self.correct_count,
            'total_count': self.total_count,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Online] Saved ({self.n_updates} updates)")

    def load(self, path: str = None) -> bool:
        path = path or SAVE_PATH
        if not os.path.exists(path):
            return False
        self._init_models()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.drift_detector = data['drift_detector']
        self.n_updates = data['n_updates']
        self.correct_count = data['correct_count']
        self.total_count = data['total_count']
        self._initialized = True
        print(f"[Online] Loaded ({self.n_updates} updates)")
        return True
