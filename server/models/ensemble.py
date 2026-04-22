"""
ensemble.py - Dynamic Weighted Ensemble
Kết hợp 4 models (TFT, LSTM, Statistical, RL) với dynamic weight EMA
"""
import json
import os
import math
import pickle
import numpy as np
from collections import defaultdict, deque

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'saved', 'ensemble_weights.pkl')
MODEL_NAMES = ['tft', 'lstm', 'statistical', 'rl_agent', 'markov', 'foundation', 'mamba']

class StackingMetaLearner:
    """Item 17: Meta-Learning Stacker (Stacking base models)."""

    def __init__(self):
        self.model = None
        self._initialized = False

    def _init_model(self):
        if self._initialized: return
        try:
            from river.linear_model import LogisticRegression
            from river.multiclass import OneVsRestClassifier
            self.model = OneVsRestClassifier(LogisticRegression())
            self._initialized = True
        except ImportError:
            pass

    def predict(self, model_predictions: dict) -> dict:
        """Combine model outputs using meta-model."""
        if not self._initialized: return None
        
        # Prepare feature vector: [tft_T, tft_CT, tft_Bonus, lstm_T, ...]
        features = {}
        for m_name in MODEL_NAMES:
            preds = model_predictions.get(m_name, {'T': 0.33, 'CT': 0.33, 'Bonus': 0.33})
            if preds is None:
                preds = {'T': 0.33, 'CT': 0.33, 'Bonus': 0.33}
            for color in ['T', 'CT', 'Bonus']:
                features[f"{m_name}_{color}"] = preds[color]
        
        try:
            return self.model.predict_proba_one(features)
        except:
            return None

    def update(self, model_predictions: dict, actual_color: str):
        """Learn which models were correct in this context."""
        self._init_model()
        if not self._initialized: return
        
        features = {}
        for m_name in MODEL_NAMES:
            preds = model_predictions.get(m_name, {'T': 0.33, 'CT': 0.33, 'Bonus': 0.33})
            if preds is None:
                preds = {'T': 0.33, 'CT': 0.33, 'Bonus': 0.33}
            for color in ['T', 'CT', 'Bonus']:
                features[f"{m_name}_{color}"] = preds[color]
        
        self.model.learn_one(features, actual_color)


class DynamicEnsemble:
    def __init__(self, alpha: float = 0.05, use_stacking: bool = True):
        self.weights = {name: 0.165 for name in MODEL_NAMES}
        self.weights['tft'] = 0.01 # Item 5: Deprioritized until True Temporal verified
        self.alpha = alpha  # EMA smoothing factor
        self.history = []  # track weight evolution
        self.prediction_log = []  # track individual model accuracy
        # Item 15: Per-model rolling accuracy (last 100 rounds)
        self.model_accuracy = {name: deque(maxlen=100) for name in MODEL_NAMES}
        self.stacker = StackingMetaLearner() if use_stacking else None
        self.use_stacking = use_stacking

    def predict(self, model_predictions: dict) -> dict:
        """
        Combine predictions from multiple models.
        model_predictions: {
            'tft':         {T: 0.45, CT: 0.50, Bonus: 0.05},
            'lstm':        {T: 0.40, CT: 0.55, Bonus: 0.05},
            'statistical': {T: 0.48, CT: 0.47, Bonus: 0.05},
            'rl_agent':    {T: 0.42, CT: 0.52, Bonus: 0.06},
        }
        Returns: {color: "CT", confidence: 0.63, probs: {...}, model_votes: {...}}
        """
        # Weighted average of probability vectors
        combined = {'T': 0.0, 'CT': 0.0, 'Bonus': 0.0}

        active_weights = {}
        total_weight = 0

        for name in MODEL_NAMES:
            if name in model_predictions and model_predictions[name] is not None:
                active_weights[name] = self.weights[name]
                total_weight += self.weights[name]

        if total_weight == 0:
            return {
                'color': 'T', 'confidence': 0.33,
                'probs': {'T': 0.33, 'CT': 0.33, 'Bonus': 0.33},
                'model_votes': {}
            }

        # Normalize active weights
        for name in active_weights:
            active_weights[name] /= total_weight

        # ───── Item 17: Stacking Staging ─────
        used_stacking = False
        if self.use_stacking:
            stack_probs = self.stacker.predict(model_predictions)
            if stack_probs:
                combined = stack_probs
                used_stacking = True
        
        if not used_stacking:
            # Fallback to weighted EMA average
            combined = {'T': 0.0, 'CT': 0.0, 'Bonus': 0.0}
            for name, weight in self.weights.items():
                if name in model_predictions:
                    preds = model_predictions[name]
                    if preds is not None:
                        for color, prob in preds.items():
                            combined[color] += prob * weight

        # Normalize combined probs
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        # Item 12: Temperature Scaling removed as a temperature > 1.0 pulls the 
        # naturally low Bonus probability (6.6%) artificially closer to 33%, causing UI confusion.
        # combined = self._apply_temperature(combined, temperature=1.5)

        # Best color
        best_color = max(combined, key=combined.get)
        confidence = combined[best_color]

        # Model votes: which color each model picks
        model_votes = {}
        for name in model_predictions:
            if name not in self.weights: continue
            preds = model_predictions[name]
            if preds is None: continue
            vote = max(preds, key=preds.get)
            model_votes[name] = {
                'vote': vote,
                'confidence': round(preds[vote], 4),
                'weight': round(self.weights[name], 4),
                'probs': preds
            }

        return {
            'color': best_color,
            'confidence': round(confidence, 4),
            'probs': {k: round(v, 4) for k, v in combined.items()},
            'model_votes': model_votes
        }

    def update_weights(self, model_predictions: dict, actual_color: str):
        """
        Update weights using EMA based on which models were correct.
        Called after each round's actual result is known.
        """
        for name in MODEL_NAMES:
            if name not in model_predictions:
                continue

            preds = model_predictions[name]
            if preds is None:
                continue
            predicted_color = max(preds, key=preds.get)
            correct = 1.0 if predicted_color == actual_color else 0.0

            # EMA update
            self.weights[name] = self.alpha * correct + (1 - self.alpha) * self.weights[name]

            # Item 15: Track per-model accuracy
            self.model_accuracy[name].append(int(correct))

        # Item 17: Online learning for Meta-Stacker
        if self.use_stacking:
            self.stacker.update(model_predictions, actual_color)

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Log
        self.history.append(dict(self.weights))
        if len(self.history) > 1000:
            self.history = self.history[-500:]

    def get_weights(self) -> dict:
        return {k: round(v, 4) for k, v in self.weights.items()}

    def get_weight_history(self, last_n: int = 100) -> list[dict]:
        return self.history[-last_n:]

    def save(self, path: str = None):
        path = path or SAVE_PATH
        data = {'weights': dict(self.weights), 'alpha': self.alpha}
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Ensemble] Saved weights: {self.get_weights()}")

    def load(self, path: str = None) -> bool:
        path = path or SAVE_PATH
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.alpha = data.get('alpha', 0.05)
        # Ensure all MODEL_NAMES have weights
        for name in MODEL_NAMES:
            if name not in self.weights:
                self.weights[name] = 0.20
        print(f"[Ensemble] Loaded weights: {self.get_weights()}")
        return True

    @staticmethod
    def _apply_temperature(probs: dict, temperature: float = 1.5) -> dict:
        """Temperature Scaling: calibrate overconfident predictions."""
        scaled = {k: math.exp(math.log(max(v, 1e-9)) / temperature) for k, v in probs.items()}
        total = sum(scaled.values())
        return {k: v / total for k, v in scaled.items()}

    def get_model_accuracy(self) -> dict:
        """Get rolling accuracy per model (last 100 rounds)."""
        result = {}
        for name in MODEL_NAMES:
            history = self.model_accuracy[name]
            if len(history) > 0:
                result[name] = round(sum(history) / len(history), 4)
            else:
                result[name] = 0.0
        return result
