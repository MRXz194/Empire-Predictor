"""
foundation_model.py - Item 20: Time-Series Foundation Models
Uses NeuralForecast (NHITS) for global pattern recognition.
"""
import os
import pandas as pd
import numpy as np
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved')
MODEL_PATH = os.path.join(SAVE_DIR, 'foundation_nhits')

class FoundationModel:
    def __init__(self, h=1):
        self.h = h # horizon
        self.model = None
        self.trained = False
        os.makedirs(SAVE_DIR, exist_ok=True)

    def _prepare_df(self, colors: list[str]):
        # NeuralForecast expects long-format df with ds, y, unique_id
        df = pd.DataFrame({
            'unique_id': [1.0] * len(colors),
            'ds': pd.date_range(start='2020-01-01', periods=len(colors), freq='h'),
            'y': [1.0 if c == 'T' else (2.0 if c == 'CT' else 0.0) for c in colors]
        })
        return df

    def train(self, colors: list[str]):
        print(f"[Foundation] Fine-tuning NHITS on {len(colors)} samples...")
        df = self._prepare_df(colors)
        
        # Define model: NHITS is a strong global forecasting model
        models = [NHITS(input_size=100, h=self.h, max_steps=100)]
        self.model = NeuralForecast(models=models, freq='h')
        
        self.model.fit(df)
        self.trained = True
        print("[Foundation] NHITS trained.")

    def predict(self, recent_colors: list[str]) -> dict:
        if not self.trained or self.model is None or not recent_colors:
            return {'T': 0.46, 'CT': 0.46, 'Bonus': 0.08}
        
        try:
            df = self._prepare_df(recent_colors)
            forecast = self.model.predict(df=df)
            
            # NHITS returns a continuous value. We map it back to probs using Softmax distance.
            # Value mapping: 0=Bonus, 1=T, 2=CT
            val = float(forecast['NHITS'].values[0])
            
            import math
            # Softmax distance mapping (Claude Audit Item 7)
            dist = {'Bonus': abs(val - 0), 'T': abs(val - 1), 'CT': abs(val - 2)}
            inv = {k: math.exp(-v * 2.0) for k, v in dist.items()} # Higher v -> lower prob
            total = sum(inv.values())
            probs = {k: v / total for k, v in inv.items()}
                
            return probs
        except Exception as e:
            print(f"[Foundation] Prediction error: {e}")
            return {'T': 0.46, 'CT': 0.46, 'Bonus': 0.08}

    def save(self):
        if self.model and self.trained:
            try:
                self.model.save(path=MODEL_PATH, overwrite=True)
                print(f"[Foundation] Model saved to {MODEL_PATH}")
            except Exception as e:
                print(f"[Foundation] Save error: {e}")

    def load(self) -> bool:
        if not os.path.exists(MODEL_PATH):
            return False
        try:
            # NeuralForecast.load requires static method
            self.model = NeuralForecast.load(path=MODEL_PATH)
            self.trained = True
            print(f"[Foundation] Model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[Foundation] Load error: {e}")
            return False
