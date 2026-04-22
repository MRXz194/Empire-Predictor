"""
optimization.py - Item 22: Hyperparameter Optimization using Optuna
Finds the best parameters for ensemble_alpha, confidence_threshold, and bet_fraction.
"""
import optuna
import os
import sys

# Ensure server path is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.engine import run_backtest
from database import get_rolls_for_training

def objective(trial):
    # 1. Define hyperparameters to tune
    alpha = trial.suggest_float("alpha", 0.01, 0.2, log=True)
    threshold = trial.suggest_float("threshold", 0.45, 0.70)
    bet_fraction = trial.suggest_float("bet_fraction", 0.01, 0.15)
    sequence_len = trial.suggest_int("seq_len", 20, 100)
    
    # 2. Get data for backtest
    rolls = get_rolls_for_training(limit=2000)
    if not rolls:
        return 0.0
    colors = [r['color'] for r in rolls]
    
    # 3. Run backtest with these params
    # Note: run_backtest needs to be slightly modified or wrapped to take these params.
    # For now, let's assume we can pass them to run_backtest or global config.
    try:
        result = run_backtest(
            colors, 
            strategy='ensemble', 
            confidence_threshold=threshold,
            warmup=sequence_len + 100
        )
        
        # We want to maximize ROI and minimize Drawdown
        roi = result.get('roi', 0.0)
        max_dd = result.get('max_drawdown', 0.0)
        
        # Combined score: ROI - 0.5 * Drawdown
        score = roi - (0.5 * max_dd)
        return score
    except Exception as e:
        print(f"Trial failed: {e}")
        return -1.0

def run_optimization(n_trials=100):
    print(f"Starting Optuna optimization session ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "="*40)
    print("Optimization Complete!")
    print(f"Best score: {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*40)
    
    return study.best_params

if __name__ == "__main__":
    run_optimization(n_trials=50)
