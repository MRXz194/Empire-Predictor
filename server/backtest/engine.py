"""
engine.py — Backtesting Engine (Synchronized with Empire-Predictor 4.8)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.tft_model import TFTModel
from models.statistical import StatisticalModel
from models.rl_agent import QLearningAgent
from models.ensemble import DynamicEnsemble
from models.markov import MarkovChain
from models.foundation_model import FoundationModel
from models.mamba_model import MambaPredictor
from models.features import compute_features_array
from engine.decision import make_decision


def run_backtest(colors: list[str], strategy: str = 'ensemble',
                 bankroll: float = 100.0, min_bet: float = 1.0,
                 confidence_threshold: float = 0.55,
                 warmup: int = 1000, kelly_mult: float = 0.5) -> dict:
    """
    Replay historical color sequence and simulate betting.
    Supports all 7 'Empire-Predictor' modules.
    """
    if len(colors) < warmup + 50:
        return {'error': 'Not enough data', 'min_required': warmup + 50}

    # 1. Warmup training
    train_colors = colors[:warmup]
    print(f"[Backtest] Warming up on {warmup} rounds...")
    
    markov = MarkovChain()
    markov.train(train_colors)
    
    stat_model = StatisticalModel()
    stat_model.train(train_colors)

    rl = QLearningAgent(epsilon=0.10)
    rl.train_sequential(train_colors) # Faster than episodes for backtest

    found_model = FoundationModel()
    found_model.train(train_colors)
    
    ensemble = DynamicEnsemble()
    
    # Sequential models usually need more data/epochs, but we'll use small epochs for backtest speed
    # Or assuming they are already pre-trained for the SOTA baseline
    tft = TFTModel()
    mamba = MambaPredictor()
    # (Note: In a real simulation, we'd skip re-training deep models if they are heavy)

    # 2. Simulation
    initial_bankroll = bankroll
    bankroll_curve = [bankroll]
    results = []
    wins = 0
    losses = 0
    skips = 0
    total_bet = 0

    # Mock rolls for feature engineering
    all_rolls_mock = [{'color': c, 'outcome': 0, 'round_id': idx} for idx, c in enumerate(colors)]

    for i in range(warmup, len(colors)):
        recent = colors[max(0, i-100):i]
        actual = colors[i]

        # Get predictions from ALL models
        model_preds = {}
        markov_p = markov.predict(recent)
        
        # Sequential feature prep
        if i >= warmup + 60:
            seq = []
            for j in range(i - 60, i):
                m_p_step = markov.predict(colors[max(0, j-100):j])
                feat = compute_features_array(all_rolls_mock, j, lookback=20, markov_probs=m_p_step)
                if feat is not None: seq.append(feat)
                
            if len(seq) == 60:
                X = np.array([seq], dtype=np.float32)
                model_preds['tft'] = tft.predict(X)
                model_preds['mamba'] = mamba.predict(X)
            else:
                model_preds['tft'] = markov_p
                model_preds['mamba'] = markov_p
        else:
            model_preds['tft'] = markov_p
            model_preds['mamba'] = markov_p

        model_preds['statistical'] = stat_model.predict(recent)
        model_preds['rl_agent'] = rl.predict(recent)
        model_preds['markov'] = markov_p
        model_preds['foundation'] = found_model.predict(recent)
        
        # Decision
        if strategy == 'ensemble':
            ens_result = ensemble.predict(model_preds)
        else:
            # Standalone strategy
            probs = model_preds.get(strategy, markov_p)
            best = max(probs, key=probs.get)
            ens_result = {'probs': probs, 'color': best, 'confidence': probs[best], 'model_votes': {}}

        # For backtest, we don't have dynamic regime detection unless we compute it per step
        # Simulating it as STABLE for now, or using a simple detector
        decision = make_decision(ens_result, bankroll, min_bet, confidence_threshold, kelly_mult=kelly_mult)

        if decision['action'] == 'BET' and bankroll > 0:
            bet_color = decision['color']
            bet_amount = min(decision['bet_amount'], bankroll)
            total_bet += bet_amount

            if bet_color == actual:
                payout = bet_amount * (14 if actual == 'Bonus' else 2)
                bankroll += payout - bet_amount
                wins += 1
            else:
                bankroll -= bet_amount
                losses += 1
        else:
            skips += 1

        bankroll_curve.append(round(bankroll, 2))

        # Online updates
        if strategy == 'ensemble':
            ensemble.update_weights(model_preds, actual)
        rl.update(recent, f"bet_{decision['color'].lower()}" if decision['action']=='BET' else 'skip', actual)

    # 3. Final Metrics
    total_bets = wins + losses
    roi = (bankroll - initial_bankroll) / initial_bankroll
    
    return {
        'strategy': strategy,
        'total_rounds': len(colors) - warmup,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'skips': skips,
        'win_rate': round(wins/max(1, total_bets), 4),
        'roi': round(roi, 4),
        'final_bankroll': round(bankroll, 2),
        'bankroll_curve': bankroll_curve,
        'curve_sample': bankroll_curve[::max(1, len(bankroll_curve)//200)]
    }
