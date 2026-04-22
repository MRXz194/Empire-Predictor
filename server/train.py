"""
train.py - Full Clean-Slate Training Pipeline
Executes training for all models and saves weights to models/saved/
"""
import os
import sys
import numpy as np

# Add current dir to path
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

# Correct imports from main.py logic
from database import get_rolls_for_training
from models.markov import MarkovChain
from models.statistical import StatisticalModel
from models.rl_agent import QLearningAgent
from models.tft_model import TFTModel
from models.mamba_model import MambaPredictor
from models.lstm_model import LSTMPredictor
from models.foundation_model import FoundationModel
from models.features import prepare_sequences
from models.ensemble import DynamicEnsemble

def main():
    print("============================================================")
    print("  CSGOEmpire Predictor - Full System Training (Clean Slate)")
    print("============================================================")
    
    # 1. Load Data
    print("[1/7] Fetching rolls from database...")
    rolls = get_rolls_for_training()
    if not rolls:
        print("Error: No data found in database!")
        return
    
    colors = [r['color'] for r in rolls]
    print(f"Total rolls: {len(rolls)}")
    
    # 2. Train Markov & Statistical (Fast)
    print("[2/7] Checking Markov and Statistical models...")
    markov = MarkovChain()
    if not os.path.exists(os.path.join(SERVER_DIR, 'models', 'saved', 'markov.pkl')):
        markov.train(colors)
        markov.save()
    else:
        print("  Skipping Markov: Already exists.")
        markov.load()
    
    stat = StatisticalModel()
    if not os.path.exists(os.path.join(SERVER_DIR, 'models', 'saved', 'statistical.pkl')):
        stat.train(colors)
        stat.save()
    else:
        print("  Skipping Statistical: Already exists.")
        stat.load()
    
    # 3. Prepare Sequences for Deep Learning
    print("[3/7] Preparing sequences (Lookback=60)...")
    X, y = prepare_sequences(rolls, seq_len=60, markov_model=markov)
    print(f"Generated {len(X)} sequences.")
    
    # 4. Train Deep Learning Models (Heavy)
    print("[4/7] Training Mamba, LSTM, TFT...")
    
    # Prepare common features count
    n_features = X.shape[2]
    
    # Mamba
    print("Starting Mamba...")
    mamba = MambaPredictor(n_features=n_features)
    if not os.path.exists(os.path.join(SERVER_DIR, 'models', 'saved', 'mamba_weights.pth')):
        mamba.train_model(X, y, epochs=10) 
        mamba.save()
    else:
        print("  Skipping Mamba: Already exists.")
    
    # LSTM
    print("Starting LSTM...")
    lstm = LSTMPredictor(n_features=n_features)
    if not os.path.exists(os.path.join(SERVER_DIR, 'models', 'saved', 'lstm_weights.weights.h5')):
        lstm.build()
        lstm.train(X, y, epochs=5)
        lstm.save()
    else:
        print("  Skipping LSTM: Already exists.")
    
    # TFT (True Temporal)
    print("Starting TFT (True Temporal)...")
    # Item 5 fix: TFT uses the new group-based logic
    tft = TFTModel(seq_len=60, n_features=n_features)
    tft.train(X, y, epochs=5)
    tft.save()
    
    # 5. Train Foundation Model
    print("[5/7] Training Foundation Model (NHITS)...")
    foundation = FoundationModel()
    foundation.train(colors)
    foundation.save()
    
    # 6. Train RL Agent
    print("[6/7] Training RL Agent (Bonus-aware)...")
    rl = QLearningAgent()
    rl.train(rolls, episodes=1)
    rl.save()
    
    # 7. Initialize Ensemble
    print("[7/7] Resetting Ensemble weights...")
    ens = DynamicEnsemble()
    ens.save()
    
    print("============================================================")
    print("  Training Complete! All models saved to server/models/saved/")
    print("============================================================")

if __name__ == "__main__":
    main()
