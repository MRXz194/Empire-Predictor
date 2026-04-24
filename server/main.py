"""
main.py - FastAPI Server (Tầng 1 endpoint + kết nối tất cả 8 tầng)
"""
import os
import sys
import json
import time
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import Optional

# Add server dir to path
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SERVER_DIR)

from database import init_db, insert_roll, get_recent_rolls, get_rolls_for_training, get_roll_count, insert_prediction, update_prediction_result
from database import get_prediction_accuracy, get_prediction_by_round, get_daily_stats
from models.tft_model import TFTModel
from models.statistical import StatisticalModel
from models.rl_agent import QLearningAgent
from models.ensemble import DynamicEnsemble
from models.markov import MarkovChain
from models.foundation_model import FoundationModel
from models.mamba_model import MambaPredictor
from models.features import compute_features_array, prepare_sequences, outcome_to_color
from engine.decision import make_decision
from engine.session import HealthScore
from reporting.stats import get_overall_stats, get_prediction_stats, get_streak_analysis
from reporting.alerts import check_high_confidence_alert
from backtest.engine import run_backtest
from backtest.monte_carlo import run_monte_carlo

# ── Global state ─────────────────────────────────────────────────────────────
tft_predictor = TFTModel()
stat_model = StatisticalModel()
rl_agent = QLearningAgent(epsilon=0.05)
markov_model = MarkovChain()
foundation_model = FoundationModel()
mamba_predictor = MambaPredictor()
ensemble = DynamicEnsemble()
health_scorer = HealthScore()
lstm_predictor = None  # Lazy loaded
online_learner = None  # Lazy loaded

connected_clients: list[WebSocket] = []
recent_colors_cache: list[str] = []  # in-memory cache of recent colors
last_pred = None
last_prediction_timestamp = 0
last_processed_round_id = 0 # Kịch Kim 4.5: Anti-duplicate guard
is_warmed_up = False        # Kịch Kim 4.6: Sequence Contiguity flag

# Bankroll tracking
user_bankroll = 100.0
user_kelly_mult = 0.5 # Default balanced


# ── Models ───────────────────────────────────────────────────────────────────

class BankrollUpdate(BaseModel):
    bankroll: float
    kelly_mult: Optional[float] = 0.5

class RollInput(BaseModel):
    round_id: int
    outcome: int
    color: Optional[str] = None
    timestamp: Optional[int] = None
    history_full: Optional[list[str]] = None # Kịch Kim 4.8

class BacktestParams(BaseModel):
    strategy: str = 'ensemble'
    bankroll: float = 100.0
    confidence_threshold: float = 0.45

class MonteCarloParams(BaseModel):
    n_simulations: int = 10000
    n_rounds: int = 100
    bankroll: float = 100.0
    bet_fraction: float = 0.05

# ── Lifespan ─────────────────────────────────────────────────────────────────
is_retraining = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + load/train models."""
    init_db()
    
    global recent_colors_cache, last_processed_round_id
    recent = get_recent_rolls(500)
    recent_colors_cache = [r['color'] for r in reversed(recent)]
    if recent:
        if recent:
            last_processed_round_id = recent[0]['round_id']
            print(f"[Lifespan] Initialized last_processed_round_id to {last_processed_round_id}")
        else:
            print("[Lifespan] Database is empty. last_processed_round_id initialized to 0.")
        # Kịch Kim 4.6 Debug
        print(f"[Core] Cache Purity Check: First={recent_colors_cache[0]}, Last={recent_colors_cache[-1]} (Total: {len(recent_colors_cache)})")
    
    import threading
    threading.Thread(target=_load_or_train_models, daemon=True).start()
    
    yield

    # Shutdown
    tft_predictor.save()
    rl_agent.save()
    ensemble.save()
    stat_model.save()
    markov_model.save()
    foundation_model.save()
    if mamba_predictor: mamba_predictor.save()
    if online_learner: online_learner.save()


def _load_or_train_models():
    global lstm_predictor, online_learner
    tft_predictor.load()
    rl_agent.load()
    ensemble.load()
    stat_model.load()
    markov_model.load()
    foundation_model.load()
    if mamba_predictor: mamba_predictor.load()
    
    # Trigger background train if missing
    needs_train = (
        not tft_predictor.trained or 
        not rl_agent.trained or 
        not foundation_model.trained or 
        (mamba_predictor and not mamba_predictor.trained) or 
        (lstm_predictor and not lstm_predictor.trained)
    )
    if needs_train:
        print("[Server] Key models missing. Starting background training...")
        import threading
        threading.Thread(target=_run_background_retrain, daemon=True).start()

    # Lazy loads
    try:
        from models.lstm_model import LSTMPredictor
        lstm_predictor = LSTMPredictor()
        lstm_predictor.load()
    except: pass

    try:
        from learning.online import OnlineLearner
        online_learner = OnlineLearner()
        online_learner.load()
    except: pass


def _run_background_retrain():
    global is_retraining
    if is_retraining: return
    is_retraining = True
    try:
        rolls = get_rolls_for_training()
        X, y = prepare_sequences(rolls, seq_len=60, markov_model=markov_model)
        if len(X) > 100:
            tft_predictor.train(X, y, epochs=3)
            stat_model.train([r['color'] for r in rolls])
            rl_agent.train(rolls, episodes=1)
            
            if mamba_predictor:
                mamba_predictor.train_model(X, y, epochs=3)
                mamba_predictor.save()
            if lstm_predictor:
                lstm_predictor.train(X, y, epochs=5)
                lstm_predictor.save()
            if foundation_model:
                foundation_model.train([r['color'] for r in rolls])
                foundation_model.save()
                
            tft_predictor.save()
            rl_agent.save()
    finally:
        is_retraining = False


# ── Core Logic ───────────────────────────────────────────────────────────────

def _get_all_model_predictions(regime: str = 'STABLE') -> dict:
    """Helper to gather predictions from all 7 models."""
    global is_warmed_up, mamba_predictor, tft_predictor, lstm_predictor
    preds = {}
    import numpy as np
    
    # 1. Statistical & Markov & Foundation (Basic patterns)
    if len(recent_colors_cache) >= 20:
        preds['statistical'] = stat_model.predict(recent_colors_cache)
        preds['markov'] = markov_model.predict(recent_colors_cache)
        preds['foundation'] = foundation_model.predict(recent_colors_cache)
    
    # 2. Sequential models (TFT, Mamba, LSTM) - Kịch Kim 4.6 Warm-up guard
    if is_warmed_up:
        # Fetch 150 from DB to ensure enough context for internal feature engineering
        rolls = get_recent_rolls(150)
        rolls_dicts = list(reversed(rolls))
        all_c_for_markov = [r['color'] for r in rolls_dicts]
        
        seq = []
        # Need exactly 60 steps for the sequential models input
        for j in range(len(rolls_dicts) - 60, len(rolls_dicts)):
            m_p = markov_model.predict(all_c_for_markov[:j])
            feat = compute_features_array(rolls_dicts, j, lookback=20, markov_probs=m_p)
            if feat is not None: seq.append(feat)
            
        if len(seq) == 60:
            X = np.array([seq], dtype=np.float32)
            preds['tft'] = tft_predictor.predict(X)
            if mamba_predictor and mamba_predictor.trained:
                preds['mamba'] = mamba_predictor.predict(X)
            if lstm_predictor and lstm_predictor.trained:
                preds['lstm'] = lstm_predictor.predict(X)
    else:
        # Neutral if not warmed up
        n_p = {'T': 0.33, 'CT': 0.33, 'Bonus': 0.08}
        preds['tft'] = n_p; preds['mamba'] = n_p; preds['lstm'] = n_p
                
    # 3. RL Agent
    if len(recent_colors_cache) >= 30:
        preds['rl_agent'] = rl_agent.predict(recent_colors_cache, regime=regime)
        
    return preds


def _predict_next(regime_info: dict = None, drift: bool = False) -> dict:
    global user_bankroll, user_kelly_mult
    
    # Kịch Kim 3.1: Extract string safely
    if regime_info and isinstance(regime_info, dict):
        regime_str = regime_info.get('regime', 'STABLE')
        if isinstance(regime_str, dict): # nested check
            regime_str = regime_str.get('regime', 'STABLE')
    else:
        regime_str = 'STABLE'
        
    model_preds = _get_all_model_predictions(regime=regime_str)
    if not model_preds:
        return {'action': 'SKIP', 'skip_reason': 'Not enough data', 'probs': {'T': 0.33, 'CT': 0.33, 'Bonus': 0.33}, 'model_votes': {}}

    ensemble_res = ensemble.predict(model_preds)
    
    # Kịch Kim Decision
    decision = make_decision(
        ensemble_res, 
        user_bankroll, 
        regime_info=regime_info,
        kelly_mult=user_kelly_mult
    )
    
    return decision


def _process_roll_sync(roll: RollInput, color: str):
    global recent_colors_cache, user_bankroll, last_pred, online_learner, last_processed_round_id, is_warmed_up
    
    # 0. Gap & Duplicate Guard (Kịch Kim 4.6 Elite Sync)
    if roll.round_id <= last_processed_round_id:
        print(f"[Core] Skipping duplicate round_id: {roll.round_id}")
        return None, None
    
    if last_processed_round_id != 0 and roll.round_id > last_processed_round_id + 1:
        gap = roll.round_id - last_processed_round_id
        print(f"[Core] ⚠️ GAP detected ({gap} rounds)! LastID={last_processed_round_id}, NewID={roll.round_id}")
        if roll.history_full:
            recent_colors_cache = roll.history_full[-500:]
            is_warmed_up = True if len(recent_colors_cache) >= 60 else False
            print(f"[Core] 🔄 Sequence Healing: Synced {len(recent_colors_cache)} rounds from live socket. Ready={is_warmed_up}")
        else:
            print(f"[Core] ❌ No live history in payload. Sequence broken. Flushing.")
            recent_colors_cache = []
            is_warmed_up = False
    
    # Kịch Kim 4.8 Power-up: Force override if current cache is empty or suspect
    if (len(recent_colors_cache) < 60 or last_processed_round_id == 0) and roll.history_full:
        recent_colors_cache = roll.history_full[-500:]
        print(f"[Core] ⚡ Elite Bootstrapping: Loaded {len(recent_colors_cache)} rounds. Ready.")
        is_warmed_up = True if len(recent_colors_cache) >= 60 else False

    last_processed_round_id = roll.round_id

    # 1. Update memory cache (Kịch Kim 4.8.1 Fix: Always sync from history_full if available)
    if roll.history_full and len(roll.history_full) > 0:
        recent_colors_cache = roll.history_full[-500:]
        is_warmed_up = True if len(recent_colors_cache) >= 60 else False
        print(f"[Core] 🔄 Sync from Live Socket: {len(recent_colors_cache)} rounds. Tail: {recent_colors_cache[-3:]}")
    else:
        # Fallback to single append
        recent_colors_cache.append(color)
        if len(recent_colors_cache) > 500: recent_colors_cache = recent_colors_cache[-500:]
        if len(recent_colors_cache) >= 60: is_warmed_up = True
        print(f"[Core] ➕ Appended Roll: {color}. Total: {len(recent_colors_cache)}")

    # 2. Database Storage
    insert_roll(roll.round_id, roll.outcome, color, roll.timestamp)
    
    # Warm-up check (Min 60 contiguous for LSTM/Mamba)
    if len(recent_colors_cache) >= 60:
        is_warmed_up = True
    
    # 2. Regime Detection for RL & Online
    h_tmp = health_scorer.compute(recent_colors_cache)
    # Extract the string name from the regime dict
    regime_data = h_tmp.get('regime', {})
    regime_str = regime_data.get('regime', 'STABLE') if isinstance(regime_data, dict) else str(regime_data)

    # 3. Ensemble Update
    all_preds_last = _get_all_model_predictions(regime=regime_str)
    ensemble.update_weights(all_preds_last, color)
    
    # 4. RL Update
    if last_pred and rl_agent:
        rl_vote = last_pred.get('model_votes', {}).get('rl_agent', {}).get('vote', 'skip')
        rl_action = f'bet_{rl_vote.lower()}' if rl_vote != 'skip' else 'skip'
        rl_agent.update(recent_colors_cache, rl_action, color, regime=regime_str)

    # 5. Online Learner & Drift
    drift_detected = False
    if online_learner and len(recent_colors_cache) >= 20:
        m_p = markov_model.predict(recent_colors_cache[:-1])
        # Online learner needs the FULL dict (switch_rate, etc)
        reg_info = regime_data 
        
        drift_res = online_learner.update(recent_colors_cache[:-1], color, markov_probs=m_p, regime_info=reg_info)
        drift_detected = drift_res.get('drift', False)
        
        if drift_detected or (drift_res.get('n_updates', 0) % 500 == 0):
            import threading
            threading.Thread(target=_run_background_retrain, daemon=True).start()

    # 5. Final Prediction + Health
    health_data = health_scorer.compute(recent_colors_cache, drift=drift_detected)
    prediction = _predict_next(regime_info=health_data.get('regime'), drift=drift_detected)
    
    # 6. Database tracking
    update_prediction_result(roll.round_id, color)
    insert_prediction(roll.round_id + 1, prediction['color'], prediction['confidence'], prediction.get('model_votes', {}), prediction.get('bet_amount', 0))
    
    return prediction, health_data


# ── FastAPI Endpoints ───────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/roll")
async def receive_roll(roll: RollInput):
    """Receive and process a new round result."""
    try:
        color = roll.color or outcome_to_color(roll.outcome)
        print(f"[Core] 📥 Received Roll: Round={roll.round_id}, Color={color}")
        
        prediction, health_data = await run_in_threadpool(_process_roll_sync, roll, color)
        
        # If _process_roll_sync returns None (duplicate), we still want to notify dashboard if it was a valid sync
        if prediction is None:
             # Just inform the dashboard of the current state without triggering a new prediction logic
             await _broadcast({
                 'type': 'status', 
                 'last_id': last_processed_round_id,
                 'warmed_up': is_warmed_up
             })
             return {"status": "skipped", "round_id": roll.round_id}

        global last_pred, last_prediction_timestamp
        last_pred = prediction
        last_prediction_timestamp = int(time.time() * 1000)
        
        ws_data = {
            'type': 'roll',
            'roll': {'round_id': roll.round_id, 'outcome': roll.outcome, 'color': color},
            'prediction': prediction,
            'health': health_data,
            'recent': recent_colors_cache[-30:],
            'warmed_up': is_warmed_up
        }
        await _broadcast(ws_data)
        print(f"[Core] ✅ Processed & Broadcasted Roll: Round={roll.round_id}")
        return {"status": "ok", "round_id": roll.round_id}
        
    except Exception as e:
        print(f"[Core] ❌ ERROR processing roll {roll.round_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/api/sync")
async def sync_history(data: dict):
    """Kịch Kim 4.6: Bulk sync of contiguous history from extension."""
    global recent_colors_cache, last_processed_round_id, is_warmed_up
    
    rolls = data.get('rolls', [])
    if not rolls:
        return {"status": "error", "message": "No rolls provided"}
    
    # 1. Persist to DB first
    for r in rolls:
        rid = r.get('round_id', 0)
        color = r.get('color')
        if rid > 0 and color:
            insert_roll(rid, 0, color) # outcome 0 for historical sync if unknown
    
    # 2. Update memory cache
    new_colors = [r.get('color') for r in rolls if r.get('color')]
    if new_colors:
        recent_colors_cache = new_colors[-500:]
        last_processed_round_id = max(r.get('round_id', 0) for r in rolls)
        if len(recent_colors_cache) >= 60:
            is_warmed_up = True
        
        print(f"[Core] 🔄 Synced & Persisted {len(new_colors)} rolls. Ready={is_warmed_up}. LastID={last_processed_round_id}")
        
        # Broadcast sync to dashboard
        await _broadcast({
            'type': 'sync',
            'recent': recent_colors_cache[-100:], # Send more for full display
            'last_id': last_processed_round_id
        })
        
    return {"status": "ok", "synced": len(new_colors)}

@app.post("/api/bankroll")
async def set_bankroll(data: BankrollUpdate):
    global user_bankroll, user_kelly_mult
    user_bankroll = data.bankroll
    if data.kelly_mult is not None:
        user_kelly_mult = data.kelly_mult
    return {"ok": True, "bankroll": user_bankroll, "kelly_mult": user_kelly_mult}

# ... (Additional standard endpoints like /api/stats, /api/health omitted for brevity but should be mapped to the new logic)
@app.get("/api/predict")
async def predict():
    return {"prediction": _predict_next(), "recent": recent_colors_cache[-30:]}

@app.get("/api/health")
async def health():
    return health_scorer.compute(recent_colors_cache)

@app.post("/api/backtest")
def backtest_endpoint(params: BacktestParams):
    try:
        from database import get_recent_rolls
        # Get enough history for warmup + simulation
        rolls = get_recent_rolls(1500)
        colors = [r['color'] for r in reversed(rolls)]
        result = run_backtest(
            colors=colors,
            strategy=params.strategy,
            bankroll=params.bankroll,
            confidence_threshold=params.confidence_threshold
        )
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/montecarlo")
def montecarlo_endpoint(params: MonteCarloParams):
    try:
        colors = recent_colors_cache[-500:]
        result = run_monte_carlo(
            historical_colors=colors,
            n_simulations=params.n_simulations,
            n_rounds=params.n_rounds,
            bankroll=params.bankroll,
            bet_fraction=params.bet_fraction
        )
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stats")
async def stats():
    return {
        "overall": get_overall_stats(), 
        "weights": ensemble.get_weights(),
        "predictions": get_prediction_stats(),
        "streaks": get_streak_analysis()
    }

@app.get("/api/recent")
async def get_recent(limit: int = 50):
    return {"rolls": [{"color": c} for c in recent_colors_cache[-limit:]], "total": len(recent_colors_cache)}

# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        prediction = _predict_next()
        await ws.send_json({
            'type': 'init',
            'recent': recent_colors_cache[-50:],
            'prediction': prediction,
            'health': health_scorer.compute(recent_colors_cache),
            'model_weights': ensemble.get_weights(),
            'stats': {
                "overall": get_overall_stats(),
                "predictions": get_prediction_stats(),
                "streaks": get_streak_analysis()
            },
        })
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(ws)

async def _broadcast(data: dict):
    for ws in list(connected_clients):
        try: await ws.send_json(data)
        except: connected_clients.remove(ws)

# Static Files
DASHBOARD_DIR = os.path.join(SERVER_DIR, '..', 'dashboard')
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

@app.get("/")
async def serve_dashboard():
    return FileResponse(os.path.join(DASHBOARD_DIR, 'index.html'))

@app.get("/style.css")
async def serve_css():
    return FileResponse(os.path.join(DASHBOARD_DIR, 'style.css'))

@app.get("/dashboard.js")
async def serve_js():
    return FileResponse(os.path.join(DASHBOARD_DIR, 'dashboard.js'))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
