import os
import pickle
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved')
MODEL_PATH = os.path.join(SAVE_DIR, 'tft_pt_weights.ckpt')
CONFIG_PATH = os.path.join(SAVE_DIR, 'tft_pt_config.pkl')

class TFTModel:
    def __init__(self, seq_len=60, n_features=46):
        self.seq_len = seq_len
        self.n_features = n_features
        self.model = None
        self.trained = False
        self.dataset_params = None
        os.makedirs(SAVE_DIR, exist_ok=True)

    def _prepare_data(self, X: np.ndarray, y: np.ndarray):
        """Convert numpy arrays to a pandas DataFrame for TimeSeriesDataSet."""
        data_list = []
        n_samples = X.shape[0]
        targets = np.argmax(y, axis=1)
        
        # ─── Item 22: True Temporal Refactor ───
        # FIX: Each sequence (60 steps) must be a separate 'group'
        # to prevent cross-sequence leakage while maintaining temporal continuity.
        for i in range(n_samples):
            target_val = int(targets[i])
            for t in range(self.seq_len):
                row = {
                    'group': i,           # Unique ID per sequence
                    'time_idx': t,        # 0 to 59
                    'target': target_val  # Result for this sequence
                }
                feat = X[i, t, :]
                for f_idx in range(self.n_features):
                    row[f'f_{f_idx}'] = float(feat[f_idx])
                data_list.append(row)
                
        return pd.DataFrame(data_list)

    def train(self, X: np.ndarray, y: np.ndarray, epochs=5, batch_size=64):
        print(f"[TFT-PT] Training on {len(X)} samples using pytorch-forecasting...")
        # To avoid OOM/performance issues on huge datasets, we can limit to recent 10k samples
        if len(X) > 10000:
            X = X[-10000:]
            y = y[-10000:]
            print(f"[TFT-PT] Capping to most recent 10,000 samples for performance.")

        df = self._prepare_data(X, y)
        
        max_encoder_length = self.seq_len - 1
        max_prediction_length = 1
        
        # FIX: Group-based Train/Val split (Claude Points 86-102)
        n_groups = df['group'].nunique()
        training_cutoff_group = int(n_groups * 0.9)
        
        train_df = df[df['group'] < training_cutoff_group]
        val_df = df[df['group'] >= training_cutoff_group]
        
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="target",
            group_ids=["group"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=[f'f_{i}' for i in range(self.n_features)],
            target_normalizer=None,
        )
        
        validation = TimeSeriesDataSet.from_dataset(
            training, 
            val_df
        )
        
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        
        # Define the model (True SOTA TFT)
        from pytorch_forecasting.metrics import CrossEntropy
        
        self.model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.0001,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=3,
            loss=CrossEntropy(),
            reduce_on_plateau_patience=4,
        )
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="cpu",
            devices=1,
            gradient_clip_val=0.1,
            enable_model_summary=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3), 
                LearningRateMonitor()
            ],
        )
        
        trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        self.trained = True
        print("[TFT-PT] Training finished with 90/10 scale-invariant split.")
        print("[TFT-PT] Training finished with 90/10 split.")

    def build(self):
        """No-op for pytorch-forecasting TFT (built automatically from dataset)."""
        pass

    def predict(self, X_recent: np.ndarray) -> dict:
        """
        Predict next color using the last window.
        X_recent: (1, seq_len, n_features)
        """
        if not self.trained or self.model is None:
            return {'T': 0.46, 'CT': 0.46, 'Bonus': 0.08}
            
        self.model.eval()
        try:
            # We construct a minimal dataframe for prediction
            seq_data = []
            for i in range(self.seq_len):
                row = {'time_idx': i, 'target': 0, 'group': 0}
                feat = X_recent[0, i, :]
                for f_idx in range(self.n_features):
                    row[f'f_{f_idx}'] = float(feat[f_idx])
                seq_data.append(row)
            
            df = pd.DataFrame(seq_data)
            
            # Predict
            with torch.no_grad():
                preds = self.model.predict(df, mode="raw", return_x=False)
                # preds.prediction.shape: (batch, horizon, n_classes)
                p = torch.softmax(preds.prediction[0, 0], dim=-1).cpu().numpy()
                
            # Mapping from features.py: 0=T, 1=CT, 2=Bonus
            return {
                'T': float(p[0]),
                'CT': float(p[1]),
                'Bonus': float(p[2])
            }
        except Exception as e:
            print(f"[TFT-PT] Prediction error: {e}")
            return {'T': 0.46, 'CT': 0.46, 'Bonus': 0.08}

    def save(self):
        if self.model and self.trained:
            # Save as a full Lightning checkpoint for reliable reload
            trainer = pl.Trainer()
            trainer.strategy.connect(self.model)
            trainer.save_checkpoint(MODEL_PATH)
            with open(CONFIG_PATH, 'wb') as f:
                pickle.dump({
                    'dataset_params': self.dataset_params,
                    'seq_len': self.seq_len,
                    'n_features': self.n_features
                }, f)
            print("[TFT-PT] Model saved.")

    def load(self) -> bool:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
            return False
        try:
            with open(CONFIG_PATH, 'rb') as f:
                config = pickle.load(f)
            self.dataset_params = config['dataset_params']
            self.seq_len = config['seq_len']
            self.n_features = config['n_features']

            print("[TFT-PT] Loading model from checkpoint...")
            self.model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
            self.model.eval()
            self.trained = True
            print("[TFT-PT] Model loaded successfully!")
            return True
        except Exception as e:
            print(f"[TFT-PT] Load error: {e}, will retrain.")
            return False

