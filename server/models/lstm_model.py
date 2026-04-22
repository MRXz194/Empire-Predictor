"""
lstm_model.py - Bidirectional LSTM model
Architecture: BiLSTM(128) → Dropout → BiLSTM(64) → Dense(32) → Dense(3, softmax)
"""
import os
import numpy as np

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'saved', 'lstm_weights.weights.h5')

# Lazy import to avoid slow startup
_model = None


def focal_loss(gamma=2.0, alpha=0.25):
    """Sigmoid Focal Loss to handle class imbalance (Item 2)."""
    import tensorflow as tf
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow((1.0 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed


def _build_model(seq_len: int, n_features: int):
    """Build the BiLSTM model architecture."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Input
    )

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')  # T / CT / Bonus
    ])

    # Item 2: Use Focal Loss instead of categorical_crossentropy
    model.compile(
        optimizer='adam',
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    return model


class LSTMPredictor:
    def __init__(self, seq_len: int = 60, n_features: int = 46):
        self.seq_len = seq_len
        self.n_features = n_features
        self.model = None
        self.trained = False

    def build(self):
        self.model = _build_model(self.seq_len, self.n_features)
        print(f"[LSTM] Model built: {self.model.count_params()} params")

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 64):
        """Train on prepared sequences. X: (n, seq_len, features), y: (n, 3)."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        if self.model is None:
            self.n_features = X.shape[2]
            self.build()

        # Stratified train/val split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.trained = True
        val_acc = max(history.history.get('val_accuracy', [0]))
        print(f"[LSTM] Training complete. Best val_accuracy: {val_acc:.4f}")
        return history

    def predict(self, X_seq: np.ndarray) -> dict:
        """
        Predict next color from a single sequence.
        X_seq: (seq_len, n_features) or (1, seq_len, n_features)
        Returns {T: float, CT: float, Bonus: float}
        """
        if not self.trained or self.model is None:
            return {'T': 7/15, 'CT': 7/15, 'Bonus': 1/15}

        if X_seq.ndim == 2:
            X_seq = X_seq[np.newaxis, ...]

        probs = self.model.predict(X_seq, verbose=0)[0]
        return {
            'T': float(probs[0]),
            'CT': float(probs[1]),
            'Bonus': float(probs[2])
        }

    def save(self, path: str = None):
        path = path or SAVE_PATH
        if self.model:
            self.model.save_weights(path)
            print(f"[LSTM] Weights saved to {path}")

    def load(self, path: str = None) -> bool:
        path = path or SAVE_PATH
        if not os.path.exists(path):
            return False
        if self.model is None:
            self.build()
        self.model.load_weights(path)
        self.trained = True
        print(f"[LSTM] Weights loaded from {path}")
        return True
