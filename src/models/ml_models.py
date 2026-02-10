import os
import pickle
import numpy as np
from typing import Optional

from xgboost import XGBClassifier, XGBRegressor

# Imports TensorFlow/Keras pour LSTM et Transformer
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# =============================
#       XGBOOST MODEL
# =============================

class XGBoostModel:
    """
    Wrapper XGBoost pour le pipeline:
      - mode='class'       -> XGBClassifier sur target_class
      - mode='reg'         -> XGBRegressor sur target_return
      - mode='multioutput' -> 2 XGBRegressor (target_return, atr)
    Le paramètre 'profile' permet de choisir un set d'hyperparamètres.
    """

    def __init__(
        self,
        lookback: int,
        features: int,
        mode: str = "class",
        profile: str = "aggressive",
        random_state: int = 42,
        n_jobs: int = -1,
        **_
    ):
        self.lookback = lookback
        self.features = features
        self.mode = mode
        self.profile = profile
        self.random_state = random_state
        self.n_jobs = n_jobs

        # objets XGBoost (init dans build_model)
        self.model_class: Optional[XGBClassifier] = None
        self.model_reg: Optional[XGBRegressor] = None
        self.model_reg2: Optional[XGBRegressor] = None  # pour multioutput

    def _get_base_params(self):
        """Params communs à tous les profils."""
        return {
            "tree_method": "hist",
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def _get_profile_params(self):
        """
        Hyperparamètres selon le profil.
        'aggressive' = plus profond, plus de trees.
        'conservative' = plus régulier, moins de variance.
        """
        if self.profile == "aggressive":
            params = {
                "n_estimators": 600,
                "max_depth": 8,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 1,
                "gamma": 0.0,
                "reg_lambda": 1.0,
            }
        else:  # conservative
            params = {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.5,
                "reg_lambda": 2.0,
            }
        return params

    def build_model(self):
        base = self._get_base_params()
        prof = self._get_profile_params()

        if self.mode == "class":
            params = {
                **base,
                **prof,
                "objective": "multi:softprob",
                "num_class": 4,
            }
            self.model_class = XGBClassifier(**params)

        elif self.mode == "reg":
            params = {
                **base,
                **prof,
                "objective": "reg:squarederror",
            }
            self.model_reg = XGBRegressor(**params)

        elif self.mode == "multioutput":
            params1 = {
                **base,
                **prof,
                "objective": "reg:squarederror",
            }
            params2 = {
                **base,
                **prof,
                "objective": "reg:squarederror",
            }
            self.model_reg = XGBRegressor(**params1)
            self.model_reg2 = XGBRegressor(**params2)
        else:
            raise ValueError(f"Mode XGBoost inconnu: {self.mode}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.mode == "class":
            self.model_class.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                verbose=False
            )

        elif self.mode == "reg":
            self.model_reg.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                verbose=False
            )

        elif self.mode == "multioutput":
            y_return = y_train[:, 0]
            y_atr = y_train[:, 1]

            y_return_val = None
            y_atr_val = None
            if y_val is not None:
                y_return_val = y_val[:, 0]
                y_atr_val = y_val[:, 1]

            self.model_reg.fit(
                X_train,
                y_return,
                eval_set=[(X_val, y_return_val)] if X_val is not None else None,
                verbose=False
            )
            self.model_reg2.fit(
                X_train,
                y_atr,
                eval_set=[(X_val, y_atr_val)] if X_val is not None else None,
                verbose=False
            )
        else:
            raise ValueError(f"Mode XGBoost inconnu: {self.mode}")

    def predict(self, X):
        if self.mode == "class":
            proba = self.model_class.predict_proba(X)
            return proba

        elif self.mode == "reg":
            preds = self.model_reg.predict(X).reshape(-1, 1)
            return preds

        elif self.mode == "multioutput":
            preds_return = self.model_reg.predict(X).reshape(-1, 1)
            preds_atr = self.model_reg2.predict(X).reshape(-1, 1)
            return np.concatenate([preds_return, preds_atr], axis=1)

        else:
            raise ValueError(f"Mode XGBoost inconnu: {self.mode}")

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        meta = {
            "mode": self.mode,
            "profile": self.profile,
            "lookback": self.lookback,
            "features": self.features,
        }

        if self.mode == "class":
            self.model_class.save_model(filepath + "_xgb_class.json")

        elif self.mode == "reg":
            self.model_reg.save_model(filepath + "_xgb_reg.json")

        elif self.mode == "multioutput":
            self.model_reg.save_model(filepath + "_xgb_reg_return.json")
            self.model_reg2.save_model(filepath + "_xgb_reg_atr.json")

        with open(filepath + "_xgb_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    def load(self, filepath: str):
        with open(filepath + "_xgb_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        self.mode = meta.get("mode", "class")
        self.profile = meta.get("profile", "aggressive")
        self.lookback = meta.get("lookback", self.lookback)
        self.features = meta.get("features", self.features)

        self.build_model()

        if self.mode == "class":
            self.model_class.load_model(filepath + "_xgb_class.json")

        elif self.mode == "reg":
            self.model_reg.load_model(filepath + "_xgb_reg.json")

        elif self.mode == "multioutput":
            self.model_reg.load_model(filepath + "_xgb_reg_return.json")
            self.model_reg2.load_model(filepath + "_xgb_reg_atr.json")

        return self


# =============================
#         LSTM MODEL
# =============================

class LSTMModel:
    """
    Modèle LSTM pour prédire [target_class, target_return, atr].
    """

    def __init__(self, lookback: int, features: int, **_):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow n'est pas installé. Installez-le avec: pip install tensorflow")
        
        self.lookback = lookback
        self.features = features
        self.model = None

    def prepare_sequences(self, X, y):
        """Transforme X (N, features) en séquences (N - lookback + 1, lookback, features)"""
        if len(X) < self.lookback:
            raise ValueError(f"Pas assez de données: {len(X)} < lookback {self.lookback}")
        
        X_seq = []
        y_seq = []
        
        for i in range(self.lookback - 1, len(X)):
            X_seq.append(X[i - self.lookback + 1:i + 1])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)

    def build_model(self, output_dim=3):
        """Construit l'architecture LSTM"""
        self.model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.lookback, self.features)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim)  # 3 outputs: class, return, atr
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, **_):
        """Entraîne le modèle LSTM"""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

    def predict(self, X):
        """Prédit sur les séquences"""
        return self.model.predict(X, verbose=0)

    def save(self, filepath: str):
        """Sauvegarde le modèle"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath + "_lstm.h5")
        
        meta = {
            "lookback": self.lookback,
            "features": self.features,
        }
        with open(filepath + "_lstm_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    def load(self, filepath: str):
        """Charge le modèle"""
        self.model = keras.models.load_model(filepath + "_lstm.h5")
        
        with open(filepath + "_lstm_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        
        self.lookback = meta.get("lookback", self.lookback)
        self.features = meta.get("features", self.features)
        
        return self


# =============================
#     TRANSFORMER MODEL
# =============================

class TransformerModel:
    """
    Modèle Transformer pour prédire [target_class, target_return, atr].
    """

    def __init__(self, lookback: int, features: int, **_):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow n'est pas installé. Installez-le avec: pip install tensorflow")
        
        self.lookback = lookback
        self.features = features
        self.model = None

    def prepare_sequences(self, X, y):
        """Transforme X (N, features) en séquences (N - lookback + 1, lookback, features)"""
        if len(X) < self.lookback:
            raise ValueError(f"Pas assez de données: {len(X)} < lookback {self.lookback}")
        
        X_seq = []
        y_seq = []
        
        for i in range(self.lookback - 1, len(X)):
            X_seq.append(X[i - self.lookback + 1:i + 1])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        """Bloc encoder Transformer"""
        # Multi-head attention
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
        
        # Feed forward
        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(inputs.shape[-1])(ff)
        
        return layers.LayerNormalization(epsilon=1e-6)(x + ff)

    def build_model(self, output_dim=3):
        """Construit l'architecture Transformer"""
        inputs = keras.Input(shape=(self.lookback, self.features))
        
        # 2 blocs transformer
        x = self.transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128)
        x = self.transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)
        
        # Pooling et dense layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(output_dim)(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, **_):
        """Entraîne le modèle Transformer"""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

    def predict(self, X):
        """Prédit sur les séquences"""
        return self.model.predict(X, verbose=0)

    def save(self, filepath: str):
        """Sauvegarde le modèle"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath + "_transformer.h5")
        
        meta = {
            "lookback": self.lookback,
            "features": self.features,
        }
        with open(filepath + "_transformer_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    def load(self, filepath: str):
        """Charge le modèle"""
        self.model = keras.models.load_model(filepath + "_transformer.h5")
        
        with open(filepath + "_transformer_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        
        self.lookback = meta.get("lookback", self.lookback)
        self.features = meta.get("features", self.features)
        
        return self
