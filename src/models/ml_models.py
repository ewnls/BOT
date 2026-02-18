import os
import pickle
import numpy as np
from typing import Optional, Tuple
from numpy.lib.stride_tricks import sliding_window_view
from xgboost import XGBClassifier, XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import torch_directml
    DML_AVAILABLE = torch_directml.is_available()
except ImportError:
    DML_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Sélection automatique du GPU discret
# Le Ryzen 9800X3D expose un iGPU (index 0) → on préfère le RX 9070 XT
# ─────────────────────────────────────────────────────────────────────────────

def _select_dml_device() -> torch.device:
    """
    Parcourt tous les périphériques DirectML et retourne le GPU discret.
    Critère : exclut les noms contenant 'integrated', 'Microsoft Basic',
              'Intel', ou 'Radeon Graphics' (iGPU AMD générique).
    Fallback : dernier device disponible (généralement le discret).
    """
    if not DML_AVAILABLE:
        print("⚠️  torch_directml non disponible → CPU")
        return torch.device('cpu')

    n = torch_directml.device_count()
    print(f"🔍 Périphériques DirectML détectés ({n}):")

    igpu_keywords = ['integrated', 'microsoft basic', 'intel', 'radeon graphics']
    best_idx      = n - 1       # fallback = dernier device (souvent le discret)

    for i in range(n):
        name = torch_directml.device_name(i)
        flag = "← iGPU (ignoré)" if any(kw in name.lower() for kw in igpu_keywords) else "← discret ✅"
        print(f"  [{i}] {name}  {flag}")
        if flag.endswith("✅"):
            best_idx = i
            break

    selected_name = torch_directml.device_name(best_idx)
    print(f"\n🖥️  GPU sélectionné: [{best_idx}] {selected_name}\n")
    return torch_directml.device(best_idx)


DML_DEVICE = _select_dml_device()


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel:
    """
    Wrapper XGBoost:
      mode='class'       → XGBClassifier  (target_class)
      mode='reg'         → XGBRegressor   (target_return)
      mode='multioutput' → 2 XGBRegressor (target_return + atr)
    """

    def __init__(
        self,
        lookback: int,
        features: int,
        mode: str    = 'class',
        profile: str = 'aggressive',
        random_state: int = 42,
        n_jobs: int       = -1,
        **_,
    ):
        self.lookback     = lookback
        self.features     = features
        self.mode         = mode
        self.profile      = profile
        self.random_state = random_state
        self.n_jobs       = n_jobs

        self.model_class: Optional[XGBClassifier] = None
        self.model_reg:   Optional[XGBRegressor]  = None
        self.model_reg2:  Optional[XGBRegressor]  = None

    def _base_params(self) -> dict:
        return {
            'tree_method' : 'hist',
            'random_state': self.random_state,
            'n_jobs'      : self.n_jobs,
        }

    def _profile_params(self) -> dict:
        if self.profile == 'aggressive':
            return {
                'n_estimators'    : 600,
                'max_depth'       : 8,
                'learning_rate'   : 0.03,
                'subsample'       : 0.9,
                'colsample_bytree': 0.9,
                'min_child_weight': 1,
                'gamma'           : 0.0,
                'reg_lambda'      : 1.0,
                'max_bin'         : 512,   # profite du 3D V-Cache du 9800X3D
            }
        return {   # conservative
            'n_estimators'    : 300,
            'max_depth'       : 5,
            'learning_rate'   : 0.05,
            'subsample'       : 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma'           : 0.5,
            'reg_lambda'      : 2.0,
            'max_bin'         : 256,
        }

    def build_model(self):
        base = self._base_params()
        prof = self._profile_params()

        if self.mode == 'class':
            self.model_class = XGBClassifier(
                **base, **prof,
                objective='multi:softprob',
                num_class=4,
            )
        elif self.mode == 'reg':
            self.model_reg = XGBRegressor(
                **base, **prof,
                objective='reg:squarederror',
            )
        elif self.mode == 'multioutput':
            params = {**base, **prof, 'objective': 'reg:squarederror'}
            self.model_reg  = XGBRegressor(**params)
            self.model_reg2 = XGBRegressor(**params)
        else:
            raise ValueError(f"Mode XGBoost inconnu: '{self.mode}'")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None

        if self.mode == 'class':
            self.model_class.fit(
                X_train, y_train,
                eval_set=eval_set, verbose=False,
            )
        elif self.mode == 'reg':
            self.model_reg.fit(
                X_train, y_train.ravel(),
                eval_set=eval_set, verbose=False,
                early_stopping_rounds=30 if eval_set else None,
            )
        elif self.mode == 'multioutput':
            ev_ret = [(X_val, y_val[:, 0])] if X_val is not None else None
            ev_atr = [(X_val, y_val[:, 1])] if X_val is not None else None
            self.model_reg.fit(
                X_train, y_train[:, 0],
                eval_set=ev_ret, verbose=False,
            )
            self.model_reg2.fit(
                X_train, y_train[:, 1],
                eval_set=ev_atr, verbose=False,
            )

    def predict(self, X) -> np.ndarray:
        if self.mode == 'class':
            return self.model_class.predict_proba(X)
        elif self.mode == 'reg':
            return self.model_reg.predict(X).reshape(-1, 1)
        elif self.mode == 'multioutput':
            r = self.model_reg.predict(X).reshape(-1, 1)
            a = self.model_reg2.predict(X).reshape(-1, 1)
            return np.concatenate([r, a], axis=1)
        raise ValueError(f"Mode XGBoost inconnu: '{self.mode}'")

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        meta = {
            'mode'    : self.mode,   'profile' : self.profile,
            'lookback': self.lookback, 'features': self.features,
        }
        if self.mode == 'class':
            self.model_class.save_model(filepath + '_xgb_class.json')
        elif self.mode == 'reg':
            self.model_reg.save_model(filepath + '_xgb_reg.json')
        elif self.mode == 'multioutput':
            self.model_reg.save_model(filepath  + '_xgb_reg_return.json')
            self.model_reg2.save_model(filepath + '_xgb_reg_atr.json')
        with open(filepath + '_xgb_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

    def load(self, filepath: str):
        with open(filepath + '_xgb_meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        self.mode     = meta.get('mode',     'class')
        self.profile  = meta.get('profile',  'aggressive')
        self.lookback = meta.get('lookback', self.lookback)
        self.features = meta.get('features', self.features)
        self.build_model()
        if self.mode == 'class':
            self.model_class.load_model(filepath + '_xgb_class.json')
        elif self.mode == 'reg':
            self.model_reg.load_model(filepath + '_xgb_reg.json')
        elif self.mode == 'multioutput':
            self.model_reg.load_model(filepath  + '_xgb_reg_return.json')
            self.model_reg2.load_model(filepath + '_xgb_reg_atr.json')
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Mixin commun LSTM / Transformer (PyTorch + DirectML)
# ─────────────────────────────────────────────────────────────────────────────

class _TorchSequenceMixin:
    """Utilitaires partagés entre LSTMModel et TransformerModel."""

    def prepare_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforme X (N, F) → (N - lookback + 1, lookback, F).
        Vectorisé via sliding_window_view (pas de boucle Python).
        """
        if len(X) < self.lookback:
            raise ValueError(
                f"Pas assez de données: {len(X)} < lookback {self.lookback}"
            )
        X_seq = sliding_window_view(
            X, (self.lookback, X.shape[1])
        ).squeeze(1).copy()
        y_seq = y[self.lookback - 1:]
        return X_seq, y_seq

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).to(DML_DEVICE)

    def _train_loop(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   Optional[np.ndarray],
        y_val:   Optional[np.ndarray],
        epochs: int, batch_size: int,
    ):
        """Boucle d'entraînement avec EarlyStopping + ReduceLROnPlateau."""
        loader = DataLoader(
            TensorDataset(self._to_tensor(X_train), self._to_tensor(y_train)),
            batch_size=batch_size, shuffle=True,
            pin_memory=False,   # pin_memory non supporté par DirectML
        )

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6
        )
        criterion    = nn.MSELoss()
        best_val     = float('inf')
        patience_ctr = 0
        best_state   = None
        has_val      = X_val is not None and len(X_val) > 0

        for epoch in range(epochs):
            # ── Train ────────────────────────────────────────────────────────
            self.net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(self.net(xb), yb).backward()
                optimizer.step()

            # ── Validation ───────────────────────────────────────────────────
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    val_loss = criterion(
                        self.net(self._to_tensor(X_val)),
                        self._to_tensor(y_val)
                    ).item()

                scheduler.step(val_loss)

                if val_loss < best_val - 1e-6:
                    best_val     = val_loss
                    patience_ctr = 0
                    best_state   = {
                        k: v.cpu().clone()
                        for k, v in self.net.state_dict().items()
                    }
                else:
                    patience_ctr += 1
                    if patience_ctr >= 10:
                        break

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(
                {k: v.to(DML_DEVICE) for k, v in best_state.items()}
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = self.net(self._to_tensor(X))
        return out.cpu().numpy()

    def _save_meta(self, filepath: str, extra: Optional[dict] = None):
        meta = {
            'lookback'     : self.lookback,
            'features'     : self.features,
            'dropout'      : self.dropout,
            'learning_rate': self.learning_rate,
            **(extra or {}),
        }
        with open(filepath + '_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

    def _load_meta(self, filepath: str) -> dict:
        with open(filepath + '_meta.pkl', 'rb') as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM (PyTorch + DirectML)
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, features, hidden1, hidden2, dropout, output_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(features, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1,  hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden2, 32)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(32, output_dim)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x     = self.drop1(x)
        x, _  = self.lstm2(x)
        x     = self.drop2(x[:, -1, :])   # dernière sortie temporelle
        return self.fc2(self.relu(self.fc1(x)))


class LSTMModel(_TorchSequenceMixin):
    """LSTM entraîné sur GPU via torch_directml (RX 9070 XT)."""

    def __init__(
        self,
        lookback: int,
        features: int,
        dropout: float       = 0.2,
        learning_rate: float = 0.001,
        units: tuple         = (128, 64),
        **_,
    ):
        self.lookback      = lookback
        self.features      = features
        self.dropout       = dropout
        self.learning_rate = learning_rate
        self.units         = units
        self.net: Optional[_LSTMNet] = None

    def build_model(self, output_dim: int = 3):
        self.net = _LSTMNet(
            self.features, self.units[0], self.units[1],
            self.dropout, output_dim,
        ).to(DML_DEVICE)

    def train(
        self,
        X_train, y_train,
        X_val=None, y_val=None,
        epochs: int     = 50,
        batch_size: int = 256,
        **_,
    ):
        self._train_loop(X_train, y_train, X_val, y_val, epochs, batch_size)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        torch.save(self.net.state_dict(), filepath + '_lstm.pt')
        self._save_meta(filepath + '_lstm', {
            'units'     : list(self.units),
            'output_dim': self.net.fc2.out_features,
        })

    def load(self, filepath: str):
        meta = self._load_meta(filepath + '_lstm')
        self.lookback      = meta['lookback']
        self.features      = meta['features']
        self.dropout       = meta['dropout']
        self.learning_rate = meta['learning_rate']
        self.units         = tuple(meta.get('units', [128, 64]))
        self.build_model(output_dim=meta.get('output_dim', 3))
        self.net.load_state_dict(
            torch.load(filepath + '_lstm.pt', map_location=DML_DEVICE)
        )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER (PyTorch + DirectML)
# ─────────────────────────────────────────────────────────────────────────────

class _TransformerNet(nn.Module):
    def __init__(self, features, num_heads, ff_dim, dense_dim, dropout, output_dim):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.fc1     = nn.Linear(features, dense_dim)
        self.relu    = nn.ReLU()
        self.drop    = nn.Dropout(dropout)
        self.fc2     = nn.Linear(dense_dim, output_dim)

    def forward(self, x):                                    # x: (B, T, F)
        x = self.encoder(x)                                  # (B, T, F)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)        # (B, F)
        return self.fc2(self.drop(self.relu(self.fc1(x))))


class TransformerModel(_TorchSequenceMixin):
    """Transformer entraîné sur GPU via torch_directml (RX 9070 XT)."""

    def __init__(
        self,
        lookback: int,
        features: int,
        dropout: float       = 0.2,
        learning_rate: float = 0.001,
        num_heads: int       = 4,
        ff_dim: int          = 128,
        dense_dim: int       = 64,
        **_,
    ):
        self.lookback      = lookback
        self.features      = features
        self.dropout       = dropout
        self.learning_rate = learning_rate
        self.num_heads     = self._safe_heads(features, num_heads)
        self.ff_dim        = ff_dim
        self.dense_dim     = dense_dim
        self.net: Optional[_TransformerNet] = None

    @staticmethod
    def _safe_heads(features: int, num_heads: int) -> int:
        """
        Garantit que features % num_heads == 0.
        Descend num_heads jusqu'à trouver un diviseur valide.
        """
        while num_heads > 1 and features % num_heads != 0:
            num_heads -= 1
        return max(1, num_heads)

    def build_model(self, output_dim: int = 3):
        self.net = _TransformerNet(
            self.features, self.num_heads, self.ff_dim,
            self.dense_dim, self.dropout, output_dim,
        ).to(DML_DEVICE)

    def train(
        self,
        X_train, y_train,
        X_val=None, y_val=None,
        epochs: int     = 50,
        batch_size: int = 256,
        **_,
    ):
        self._train_loop(X_train, y_train, X_val, y_val, epochs, batch_size)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        torch.save(self.net.state_dict(), filepath + '_transformer.pt')
        self._save_meta(filepath + '_transformer', {
            'num_heads' : self.num_heads,
            'ff_dim'    : self.ff_dim,
            'dense_dim' : self.dense_dim,
            'output_dim': self.net.fc2.out_features,
        })

    def load(self, filepath: str):
        meta = self._load_meta(filepath + '_transformer')
        self.lookback      = meta['lookback']
        self.features      = meta['features']
        self.dropout       = meta['dropout']
        self.learning_rate = meta['learning_rate']
        self.num_heads     = meta.get('num_heads',  4)
        self.ff_dim        = meta.get('ff_dim',     128)
        self.dense_dim     = meta.get('dense_dim',  64)
        self.build_model(output_dim=meta.get('output_dim', 3))
        self.net.load_state_dict(
            torch.load(filepath + '_transformer.pt', map_location=DML_DEVICE)
        )
        return self
