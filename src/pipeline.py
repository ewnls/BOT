"""
Pipeline principal pour entraîner et utiliser les modèles de trading
Avec Risk Management intégré + Targets dynamiques (ATR-based TP/SL)
"""

import numpy as np
import pandas as pd
import os
import json
import pickle

# FIX: import StandardScaler au niveau module (pas dans la fonction)
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

from src.utils.preprocessing import DataPreprocessorSimple
from src.utils.backtesting import Backtester
from src.utils.risk_manager import AGGRESSIVE_CONFIG, CONSERVATIVE_CONFIG
from src.models.ml_models import XGBoostModel, LSTMModel, TransformerModel


class TradingPipeline:
    """Pipeline complet pour ML trading."""

    def __init__(self, config_path='config.json'):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.config = {
                "data_path"  : "data/",
                "model_path" : "models/",
                "lookback"   : 60,
                "features"   : 23,
                "epochs"     : 100,
                "batch_size" : 32,
                "test_split" : 0.15,
                "val_split"  : 0.15,
            }

        # FIX: initialisation explicite des attributs pour éviter AttributeError
        self.model             = None
        self.scaler_X          = None
        self.scaler_y          = None
        self.test_prices       = None
        self.primary_timeframe = None
        self.preprocessor      = None

    # ─────────────────────────────────────────────────────────────────────────
    # Chargement des données
    # ─────────────────────────────────────────────────────────────────────────

    def load_data(self, csv_files: dict, timeframes: list = None):
        """Charge les données depuis plusieurs CSV."""
        if timeframes is None:
            timeframes = list(csv_files.keys())

        self.preprocessor = DataPreprocessorSimple(timeframes=timeframes)

        for tf, filepath in csv_files.items():
            df = self.preprocessor.load_csv(filepath, timeframe=tf)
            df = self.preprocessor.calculate_indicators(df)
            self.preprocessor.data[tf] = df

            if not hasattr(self.preprocessor, 'symbol'):
                filename = os.path.basename(filepath)
                self.preprocessor.symbol = (
                    filename.split('_')[0] if '_' in filename
                    else filename.split('.')[0]
                )

        return self.preprocessor

    # ─────────────────────────────────────────────────────────────────────────
    # ATR
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule l'ATR (Average True Range)."""
        high  = df['high']
        low   = df['low']
        close = df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # Targets dynamiques — version vectorisée (sliding_window_view)
    # FIX: remplace la double boucle Python O(n × horizon) → ~10× plus rapide
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_dynamic_targets(self, df: pd.DataFrame, feature_cols,
                                 vol_window: int = 14, k_tp1: float = 1.0,
                                 k_tp2: float = 2.0, k_sl: float = 1.0,
                                 horizon_bars: int = 20) -> pd.DataFrame:
        """
        Calcule les targets dynamiques basées sur l'ATR:
          target_class  : 0=SL, 1=TP1, 2=TP2, 3=timeout
          target_return : variation % entre entry et exit réel
        """
        atr = self._calculate_atr(df, vol_window)
        df  = df.copy()
        df['atr'] = atr

        n           = len(df)
        n_valid     = n - horizon_bars
        highs       = df['high'].values
        lows        = df['low'].values
        closes      = df['close'].values
        entry_price = closes.copy()
        atr_vals    = atr.values

        tp1_level = entry_price + k_tp1 * atr_vals
        tp2_level = entry_price + k_tp2 * atr_vals
        sl_level  = entry_price - k_sl  * atr_vals

        target_class  = np.full(n, 3, dtype=int)
        target_return = np.zeros(n, dtype=float)

        if n_valid <= 0:
            return pd.DataFrame({
                'target_class' : target_class,
                'target_return': target_return,
                'atr'          : atr_vals,
            }, index=df.index)

        # Fenêtres glissantes de bars futurs  (shape: n_valid × horizon_bars)
        # sliding_window_view(arr, W) → shape (n-W+1, W) ; on prend [:, 1:] pour exclure bar i
        high_w = sliding_window_view(highs, horizon_bars + 1)[:n_valid, 1:]
        low_w  = sliding_window_view(lows,  horizon_bars + 1)[:n_valid, 1:]

        # Matrices booléennes de hit  (n_valid × horizon_bars)
        tp2_hit = high_w >= tp2_level[:n_valid, None]
        tp1_hit = high_w >= tp1_level[:n_valid, None]
        sl_hit  = low_w  <= sl_level[:n_valid, None]

        def _first_hit(mat: np.ndarray) -> np.ndarray:
            """Index de la première occurrence True par ligne (horizon_bars si aucune)."""
            result = np.full(mat.shape[0], horizon_bars, dtype=int)
            rows, cols = np.where(mat)
            if len(rows):
                unique_rows, idx = np.unique(rows, return_index=True)
                result[unique_rows] = cols[idx]
            return result

        f_tp2 = _first_hit(tp2_hit)
        f_tp1 = _first_hit(tp1_hit)
        f_sl  = _first_hit(sl_hit)

        # Règles de priorité (identiques à la boucle originale)
        # TP2 gagne si atteint avant ou en même temps que TP1 et SL
        is_tp2 = f_tp2 <= np.minimum(f_tp1, f_sl)
        # TP1 gagne si avant SL (strict), et TP2 non atteint
        is_tp1 = (~is_tp2) & (f_tp1 < f_sl)  & (f_tp1 < horizon_bars)
        # SL gagne dans tous les autres cas où il est atteint
        is_sl  = (~is_tp2) & (~is_tp1) & (f_sl < horizon_bars)

        target_class[:n_valid] = np.where(is_tp2, 2,
                                  np.where(is_tp1, 1,
                                  np.where(is_sl,  0, 3)))

        # Prix de sortie réels
        exit_prices = closes[horizon_bars:horizon_bars + n_valid].copy()
        exit_prices[is_tp2] = tp2_level[:n_valid][is_tp2]
        exit_prices[is_tp1] = tp1_level[:n_valid][is_tp1]
        exit_prices[is_sl]  = sl_level[:n_valid][is_sl]

        e = entry_price[:n_valid]
        target_return[:n_valid] = np.where(e != 0, (exit_prices - e) / e, 0.0)

        return pd.DataFrame({
            'target_class' : target_class,
            'target_return': target_return,
            'atr'          : atr_vals,
        }, index=df.index)

    # ─────────────────────────────────────────────────────────────────────────
    # Préparation des données d'entraînement
    # ─────────────────────────────────────────────────────────────────────────

    def prepare_training_data(self, primary_timeframe=None, multi_tf=False,
                              train_end=None, val_end=None, test_end=None,
                              vol_window=14, k_tp1=1.0, k_tp2=2.0, k_sl=1.0,
                              horizon_bars=20):
        """Prépare les données avec targets dynamiques ATR."""
        if primary_timeframe is None:
            primary_timeframe = list(self.preprocessor.data.keys())[0]
        self.primary_timeframe = primary_timeframe

        df = (self.preprocessor.merge_multi_timeframe(primary_timeframe)
              if multi_tf and len(self.preprocessor.timeframes) > 1
              else self.preprocessor.data[primary_timeframe])

        df_clean, feature_cols = self.preprocessor.prepare_features(df)

        df_targets = self._compute_dynamic_targets(
            df_clean, feature_cols,
            vol_window=vol_window, k_tp1=k_tp1, k_tp2=k_tp2,
            k_sl=k_sl, horizon_bars=horizon_bars,
        )

        df_clean = df_clean.join(df_targets, how='inner')
        df_clean = (df_clean
                    .dropna()
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna())

        X = df_clean[feature_cols].values
        y = df_clean[['target_class', 'target_return', 'atr']].values

        if not np.isfinite(X).all():
            raise ValueError("X contient des valeurs non finies après nettoyage")
        if not np.isfinite(y).all():
            raise ValueError("y contient des valeurs non finies après nettoyage")

        # FIX: StandardScaler déjà importé en haut du fichier
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X).astype(np.float32)

        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y).astype(np.float32)

        xy_df      = pd.DataFrame(
            np.concatenate([X_scaled, y_scaled], axis=1),
            index=df_clean.index
        )
        n_features = X_scaled.shape[1]

        if train_end is not None and val_end is not None and test_end is not None:
            train_df, val_df, test_df = self.preprocessor.split_data_by_date(
                xy_df, train_end=train_end, val_end=val_end, test_end=test_end
            )
        else:
            train_df, val_df, test_df = self.preprocessor.split_data(
                xy_df, train_ratio=0.70, val_ratio=0.15
            )

        X_train = train_df.iloc[:, :n_features].values
        y_train = train_df.iloc[:, n_features:].values
        X_val   = val_df.iloc[:, :n_features].values
        y_val   = val_df.iloc[:, n_features:].values
        X_test  = test_df.iloc[:, :n_features].values
        y_test  = test_df.iloc[:, n_features:].values

        self.test_prices = df_clean.loc[test_df.index][['open', 'high', 'low', 'close']]

        return X_train, y_train, X_val, y_val, X_test, y_test, n_features

    # ─────────────────────────────────────────────────────────────────────────
    # Entraînement
    # ─────────────────────────────────────────────────────────────────────────

    def train_model(self, model_type='lstm', X_train=None, y_train=None,
                    X_val=None, y_val=None, n_features=30, lookback=60,
                    dropout=0.2, learning_rate=0.001, **kwargs):
        """Entraîne le modèle sélectionné."""

        if model_type in ('lstm', 'transformer'):
            ModelClass = LSTMModel if model_type == 'lstm' else TransformerModel
            self.model = ModelClass(
                lookback=lookback, features=n_features,
                dropout=dropout, learning_rate=learning_rate,
            )
            X_train_seq, y_train_seq = self.model.prepare_sequences(X_train, y_train)
            X_val_seq,   y_val_seq   = (self.model.prepare_sequences(X_val, y_val)
                                         if X_val is not None and len(X_val) > 0
                                         else (None, None))
            self.model.build_model(output_dim=y_train.shape[1])
            self.model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, **kwargs)

        elif model_type == 'xgboost':
            self.model = XGBoostModel(
                lookback=lookback, features=n_features,
                mode='reg', profile='aggressive',
            )
            # XGBoost prédit uniquement target_return (colonne 1)
            y_train_ret = y_train[:, 1:2]
            # FIX: y_val peut être None si X_val a déjà été fusionné dans X_train
            y_val_ret   = (y_val[:, 1:2]
                           if y_val is not None and len(y_val) > 0
                           else None)
            x_val_pass  = X_val if y_val_ret is not None else None

            if X_train.shape[0] == 0:
                raise ValueError("Aucun sample pour XGBoost (train) – vérifie le split")

            self.model.build_model()
            self.model.train(X_train, y_train_ret, x_val_pass, y_val_ret)

        else:
            raise ValueError(f"model_type inconnu: '{model_type}'")

    # ─────────────────────────────────────────────────────────────────────────
    # Backtesting
    # ─────────────────────────────────────────────────────────────────────────

    def backtest(self, X_test, y_test, initial_capital=10000, asset_type='crypto'):
        """Effectue le backtesting."""
        # FIX: vérification explicite que le modèle est entraîné
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné. Appelez train_model() d'abord.")

        risk_config = (AGGRESSIVE_CONFIG
                       if asset_type in ('crypto', 'crypto_binance')
                       else CONSERVATIVE_CONFIG)

        # Prédictions normalisées
        if hasattr(self.model, 'prepare_sequences'):
            X_seq, _ = self.model.prepare_sequences(X_test, y_test)
            preds_norm = self.model.predict(X_seq)
        else:
            preds_norm = self.model.predict(X_test)

        # ── CAS LSTM / TRANSFORMER (output 3 colonnes) ──────────────────────
        if preds_norm.ndim == 2 and preds_norm.shape[1] == 3:
            predictions = self.scaler_y.inverse_transform(preds_norm)

        # ── CAS XGBOOST (output 1 colonne : target_return) ──────────────────
        elif preds_norm.ndim == 1 or preds_norm.shape[1] == 1:
            preds_ret_norm = preds_norm.reshape(-1, 1)
            n_preds        = len(preds_ret_norm)

            # FIX: utiliser le vrai ATR de y_test au lieu de dummy = 1
            # FIX: utiliser la vraie target_class de y_test au lieu de toujours 2 (TP2)
            y_test_aligned = y_test[-n_preds:]
            y_concat_norm  = np.concatenate([
                y_test_aligned[:, 0:1],  # vraie target_class normalisée
                preds_ret_norm,           # retour prédit par XGBoost
                y_test_aligned[:, 2:3],  # vrai ATR normalisé
            ], axis=1)

            y_concat           = self.scaler_y.inverse_transform(y_concat_norm)
            target_return_real = y_concat[:, 1]
            real_atr           = y_concat[:, 2]

            # FIX: classe dérivée du signe du retour prédit (pas toujours TP2)
            pred_class = np.where(target_return_real > 0, 2.0, 0.0)

            # Filtre de confiance : top 30% des signaux
            abs_ret   = np.abs(target_return_real)
            threshold = np.percentile(abs_ret, 70)
            mask      = abs_ret >= threshold
            if not np.any(mask):
                mask = np.ones(len(target_return_real), dtype=bool)

            predictions = np.column_stack([
                pred_class[mask],
                target_return_real[mask],
                real_atr[mask],
            ])
            prices_aligned   = self.test_prices.iloc[-n_preds:]
            self.test_prices = prices_aligned.iloc[mask]

        else:
            raise ValueError(
                f"Dimension des prédictions inattendue: {preds_norm.shape}"
            )

        backtester = Backtester(
            initial_capital=initial_capital,
            risk_config=risk_config,
            timeframe=self.primary_timeframe,
        )

        backtester.execute_trades(predictions, self.test_prices)
        backtester.print_report()

        return (
            backtester.calculate_metrics(),
            backtester.get_trades_dataframe(),
            backtester.get_equity_curve(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Sauvegarde / Chargement
    # ─────────────────────────────────────────────────────────────────────────

    def save_model(self, filepath='models/trained_model'):
        """Sauvegarde le modèle et ses scalers."""
        if self.model is None:
            raise RuntimeError("Aucun modèle à sauvegarder.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        with open(filepath + '_scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open(filepath + '_scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)
        return filepath

    def load_model(self, filepath, model_type='lstm', lookback=60, features=30):
        """Charge un modèle sauvegardé."""
        model_map = {
            'lstm'       : lambda: LSTMModel(lookback=lookback, features=features),
            'transformer': lambda: TransformerModel(lookback=lookback, features=features),
            'xgboost'    : lambda: XGBoostModel(lookback=lookback, features=features, mode='reg'),
        }
        if model_type not in model_map:
            raise ValueError(f"model_type inconnu: '{model_type}'")

        self.model = model_map[model_type]()
        self.model.load(filepath)

        with open(filepath + '_scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(filepath + '_scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)

        return self.model, self.scaler_X, self.scaler_y


if __name__ == '__main__':
    print("Pipeline de trading ML prêt !")
