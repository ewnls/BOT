"""
Pipeline principal pour entraîner et utiliser les modèles de trading
Avec Risk Management intégré
"""
import numpy as np
import pandas as pd
import os
import json

from src.utils.preprocessing import DataPreprocessorSimple
from src.utils.backtesting import Backtester
from src.utils.risk_manager import AGGRESSIVE_CONFIG, CONSERVATIVE_CONFIG



class TradingPipeline:
    """Pipeline complet pour ML trading"""

    def __init__(self, config_path='config.json'):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Config par défaut si fichier absent/vide
            self.config = {
                "data_path": "data/",
                "model_path": "models/",
                "lookback": 60,
                "features": 23,
                "epochs": 100,
                "batch_size": 32,
                "test_split": 0.15,
                "val_split": 0.15
            }

    def load_data(self, csv_files: dict, timeframes: list = None):
        """Charge les données depuis plusieurs CSV"""
        if timeframes is None:
            timeframes = list(csv_files.keys())

        self.preprocessor = DataPreprocessorSimple(timeframes=timeframes)

        for tf, filepath in csv_files.items():
            df = self.preprocessor.load_csv(filepath, timeframe=tf)
            df = self.preprocessor.calculate_indicators(df)
            self.preprocessor.data[tf] = df

            # Essaie de déduire le symbole depuis le nom de fichier (une seule fois)
            if not hasattr(self.preprocessor, "symbol"):
                filename = os.path.basename(filepath)
                if "_" in filename:
                    self.preprocessor.symbol = filename.split("_")[0]
                else:
                    self.preprocessor.symbol = filename.split(".")[0]

        return self.preprocessor

    def prepare_training_data(self, primary_timeframe=None, multi_tf=False,
                              train_end=None, val_end=None, test_end=None):
        """Prépare les données pour l'entraînement"""

        # Détecte automatiquement la timeframe si non spécifiée
        if primary_timeframe is None:
            primary_timeframe = list(self.preprocessor.data.keys())[0]

        self.primary_timeframe = primary_timeframe

        if multi_tf and len(self.preprocessor.timeframes) > 1:
            df = self.preprocessor.merge_multi_timeframe(primary_timeframe)
        else:
            df = self.preprocessor.data[primary_timeframe]

        df_clean, feature_cols = self.preprocessor.prepare_features(df)

        # TARGETS EN VARIATIONS % (pas prix absolus)
        future_price = df_clean['close'].shift(-1)
        current_price = df_clean['close']

        df_clean['target_entry'] = (future_price - current_price) / current_price
        df_clean['target_tp1'] = 0.015   # +1.5%
        df_clean['target_tp2'] = 0.030   # +3.0%
        df_clean['target_sl'] = -0.012   # -1.2%

        df_clean = df_clean.dropna()

        # Remplace infinity par NaN puis supprime
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()

        X = df_clean[feature_cols].values
        y = df_clean[['target_entry', 'target_tp1', 'target_tp2', 'target_sl']].values

        # Vérifie qu'il n'y a pas d'infinity
        if not np.isfinite(X).all():
            raise ValueError("X contient des valeurs infinies après nettoyage")

        if not np.isfinite(y).all():
            raise ValueError("y contient des valeurs infinies après nettoyage")

        # NORMALISATION de X ET y
        from sklearn.preprocessing import StandardScaler

        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)

        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y)

        # DataFrame aligné sur l'index temps
        xy_df = pd.DataFrame(
            np.concatenate([X_scaled, y_scaled], axis=1),
            index=df_clean.index
        )

        n_features = X_scaled.shape[1]

        # Split par dates si demandé, sinon par ratios
        if train_end is not None and val_end is not None and test_end is not None:
            train_df, val_df, test_df = self.preprocessor.split_data_by_date(
                xy_df,
                train_end=train_end,
                val_end=val_end,
                test_end=test_end
            )
        else:
            train_df, val_df, test_df = self.preprocessor.split_data(
                xy_df,
                train_ratio=0.70,
                val_ratio=0.15
            )

        X_train = train_df.iloc[:, :n_features].values
        y_train = train_df.iloc[:, n_features:].values

        X_val = val_df.iloc[:, :n_features].values
        y_val = val_df.iloc[:, n_features:].values

        X_test = test_df.iloc[:, :n_features].values
        y_test = test_df.iloc[:, n_features:].values

        # Prix pour le backtesting alignés sur la partie test
        self.test_prices = df_clean.loc[test_df.index][['open', 'high', 'low', 'close']]

        return X_train, y_train, X_val, y_val, X_test, y_test, n_features

    def train_model(self, model_type='lstm', X_train=None, y_train=None,
                    X_val=None, y_val=None, n_features=30, lookback=60, **kwargs):
        """Entraîne le modèle sélectionné"""
        from src.models.ml_models import LSTMModel, TransformerModel, XGBoostModel

        if model_type == 'lstm':
            self.model = LSTMModel(lookback=lookback, features=n_features)
            X_train_seq, y_train_seq = self.model.prepare_sequences(X_train, y_train)
            X_val_seq, y_val_seq = self.model.prepare_sequences(X_val, y_val)

            self.model.build_model(output_dim=y_train.shape[1])
            self.model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, **kwargs)

        elif model_type == 'transformer':
            self.model = TransformerModel(lookback=lookback, features=n_features)
            X_train_seq, y_train_seq = self.model.prepare_sequences(X_train, y_train)
            X_val_seq, y_val_seq = self.model.prepare_sequences(X_val, y_val)

            self.model.build_model(output_dim=y_train.shape[1])
            self.model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, **kwargs)

        elif model_type == 'xgboost':
            self.model = XGBoostModel(lookback=lookback, features=n_features)
            self.model.build_model()
            self.model.train(X_train, y_train, X_val, y_val)

    def backtest(self, X_test, y_test, initial_capital=10000, asset_type='crypto'):
        """Effectue le backtesting avec Risk Management adapté"""

        if asset_type in ['crypto', 'crypto_binance']:
            risk_config = AGGRESSIVE_CONFIG
        else:
            risk_config = CONSERVATIVE_CONFIG

        # Prédictions (normalisées)
        if hasattr(self.model, 'prepare_sequences'):
            X_test_seq, _ = self.model.prepare_sequences(X_test, y_test)
            predictions_normalized = self.model.predict(X_test_seq)
        else:
            predictions_normalized = self.model.predict(X_test)

        # DÉNORMALISATION des prédictions
        predictions = self.scaler_y.inverse_transform(predictions_normalized)

        backtester = Backtester(
            initial_capital=initial_capital,
            risk_config=risk_config,
            timeframe=self.primary_timeframe
        )

        trades, _ = backtester.execute_trades(predictions, self.test_prices)
        equity_series = backtester.get_equity_curve()

        backtester.print_report()

        return backtester.calculate_metrics(), backtester.get_trades_dataframe(), equity_series

    def save_model(self, filepath='models/trained_model'):
        """Sauvegarde le modèle"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.model.save(filepath)

        import pickle
        with open(filepath + '_scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)

        with open(filepath + '_scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)

    def load_model(self, filepath, model_type='lstm', lookback=60, features=30):
        """Charge un modèle sauvegardé"""
        from src.models.ml_models import LSTMModel, TransformerModel, XGBoostModel


        if model_type == 'lstm':
            self.model = LSTMModel(lookback=lookback, features=features)
        elif model_type == 'transformer':
            self.model = TransformerModel(lookback=lookback, features=features)
        elif model_type == 'xgboost':
            self.model = XGBoostModel(lookback=lookback, features=features)

        self.model.load(filepath)

        import pickle
        with open(filepath + '_scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)

        with open(filepath + '_scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)

        return self.model, self.scaler_X, self.scaler_y


if __name__ == '__main__':
    print("Pipeline de trading ML prêt!")
