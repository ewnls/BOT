"""
Module de preprocessing et calcul d'indicateurs techniques
Supporte multi-timeframe et calcul automatique des features
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class DataPreprocessor:
    """Préprocessing des données OHLCV avec indicateurs techniques (version TA-Lib)"""

    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or ['1h']
        self.data = {}

    def load_csv(self, filepath: str, timeframe: str = '1h') -> pd.DataFrame:
        """Charge un fichier CSV avec données OHLCV"""
        df = pd.read_csv(filepath)
        
        # Standardisation des colonnes
        df.columns = [col.lower() for col in df.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV doit contenir: {required_cols}")
        
        # Conversion timestamp si présent
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df.set_index('date', inplace=True)
        
        # Supprime les lignes avec des valeurs manquantes
        df = df.dropna()
        
        self.data[timeframe] = df
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcul automatique de tous les indicateurs techniques (nécessite TA-Lib)"""
        try:
            import talib
        except ImportError:
            raise ImportError("TA-Lib requis pour DataPreprocessor. Utilisez DataPreprocessorSimple à la place.")
        
        # Détection volume = 0
        if df['volume'].sum() == 0:
            print(f"⚠️ Warning: Volume = 0, certains indicateurs seront limités")
        
        # Prix
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_ = df['open'].values
        volume = df['volume'].values

        # Moyennes mobiles
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['sma_200'] = talib.SMA(close, timeperiod=200)
        df['ema_9'] = talib.EMA(close, timeperiod=9)
        df['ema_21'] = talib.EMA(close, timeperiod=21)

        # RSI
        df['rsi_14'] = talib.RSI(close, timeperiod=14)

        # MACD
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle

        # ATR (Average True Range)
        df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

        # ADX (Average Directional Index)
        df['adx_14'] = talib.ADX(high, low, close, timeperiod=14)

        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd

        # Volume indicators (skip si volume = 0)
        if df['volume'].sum() > 0:
            df['obv'] = talib.OBV(close, volume)
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        else:
            df['obv'] = 0
            df['vwap'] = df['close']

        # Momentum
        df['momentum'] = talib.MOM(close, timeperiod=10)

        # CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(high, low, close, timeperiod=20)

        # Williams %R
        df['willr'] = talib.WILLR(high, low, close, timeperiod=14)

        # Patterns de prix
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()

        # Niveaux de support/résistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        return df

    def merge_multi_timeframe(self, primary_tf: str) -> pd.DataFrame:
        """Fusion des données multi-timeframe pour analyse croisée"""
        if primary_tf not in self.data:
            raise ValueError(f"Timeframe {primary_tf} non chargée")

        primary_df = self.data[primary_tf].copy()

        # Ajouter les features des autres timeframes
        for tf in self.timeframes:
            if tf != primary_tf and tf in self.data:
                tf_df = self.data[tf]

                # Resample sur la timeframe primaire
                for col in tf_df.columns:
                    if col not in ['open', 'high', 'low', 'close', 'volume']:
                        primary_df[f'{col}_{tf}'] = tf_df[col].reindex(primary_df.index, method='ffill')

        return primary_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prépare les features pour le ML (supprime NaN, normalise)"""
        # Supprime les NaN
        df_clean = df.dropna()

        # Liste des features (exclut OHLCV bruts)
        feature_cols = [col for col in df_clean.columns 
                        if col not in ['open', 'high', 'low', 'close', 'volume']]

        return df_clean, feature_cols

    def split_data(self, df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
        """Split train/validation/test par ratios"""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        return train, val, test

    def split_data_by_date(self, df: pd.DataFrame,
                           train_end: str,
                           val_end: str,
                           test_end: str):
        """
        Split train/val/test par dates (index datetime déjà mis en place dans load_csv).
        Les dates sont des strings 'YYYY-MM-DD'.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("L'index du DataFrame doit être un DatetimeIndex pour split_data_by_date.")

        # Force les bornes à être en UTC comme l'index
        train_end_ts = pd.to_datetime(train_end, utc=True)
        val_end_ts = pd.to_datetime(val_end, utc=True)
        test_end_ts = pd.to_datetime(test_end, utc=True)

        # Train: tout jusqu'à train_end inclus
        train = df.loc[:train_end_ts]

        # Val: (train_end, val_end]
        val = df.loc[(df.index > train_end_ts) & (df.index <= val_end_ts)]

        # Test: (val_end, test_end]
        test = df.loc[(df.index > val_end_ts) & (df.index <= test_end_ts)]

        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError(
                f"Split par date vide: train={len(train)}, val={len(val)}, test={len(test)}. "
                f"Vérifie les dates et la profondeur d'historique."
            )

        return train, val, test


class DataPreprocessorSimple(DataPreprocessor):
    """Version simplifiée sans TA-Lib, calculs manuels (RECOMMANDÉE)"""

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcul manuel des indicateurs (sans TA-Lib)"""
        
        # Détection volume = 0
        has_volume = df['volume'].sum() > 0
        if not has_volume:
            print(f"⚠️ Warning: Volume = 0, skip volume-based indicators")

        # SMA
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()

        # EMA
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)  # Évite division par zéro
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        range_14 = high_14 - low_14
        df['stoch_k'] = 100 * (df['close'] - low_14) / (range_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # Volume indicators (uniquement si volume > 0)
        if has_volume:
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (df['volume'] * typical_price).cumsum() / (df['volume'].cumsum() + 1e-10)
        else:
            df['obv'] = 0
            df['vwap'] = df['close']  # Fallback

        # Momentum
        df['momentum'] = df['close'].diff(10)

        # Price change
        df['price_change'] = df['close'].pct_change()
        
        if has_volume:
            df['volume_change'] = df['volume'].pct_change()
        else:
            df['volume_change'] = 0

        # Support/Resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # Remplace les inf/nan générés
        df = df.replace([np.inf, -np.inf], np.nan)

        return df


if __name__ == '__main__':
    # Test
    processor = DataPreprocessorSimple(timeframes=['1h', '4h'])
    print("✅ DataPreprocessor prêt !")
