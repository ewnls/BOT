"""
Module de preprocessing et calcul d'indicateurs techniques
Supporte multi-timeframe et calcul automatique des features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Classe de base commune (FIX: évite la duplication de code)
# ─────────────────────────────────────────────────────────────────────────────

class _DataPreprocessorBase:
    """Méthodes communes à toutes les variantes du préprocesseur."""

    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or ['1h']
        self.data: dict = {}

    def load_csv(self, filepath: str, timeframe: str = '1h') -> pd.DataFrame:
        try:
            # Lecture rapide via PyArrow
            df = pd.read_csv(filepath, engine='pyarrow')
        except Exception:
            # Fallback si pyarrow pas installé
            df = pd.read_csv(filepath)

        df.columns = [c.lower() for c in df.columns]
        """Charge un fichier CSV avec données OHLCV."""
        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]

        required = ['open', 'high', 'low', 'close', 'volume']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

        # Détection et parsing de la colonne de temps
        for col in ('timestamp', 'date', 'time', 'open_time'):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
                df.set_index(col, inplace=True)
                break

        df = df.dropna(subset=required)

        # FIX: garde-fou sur la longueur minimale
        if len(df) < 60:
            raise ValueError(
                f"Trop peu de données ({len(df)} lignes) dans '{filepath}'"
            )

        self.data[timeframe] = df
        return df

    def merge_multi_timeframe(self, primary_tf: str) -> pd.DataFrame:
        """Fusion des données multi-timeframe pour analyse croisée."""
        if primary_tf not in self.data:
            raise ValueError(f"Timeframe '{primary_tf}' non chargée")

        primary_df = self.data[primary_tf].copy()
        for tf in self.timeframes:
            if tf == primary_tf or tf not in self.data:
                continue
            tf_df = self.data[tf]
            for col in tf_df.columns:
                if col not in ('open', 'high', 'low', 'close', 'volume'):
                    primary_df[f'{col}_{tf}'] = (
                        tf_df[col].reindex(primary_df.index, method='ffill')
                    )
        return primary_df

    def prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prépare les features pour le ML.
        FIX: supprime les colonnes entièrement NaN (ex: sma_200 sur dataset court).
        """
        df_clean = df.dropna()

        base_cols = {'open', 'high', 'low', 'close', 'volume'}
        feature_cols = [
            col for col in df_clean.columns
            if col not in base_cols
        ]

        # FIX: retire les features 100% NaN (silencieux dans l'original)
        all_nan = [c for c in feature_cols if df_clean[c].isna().all()]
        if all_nan:
            print(f"⚠️ Features retirées (toutes NaN): {all_nan}")
            feature_cols = [c for c in feature_cols if c not in all_nan]

        df_clean = df_clean[list(base_cols & set(df_clean.columns)) + feature_cols]
        df_clean = df_clean.dropna(subset=feature_cols)

        return df_clean, feature_cols

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float   = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split train/validation/test par ratios."""
        n         = len(df)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    def split_data_by_date(
        self,
        df: pd.DataFrame,
        train_end: str,
        val_end: str,
        test_end: Optional[str] = None,   # FIX: optionnel, défaut = fin du dataset
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split train/val/test par dates.
        FIX: test_end optionnel (prend la fin du dataset si non fourni).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("L'index doit être un DatetimeIndex (voir load_csv).")

        train_end_ts = pd.to_datetime(train_end, utc=True)
        val_end_ts   = pd.to_datetime(val_end,   utc=True)
        test_end_ts  = (pd.to_datetime(test_end, utc=True)
                        if test_end else df.index.max())

        train = df.loc[df.index <= train_end_ts]
        val   = df.loc[(df.index > train_end_ts) & (df.index <= val_end_ts)]
        test  = df.loc[(df.index > val_end_ts)   & (df.index <= test_end_ts)]

        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError(
                f"Split par date vide → train={len(train)}, "
                f"val={len(val)}, test={len(test)}. "
                f"Vérifie les dates et la profondeur d'historique."
            )
        return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Version TA-Lib (optionnelle)
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessor(_DataPreprocessorBase):
    """Préprocessing avec TA-Lib (optionnel)."""

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import talib
        except ImportError:
            raise ImportError(
                "TA-Lib non installé. Utilisez DataPreprocessorSimple."
            )

        has_volume = df['volume'].sum() > 0
        if not has_volume:
            print("⚠️ Volume = 0, indicateurs volume désactivés")

        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        v = df['volume'].values

        df['sma_20']  = talib.SMA(c, 20)
        df['sma_50']  = talib.SMA(c, 50)
        # FIX: min_periods implicite via TA-Lib (NaN pour bars < 200)
        df['sma_200'] = talib.SMA(c, 200)
        df['ema_9']   = talib.EMA(c, 9)
        df['ema_21']  = talib.EMA(c, 21)
        df['rsi_14']  = talib.RSI(c, 14)

        macd, sig, hist   = talib.MACD(c, 12, 26, 9)
        df['macd']        = macd
        df['macd_signal'] = sig
        df['macd_hist']   = hist

        up, mid, lo       = talib.BBANDS(c, 20)
        df['bb_upper']    = up
        df['bb_middle']   = mid
        df['bb_lower']    = lo
        df['bb_width']    = (up - lo) / (mid + 1e-10)

        df['atr_14']      = talib.ATR(h, l, c, 14)
        df['adx_14']      = talib.ADX(h, l, c, 14)
        df['cci']         = talib.CCI(h, l, c, 20)
        df['willr']       = talib.WILLR(h, l, c, 14)

        sk, sd            = talib.STOCH(h, l, c)
        df['stoch_k']     = sk
        df['stoch_d']     = sd

        if has_volume:
            df['obv']  = talib.OBV(c, v)
            tp         = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (df['volume'] * tp).rolling(24).sum() / (
                df['volume'].rolling(24).sum() + 1e-10
            )
        else:
            df['obv']  = 0.0
            df['vwap'] = df['close']

        df['momentum']      = talib.MOM(c, 10)
        df['price_change']  = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change() if has_volume else 0.0
        df['resistance']    = df['high'].rolling(20).max()
        df['support']       = df['low'].rolling(20).min()

        df = df.replace([np.inf, -np.inf], np.nan)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Version Simple — sans TA-Lib (RECOMMANDÉE)
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessorSimple(_DataPreprocessorBase):
    """Calcul manuel des indicateurs, sans dépendance TA-Lib."""

    # ── Indicateurs ──────────────────────────────────────────────────────────

    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        ADX manuel (Average Directional Index).
        FIX: indicateur présent dans DataPreprocessor (TA-Lib) mais absent
             de l'original DataPreprocessorSimple.
        """
        high  = df['high']
        low   = df['low']
        close = df['close']

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        up_move   =  high.diff()
        down_move = -low.diff()

        plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr_s      = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di    = pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean() / (atr_s + 1e-10) * 100
        minus_di   = pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr_s + 1e-10) * 100

        dx  = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcul manuel de tous les indicateurs techniques."""
        has_volume = df['volume'].sum() > 0
        if not has_volume:
            print("⚠️ Volume = 0, indicateurs volume désactivés")

        close = df['close']
        high  = df['high']
        low   = df['low']

        # ── Moyennes mobiles ─────────────────────────────────────────────────
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        # FIX: min_periods=20 pour éviter 200 NaN sur dataset court
        df['sma_200'] = close.rolling(200, min_periods=20).mean()
        df['ema_9']   = close.ewm(span=9,  adjust=False).mean()
        df['ema_21']  = close.ewm(span=21, adjust=False).mean()

        # ── RSI ──────────────────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # ── MACD ─────────────────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd']        = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist']   = df['macd'] - df['macd_signal']

        # ── Bollinger Bands ──────────────────────────────────────────────────
        bb_mid          = close.rolling(20).mean()
        bb_std          = close.rolling(20).std()
        df['bb_middle'] = bb_mid
        df['bb_upper']  = bb_mid + bb_std * 2
        df['bb_lower']  = bb_mid - bb_std * 2
        df['bb_width']  = (df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-10)

        # ── ATR ──────────────────────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()

        # ── ADX ──────────────────────────────────────────────────────────────
        # FIX: ajout de l'ADX (présent dans DataPreprocessor mais pas dans l'original Simple)
        df['adx_14'] = self._calc_adx(df, period=14)

        # ── Stochastique ─────────────────────────────────────────────────────
        low_14  = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ── CCI ──────────────────────────────────────────────────────────────
        # FIX: ajout du CCI (présent dans DataPreprocessor mais pas dans l'original Simple)
        tp  = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad    = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-10)

        # ── Williams %R ──────────────────────────────────────────────────────
        # FIX: ajout du Williams %R (présent dans DataPreprocessor mais pas dans l'original Simple)
        df['willr'] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)

        # ── Volume ───────────────────────────────────────────────────────────
        if has_volume:
            df['obv'] = (
                np.sign(close.diff()) * df['volume']
            ).fillna(0).cumsum()

            # FIX: VWAP sur fenêtre glissante (original = cumulatif sur tout
            # le dataset → biais croissant et reset journalier absent)
            tp_vol = (tp * df['volume']).rolling(24).sum()
            vol_24 = df['volume'].rolling(24).sum()
            df['vwap'] = tp_vol / (vol_24 + 1e-10)
        else:
            df['obv']  = 0.0
            df['vwap'] = close

        # ── Momentum & Prix ──────────────────────────────────────────────────
        df['momentum']      = close.diff(10)
        df['price_change']  = close.pct_change()
        df['volume_change'] = df['volume'].pct_change() if has_volume else 0.0

        # ── Support / Résistance ─────────────────────────────────────────────
        df['resistance'] = high.rolling(20).max()
        df['support']    = low.rolling(20).min()

        df = df.replace([np.inf, -np.inf], np.nan)
        return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    processor = DataPreprocessorSimple(timeframes=['1h', '4h'])
    print("✅ DataPreprocessorSimple prêt !")
