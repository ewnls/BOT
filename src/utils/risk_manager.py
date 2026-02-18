"""
Risk Management avancé pour trading ML
Position sizing, stop-loss dynamique, limite de drawdown
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple


class RiskManager:
    """Gestion du risque professionnelle."""

    def __init__(self, config: Optional[dict] = None):
        default_config = {
            # Capital & Position Sizing
            'initial_capital'      : 10000,
            'max_risk_per_trade'   : 0.02,   # 2% max risqué par trade
            'max_position_size'    : 0.20,   # 20% max du capital par position
            'max_daily_risk'       : 0.06,   # 6% max perdu par jour
            'max_drawdown'         : 0.20,   # Stop si -20% depuis le peak
            # Stops & Targets
            'use_atr_stops'        : True,
            'atr_multiplier_sl'    : 2.0,
            'atr_multiplier_tp'    : 3.0,
            'min_risk_reward'      : 1.5,    # Ratio RR minimum acceptable
            # Limites globales
            'max_concurrent_trades': 3,
            'max_correlated_trades': 2,
            'min_time_between_trades': 60,   # minutes
            # Kelly Criterion
            'use_kelly'            : False,
            'kelly_fraction'       : 0.25,
        }
        self.config = {**default_config, **(config or {})}

        self.capital       = self.config['initial_capital']
        self.peak_capital  = self.capital
        self.daily_pnl     = 0.0
        self.daily_trades  = []
        self.open_positions = []
        self.last_trade_time: Optional[datetime] = None
        self._last_reset_day: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Position sizing
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        win_rate: Optional[float] = None,
    ) -> float:
        """
        Taille de position optimale.
        Méthodes disponibles : Fixed Risk, Kelly Criterion.
        """
        risk_amount   = self.capital * self.config['max_risk_per_trade']
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return 0.0

        position_size = risk_amount / risk_per_unit
        max_units     = (self.capital * self.config['max_position_size']) / entry_price
        position_size = min(position_size, max_units)

        if self.config['use_kelly'] and win_rate:
            kelly_size    = self._kelly_criterion(win_rate, entry_price, stop_loss)
            position_size = min(position_size, kelly_size)

        return position_size

    def _kelly_criterion(
        self,
        win_rate: float,
        entry: float,
        sl: float,
        tp: Optional[float] = None,
    ) -> float:
        """
        Kelly Criterion conservateur:
          f* = (p × b − q) / b
          où p=win_rate, q=1-p, b=gain_moyen/perte_moyenne
        """
        if tp is None:
            tp = entry * 1.03

        p       = win_rate
        q       = 1.0 - p
        avg_win = abs(tp - entry)
        avg_loss = abs(entry - sl)

        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss
        kelly_pct = max(0.0, (p * b - q) / b)
        kelly_fraction = kelly_pct * self.config['kelly_fraction']
        return (self.capital * kelly_fraction) / entry

    # ─────────────────────────────────────────────────────────────────────────
    # ATR Stops
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_atr_stops(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr_period: int = 14,
    ) -> Tuple[float, float, float]:
        """Stop-loss et take-profit basés sur l'ATR."""
        tail   = df.tail(atr_period)
        high   = tail['high']
        low    = tail['low']
        close  = tail['close']

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.mean()

        sl = current_price - atr * self.config['atr_multiplier_sl']
        tp = current_price + atr * self.config['atr_multiplier_tp']
        return sl, tp, float(atr)

    # ─────────────────────────────────────────────────────────────────────────
    # Vérifications avant ouverture de trade
    # ─────────────────────────────────────────────────────────────────────────

    def can_open_trade(
        self,
        signal_time: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """Vérifie si un nouveau trade est autorisé."""
        # FIX: reset automatique si nouveau jour
        if signal_time is not None:
            if (self._last_reset_day is None
                    or signal_time.date().toordinal() > self._last_reset_day):
                self.reset_daily()
                self._last_reset_day = signal_time.date().toordinal()

        if len(self.open_positions) >= self.config['max_concurrent_trades']:
            return False, "Max positions ouvertes atteint"

        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if current_drawdown >= self.config['max_drawdown']:
            return False, f"Max drawdown atteint ({current_drawdown:.1%})"

        if abs(self.daily_pnl) >= self.capital * self.config['max_daily_risk']:
            return False, "Limite quotidienne de perte atteinte"

        if signal_time and self.last_trade_time:
            minutes_since = (signal_time - self.last_trade_time).total_seconds() / 60
            if minutes_since < self.config['min_time_between_trades']:
                return False, (f"Attendre {self.config['min_time_between_trades']} min "
                               f"entre trades")

        return True, "OK"

    def validate_trade(
        self,
        entry: float,
        sl: float,
        tp: float,
    ) -> Tuple[bool, str]:
        """Valide qu'un trade respecte le ratio risk/reward minimum."""
        risk   = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False, "Stop-loss invalide (risque = 0)"

        rr = reward / risk
        if rr < self.config['min_risk_reward']:
            return False, (f"Risk/Reward insuffisant "
                           f"({rr:.2f} < {self.config['min_risk_reward']})")

        return True, f"RR: {rr:.2f} ✅"

    # ─────────────────────────────────────────────────────────────────────────
    # Mise à jour du capital
    # ─────────────────────────────────────────────────────────────────────────

    def update_capital(self, pnl: float, trade_time: Optional[datetime] = None):
        """Met à jour le capital après fermeture d'un trade."""
        self.capital    += pnl
        self.daily_pnl  += pnl

        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        if trade_time:
            self.last_trade_time = trade_time

    def add_position(self, position: dict):
        """Enregistre une position ouverte."""
        self.open_positions.append(position)

    def close_position(self, position_id):
        """Retire une position de la liste des positions ouvertes."""
        self.open_positions = [
            p for p in self.open_positions if p.get('id') != position_id
        ]

    def get_metrics(self) -> dict:
        """Retourne les métriques de risque actuelles."""
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        return {
            'capital'        : round(self.capital, 2),
            'peak_capital'   : round(self.peak_capital, 2),
            'current_drawdown': round(current_drawdown, 4),
            'daily_pnl'      : round(self.daily_pnl, 2),
            'open_positions' : len(self.open_positions),
            'risk_used_pct'  : round(
                len(self.open_positions) / self.config['max_concurrent_trades'] * 100, 1
            ),
        }

    def reset_daily(self):
        """Reset les compteurs journaliers."""
        self.daily_pnl    = 0.0
        self.daily_trades = []


# ─────────────────────────────────────────────────────────────────────────────
# Configurations prédéfinies
# ─────────────────────────────────────────────────────────────────────────────

# FIX: AGGRESSIVE_CONFIG — valeurs corrigées (l'original était suicidaire)
#   max_drawdown     : 0.80 → 0.30  (80% de perte = ruine certaine)
#   max_position_size: 0.90 → 0.30  (90% du capital sur 1 trade = trop risqué)
#   max_risk_per_trade: 0.10 → 0.05 (10% risqué/trade trop élevé)
#   min_risk_reward  : 0.50 → 1.20  (0.5 = perdre 2 pour gagner 1, ruine structurelle)
AGGRESSIVE_CONFIG = {
    'initial_capital'        : 10000,
    'max_risk_per_trade'     : 0.05,   # 5% du capital risqué par trade (crypto ok)
    'max_position_size'      : 0.30,   # 30% max du capital par position
    'max_daily_risk'         : 0.15,   # 15% max perdu par jour
    'max_drawdown'           : 0.30,   # Stop si -30% depuis le peak
    'use_atr_stops'          : True,
    'atr_multiplier_sl'      : 1.5,
    'atr_multiplier_tp'      : 2.5,
    'min_risk_reward'        : 1.20,   # Minimum : gagner 1.2× ce qu'on risque
    'max_concurrent_trades'  : 5,
    'max_correlated_trades'  : 3,
    'min_time_between_trades': 30,
    'use_kelly'              : False,
    'kelly_fraction'         : 0.25,
}

# FIX: CONSERVATIVE_CONFIG — min_risk_reward corrigé (0.5 → 1.5)
CONSERVATIVE_CONFIG = {
    'initial_capital'        : 10000,
    'max_risk_per_trade'     : 0.02,   # 2% du capital risqué par trade
    'max_position_size'      : 0.20,   # 20% max du capital par position
    'max_daily_risk'         : 0.05,   # 5% max perdu par jour
    'max_drawdown'           : 0.15,   # Stop si -15% depuis le peak
    'use_atr_stops'          : True,
    'atr_multiplier_sl'      : 2.0,
    'atr_multiplier_tp'      : 3.0,
    'min_risk_reward'        : 1.50,   # RR minimum strict
    'max_concurrent_trades'  : 3,
    'max_correlated_trades'  : 2,
    'min_time_between_trades': 60,
    'use_kelly'              : False,
    'kelly_fraction'         : 0.25,
}
