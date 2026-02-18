"""
Backtesting simple ATR avec filtres anti-surtrading
PNL, Drawdown, Sharpe Ratio, Précision, Win Rate, Profit Factor
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# FIX: facteur d'annualisation Sharpe correct par timeframe
# (l'original utilisait sqrt(252) pour TOUS les timeframes → faux pour 1h, 4h, 1w…)
_BARS_PER_YEAR: Dict[str, int] = {
    '1m' : 252 * 24 * 60,   # 362 880
    '5m' : 252 * 24 * 12,   #  72 576
    '15m': 252 * 24 *  4,   #  24 192
    '30m': 252 * 24 *  2,   #  12 096
    '1h' : 252 * 24,        #   6 048
    '2h' : 252 * 12,        #   3 024
    '4h' : 252 *  6,        #   1 512
    '6h' : 252 *  4,        #   1 008
    '12h': 252 *  2,        #     504
    '1d' : 252,             #     252
    '1w' : 52,              #      52
}

# FIX: TIMEOUT adaptatif (~5 jours réels quelle que soit la timeframe)
# (l'original avait 20 barres fixe → 20 semaines en 1w, 20 minutes en 1m)
_TIMEOUT_BARS: Dict[str, int] = {
    '1m' : 7200,
    '5m' : 1440,
    '15m': 480,
    '30m': 240,
    '1h' : 120,
    '2h' :  60,
    '4h' :  30,
    '6h' :  20,
    '12h':  10,
    '1d' :   5,
    '1w' :   2,
}


def _get_bars_per_year(timeframe: str) -> int:
    for tf, val in _BARS_PER_YEAR.items():
        if tf in timeframe:
            return val
    return 252


def _get_timeout_bars(timeframe: str) -> int:
    for tf, val in _TIMEOUT_BARS.items():
        if tf in timeframe:
            return val
    return 20


class Backtester:
    """Système de backtesting simple basé sur l'ATR."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_config: Optional[dict] = None,
        timeframe: str = '1d',
    ):
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage
        self.timeframe       = timeframe
        self.risk_config     = risk_config or {}
        self.trades          = []
        self.equity_curve    = []

        # FIX: paramètres de risque issus du risk_config (enfin utilisés ici)
        self.max_risk_per_trade = self.risk_config.get('max_risk_per_trade', 0.02)
        self.max_position_size  = self.risk_config.get('max_position_size', 0.20)
        self.max_drawdown_limit = self.risk_config.get('max_drawdown', 0.25)

    # ─────────────────────────────────────────────────────────────────────────
    # Position sizing basé sur le risque réel
    # FIX: remplace la taille fixe de 10% par un sizing basé sur la distance au SL
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_position_size(
        self, capital: float, entry_price: float, sl_price: float
    ) -> float:
        """
        Taille de position basée sur le risque réel:
          shares = (capital × max_risk_per_trade) / (entry - sl)
        Plafonné par max_position_size.
        """
        risk_amount   = capital * self.max_risk_per_trade
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit == 0:
            return 0.0
        shares     = risk_amount / risk_per_unit
        max_shares = (capital * self.max_position_size) / entry_price
        return min(shares, max_shares)

    # ─────────────────────────────────────────────────────────────────────────
    # Exécution des trades
    # ─────────────────────────────────────────────────────────────────────────

    def execute_trades(
        self,
        predictions: np.ndarray,
        actual_prices: pd.DataFrame,
        k_tp1: float = 1.0,
        k_tp2: float = 2.0,
        k_sl: float = 1.0,
        use_model_filters: bool = True,
    ):
        """
        predictions   : array (n_samples, 3) = [target_class, target_return, atr]
        actual_prices : DataFrame avec colonnes ['open', 'high', 'low', 'close']
        """
        capital       = self.initial_capital
        peak_capital  = self.initial_capital
        position      = None
        self.trades   = []
        self.equity_curve = []

        target_class_pred  = predictions[:, 0]
        target_return_pred = predictions[:, 1]
        atr_pred           = predictions[:, 2]

        # Cooldown adaptatif
        if '1w' in self.timeframe:
            min_bars_between_trades = 1
        elif '1d' in self.timeframe:
            min_bars_between_trades = 2
        elif any(tf in self.timeframe for tf in ['12h', '6h']):
            min_bars_between_trades = 3
        elif any(tf in self.timeframe for tf in ['4h', '2h', '1h']):
            min_bars_between_trades = 5
        else:
            min_bars_between_trades = 8     # 5m/15m/30m

        # FIX: TIMEOUT adaptatif (représente toujours ~5 jours réels)
        timeout_bars = _get_timeout_bars(self.timeframe)

        # Ajustement TP2 pour l'intraday
        k_tp2_local = k_tp2 * 0.9 if any(
            tf in self.timeframe for tf in ['30m', '15m', '5m']
        ) else k_tp2

        # Seuil minimum retour prédit
        if any(tf in self.timeframe for tf in ['5m', '15m']):
            min_predicted_return = 0.0003    # 0.03%
        elif any(tf in self.timeframe for tf in ['30m', '1h']):
            min_predicted_return = 0.0005    # 0.05%
        elif any(tf in self.timeframe for tf in ['2h', '4h']):
            min_predicted_return = 0.001     # 0.1%
        else:
            min_predicted_return = 0.002     # 0.2%

        last_entry_idx = -9999

        for i in range(len(predictions)):
            current_price = actual_prices.iloc[i]['close']
            atr           = atr_pred[i]
            cls           = target_class_pred[i]
            pred_ret      = target_return_pred[i]

            # ATR invalide → skip
            if not np.isfinite(atr) or atr <= 0:
                self.equity_curve.append(
                    capital + position['shares'] * current_price
                    if position else capital
                )
                continue

            tp1_price = current_price + k_tp1 * atr
            tp2_price = current_price + k_tp2_local * atr
            sl_price  = current_price - k_sl * atr

            amplitude_pct = (tp1_price - current_price) / current_price

            # Amplitude minimale selon timeframe
            if '1w' in self.timeframe or '1d' in self.timeframe:
                min_amplitude = 0.005     # 0.5%
            elif any(tf in self.timeframe for tf in ['12h', '6h']):
                min_amplitude = 0.007     # 0.7%
            elif any(tf in self.timeframe for tf in ['4h', '2h']):
                min_amplitude = 0.008     # 0.8%
            elif '1h' in self.timeframe:
                min_amplitude = 0.008     # 0.8%
            else:
                min_amplitude = 0.005     # 0.5% pour intraday court

            if amplitude_pct < min_amplitude:
                self.equity_curve.append(
                    capital + position['shares'] * current_price
                    if position else capital
                )
                continue

            too_soon = (i - last_entry_idx) < min_bars_between_trades

            # FIX: coupe-circuit drawdown global (enfin branché sur risk_config)
            current_drawdown = (peak_capital - capital) / peak_capital
            if current_drawdown >= self.max_drawdown_limit:
                self.equity_curve.append(
                    capital + position['shares'] * current_price
                    if position else capital
                )
                continue

            # ── ENTRÉE ──────────────────────────────────────────────────────
            if position is None and not too_soon:
                if use_model_filters:
                    if cls < 0:
                        self.equity_curve.append(capital)
                        continue
                    if np.isfinite(pred_ret) and pred_ret < min_predicted_return:
                        self.equity_curve.append(capital)
                        continue

                entry_price = current_price * (1 + self.slippage)

                # FIX: sizing basé sur le risque réel (entry → SL)
                shares = self._compute_position_size(capital, entry_price, sl_price)
                if shares <= 0:
                    self.equity_curve.append(capital)
                    continue

                position_value  = shares * entry_price
                commission_cost = position_value * self.commission

                if position_value + commission_cost > capital:
                    self.equity_curve.append(capital)
                    continue

                capital -= position_value + commission_cost
                position = {
                    'entry_idx'  : i,
                    'entry_price': entry_price,
                    'shares'     : shares,
                    'tp1'        : tp1_price,
                    'tp2'        : tp2_price,
                    'sl'         : sl_price,
                    'commission' : commission_cost,
                }
                last_entry_idx = i

            # ── GESTION POSITION ─────────────────────────────────────────────
            elif position is not None:
                high = actual_prices.iloc[i]['high']
                low  = actual_prices.iloc[i]['low']

                exit_price  = None
                exit_reason = None

                if high >= position['tp2']:
                    exit_price  = position['tp2'] * (1 - self.slippage)
                    exit_reason = 'TP2'
                elif high >= position['tp1']:
                    exit_price  = position['tp1'] * (1 - self.slippage)
                    exit_reason = 'TP1'
                elif low <= position['sl']:
                    exit_price  = position['sl'] * (1 - self.slippage)
                    exit_reason = 'SL'
                elif i - position['entry_idx'] >= timeout_bars:
                    exit_price  = current_price * (1 - self.slippage)
                    exit_reason = 'TIMEOUT'

                if exit_price is not None:
                    exit_value      = position['shares'] * exit_price
                    commission_cost = exit_value * self.commission
                    entry_cost      = position['shares'] * position['entry_price']
                    pnl = (exit_value - entry_cost
                           - commission_cost - position['commission'])

                    capital += exit_value - commission_cost
                    if capital > peak_capital:
                        peak_capital = capital

                    self.trades.append({
                        'entry_idx'  : position['entry_idx'],
                        'exit_idx'   : i,
                        'entry_price': position['entry_price'],
                        'exit_price' : exit_price,
                        'shares'     : position['shares'],
                        'pnl'        : pnl,
                        'pnl_pct'    : (pnl / entry_cost * 100) if entry_cost != 0 else 0.0,
                        'exit_reason': exit_reason,
                        'duration'   : i - position['entry_idx'],
                    })
                    position = None

            # Mise à jour courbe de valeur
            current_value = (
                capital + position['shares'] * current_price
                if position else capital
            )
            self.equity_curve.append(current_value)

        return self.trades, self.equity_curve

    # ─────────────────────────────────────────────────────────────────────────
    # Métriques
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_metrics(self) -> Dict:
        """Calcule toutes les métriques de performance."""
        if not self.trades:
            return {'error': 'Aucun trade exécuté'}

        trades_df = pd.DataFrame(self.trades)
        equity    = np.array(self.equity_curve)

        total_pnl = trades_df['pnl'].sum()
        pnl_pct   = (equity[-1] - self.initial_capital) / self.initial_capital * 100

        winning = trades_df[trades_df['pnl'] > 0]
        losing  = trades_df[trades_df['pnl'] <= 0]

        win_rate      = len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0.0
        gross_profit  = winning['pnl'].sum() if len(winning) > 0 else 0.0
        gross_loss    = abs(losing['pnl'].sum()) if len(losing) > 0 else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        cumulative   = np.maximum.accumulate(equity)
        drawdown     = (equity - cumulative) / cumulative * 100
        max_drawdown = drawdown.min()

        # FIX: annualisation correcte selon la timeframe
        returns       = np.diff(equity) / equity[:-1]
        bars_per_year = _get_bars_per_year(self.timeframe)
        sharpe_ratio  = (
            (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)
            if np.std(returns) > 0 else 0.0
        )

        # Sortino (pénalise uniquement les retours négatifs)
        neg_returns   = returns[returns < 0]
        downside_std  = np.std(neg_returns) if len(neg_returns) > 1 else 1e-10
        sortino_ratio = (
            (np.mean(returns) / downside_std) * np.sqrt(bars_per_year)
            if downside_std > 0 else 0.0
        )

        tp1_hits  = int((trades_df['exit_reason'] == 'TP1').sum())
        tp2_hits  = int((trades_df['exit_reason'] == 'TP2').sum())
        sl_hits   = int((trades_df['exit_reason'] == 'SL').sum())
        precision = (tp1_hits + tp2_hits) / len(trades_df) * 100

        return {
            'total_trades'      : len(trades_df),
            'winning_trades'    : len(winning),
            'losing_trades'     : len(losing),
            'total_pnl'         : round(total_pnl, 2),
            'pnl_pct'           : round(pnl_pct, 2),
            'final_capital'     : round(float(equity[-1]), 2),
            'win_rate'          : round(win_rate, 2),
            'profit_factor'     : round(profit_factor, 2),
            'precision'         : round(precision, 2),
            'max_drawdown'      : round(max_drawdown, 2),
            'sharpe_ratio'      : round(sharpe_ratio, 2),
            'sortino_ratio'     : round(sortino_ratio, 2),
            'avg_win'           : round(float(winning['pnl'].mean()), 2) if len(winning) > 0 else 0.0,
            'avg_loss'          : round(float(losing['pnl'].mean()), 2) if len(losing) > 0 else 0.0,
            'avg_trade_duration': round(float(trades_df['duration'].mean()), 2),
            'tp1_hits'          : tp1_hits,
            'tp2_hits'          : tp2_hits,
            'sl_hits'           : sl_hits,
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> pd.Series:
        return pd.Series(self.equity_curve)

    def print_report(self):
        metrics = self.calculate_metrics()
        if 'error' in metrics:
            print(metrics['error'])
            return

        bars_per_year = _get_bars_per_year(self.timeframe)
        print('=' * 60)
        print('RAPPORT DE BACKTESTING (ATR + filtres)')
        print('=' * 60)
        print(f"\n📊 RÉSUMÉ DES TRADES")
        print(f"Total trades : {metrics['total_trades']}")
        print(f"Gagnants     : {metrics['winning_trades']} | Perdants : {metrics['losing_trades']}")
        print(f"TP1: {metrics['tp1_hits']} | TP2: {metrics['tp2_hits']} | SL: {metrics['sl_hits']}")
        print(f"\n💰 PROFIT & LOSS")
        print(f"PNL Total      : ${metrics['total_pnl']} ({metrics['pnl_pct']}%)")
        print(f"Capital Initial: ${self.initial_capital}")
        print(f"Capital Final  : ${metrics['final_capital']}")
        print(f"\n📈 PERFORMANCE")
        print(f"Win Rate      : {metrics['win_rate']}%")
        print(f"Profit Factor : {metrics['profit_factor']}")
        print(f"Précision     : {metrics['precision']}%")
        print(f"\n⚠️ RISQUE")
        print(f"Max Drawdown  : {metrics['max_drawdown']}%")
        print(f"Sharpe Ratio  : {metrics['sharpe_ratio']}  "
              f"(annualisé sur {bars_per_year} barres/an, tf={self.timeframe})")
        print(f"Sortino Ratio : {metrics['sortino_ratio']}")
        print(f"\n📊 MOYENNES")
        print(f"Gain moyen    : ${metrics['avg_win']}")
        print(f"Perte moyenne : ${metrics['avg_loss']}")
        print(f"Durée moyenne : {metrics['avg_trade_duration']} barres")
        print('=' * 60)
