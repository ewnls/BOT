"""
Backtesting simplifié
PNL, Drawdown, Sharpe Ratio, Précision, Win Rate, Profit Factor
"""

import numpy as np
import pandas as pd
from typing import Dict
from src.utils.risk_manager import RiskManager  # toujours utilisé pour les métriques


class Backtester:
    """Système de backtesting (version simplifiée pour debug)"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_config=None,
        timeframe: str = "1d"
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.timeframe = timeframe
        self.trades = []
        self.equity_curve = []

        # On garde un RiskManager pour les métriques, mais on ne l'utilise pas pour bloquer des trades
        self.risk_manager = RiskManager(risk_config)

    def execute_trades(
        self,
        predictions: np.ndarray,
        actual_prices: pd.DataFrame,
        k_tp1: float = 1.0,
        k_tp2: float = 2.0,
        k_sl: float = 1.0
    ):
        """
        VERSION DEBUG SIMPLE :
        - Ignore totalement RiskManager pour l'ouverture/fermeture
        - Ouvre des trades forcés sur les premières bougies

        predictions: array (n_samples, 3) = [target_class, target_return, atr]
        actual_prices: DataFrame avec ['open', 'high', 'low', 'close']
        """

        capital = self.initial_capital
        position = None
        self.trades = []
        self.equity_curve = []

        atr_pred = predictions[:, 2]

        print(">>> execute_trades called with", len(predictions), "samples")

        for i in range(len(predictions)):
            current_price = actual_prices.iloc[i]["close"]
            atr = atr_pred[i]

            if i < 5:
                print(f"i={i}, close={current_price}, atr={atr}")

            # skip ATR invalide
            if not np.isfinite(atr) or atr <= 0:
                if position:
                    self.equity_curve.append(
                        capital + position["shares"] * current_price
                    )
                else:
                    self.equity_curve.append(capital)
                continue

            tp1_price = current_price + k_tp1 * atr
            tp2_price = current_price + k_tp2 * atr
            sl_price  = current_price - k_sl * atr

            # === ENTRÉE FORCÉE POUR DEBUG ===
            if position is None and i < 3:
                print("OPENING FORCE TRADE at i", i)
                entry_price = current_price * (1 + self.slippage)

                # Position = 10% du capital
                position_size = capital * 0.10
                if position_size > capital * 0.95:
                    position_size = capital * 0.95

                shares = position_size / entry_price
                commission_cost = position_size * self.commission

                position = {
                    "entry_idx": i,
                    "entry_price": entry_price,
                    "shares": shares,
                    "tp1": tp1_price,
                    "tp2": tp2_price,
                    "sl": sl_price,
                    "commission": commission_cost,
                }

                capital -= position_size + commission_cost

            elif position is not None:
                high = actual_prices.iloc[i]["high"]
                low = actual_prices.iloc[i]["low"]

                exit_price = None
                exit_reason = None

                if high >= position["tp2"]:
                    exit_price = position["tp2"] * (1 - self.slippage)
                    exit_reason = "TP2"
                elif high >= position["tp1"]:
                    exit_price = position["tp1"] * (1 - self.slippage)
                    exit_reason = "TP1"
                elif low <= position["sl"]:
                    exit_price = position["sl"] * (1 - self.slippage)
                    exit_reason = "SL"
                elif i - position["entry_idx"] >= 20:
                    exit_price = current_price * (1 - self.slippage)
                    exit_reason = "TIMEOUT"

                if exit_price is not None:
                    exit_value = position["shares"] * exit_price
                    commission_cost = exit_value * self.commission
                    entry_cost = position["shares"] * position["entry_price"]
                    pnl = (
                        exit_value
                        - entry_cost
                        - commission_cost
                        - position["commission"]
                    )

                    capital += exit_value - commission_cost

                    self.trades.append(
                        {
                            "entry_idx": position["entry_idx"],
                            "exit_idx": i,
                            "entry_price": position["entry_price"],
                            "exit_price": exit_price,
                            "shares": position["shares"],
                            "pnl": pnl,
                            "pnl_pct": (pnl / entry_cost) * 100 if entry_cost != 0 else 0.0,
                            "exit_reason": exit_reason,
                            "duration": i - position["entry_idx"],
                        }
                    )

                    position = None

            # Mise à jour equity
            if position:
                current_value = capital + position["shares"] * current_price
            else:
                current_value = capital

            self.equity_curve.append(current_value)

        return self.trades, self.equity_curve

    def calculate_metrics(self) -> Dict:
        """Calcule toutes les métriques de performance"""
        if not self.trades:
            return {"error": "Aucun trade exécuté"}

        trades_df = pd.DataFrame(self.trades)
        equity = np.array(self.equity_curve)

        total_pnl = trades_df["pnl"].sum()
        pnl_pct = ((equity[-1] - self.initial_capital) / self.initial_capital) * 100

        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] <= 0]
        win_rate = (
            len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0.0
        )

        gross_profit = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0.0
        gross_loss = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        cumulative = np.maximum.accumulate(equity)
        drawdown = (equity - cumulative) / cumulative * 100
        max_drawdown = drawdown.min()

        returns = np.diff(equity) / equity[:-1]
        sharpe_ratio = (
            (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            if np.std(returns) > 0
            else 0.0
        )

        tp1_hits = len(trades_df[trades_df["exit_reason"] == "TP1"])
        tp2_hits = len(trades_df[trades_df["exit_reason"] == "TP2"])
        sl_hits = len(trades_df[trades_df["exit_reason"] == "SL"])
        precision = (
            ((tp1_hits + tp2_hits) / len(trades_df)) * 100
            if len(trades_df) > 0
            else 0.0
        )

        avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0.0
        avg_trade_duration = (
            trades_df["duration"].mean() if len(trades_df) > 0 else 0.0
        )

        # On continue d'utiliser les métriques internes du RiskManager pour info
        risk_metrics = self.risk_manager.get_metrics()

        metrics = {
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "total_pnl": round(total_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "final_capital": round(equity[-1], 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "precision": round(precision, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_trade_duration": round(avg_trade_duration, 2),
            "tp1_hits": tp1_hits,
            "tp2_hits": tp2_hits,
            "sl_hits": sl_hits,
            "risk_drawdown": round(risk_metrics["current_drawdown"] * 100, 2),
            "peak_capital": round(risk_metrics["peak_capital"], 2),
        }

        return metrics

    def get_trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> pd.Series:
        return pd.Series(self.equity_curve)

    def print_report(self):
        metrics = self.calculate_metrics()

        if "error" in metrics:
            print(metrics["error"])
            return

        print("=" * 60)
        print("RAPPORT DE BACKTESTING (DEBUG SIMPLE)")
        print("=" * 60)

        print("\n📊 RÉSUMÉ DES TRADES")
        print(f"Total trades: {metrics['total_trades']}")
        print(
            f"Gagnants: {metrics['winning_trades']} | Perdants: {metrics['losing_trades']}"
        )
        print(
            f"TP1: {metrics['tp1_hits']} | TP2: {metrics['tp2_hits']} | SL: {metrics['sl_hits']}"
        )

        print("\n💰 PROFIT & LOSS")
        print(f"PNL Total: ${metrics['total_pnl']} ({metrics['pnl_pct']}%)")
        print(f"Capital Initial: ${self.initial_capital}")
        print(f"Capital Final: ${metrics['final_capital']}")
        print(f"Peak Capital: ${metrics['peak_capital']}")

        print("\n📈 PERFORMANCE")
        print(f"Win Rate: {metrics['win_rate']}%")
        print(f"Profit Factor: {metrics['profit_factor']}")
        print(f"Précision: {metrics['precision']}%")

        print("\n⚠️ RISQUE")
        print(f"Max Drawdown: {metrics['max_drawdown']}%")
        print(f"Risk Manager Drawdown: {metrics['risk_drawdown']}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")

        print("\n📊 MOYENNES")
        print(f"Gain moyen: ${metrics['avg_win']}")
        print(f"Perte moyenne: ${metrics['avg_loss']}")
        print(f"Durée moyenne: {metrics['avg_trade_duration']} périodes")

        print("\n🛡️ CONFIGURATION RISK MANAGEMENT")
        print(
            f"Max risque/trade: {self.risk_manager.config['max_risk_per_trade']*100}%"
        )
        print(
            f"Max position size: {self.risk_manager.config['max_position_size']*100}%"
        )
        print(f"Max drawdown: {self.risk_manager.config['max_drawdown']*100}%")
        print(f"Min Risk/Reward: {self.risk_manager.config['min_risk_reward']}")

        print("=" * 60)
