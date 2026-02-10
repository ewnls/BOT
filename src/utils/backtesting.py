"""
Backtesting avec Risk Management intégré
PNL, Drawdown, Sharpe Ratio, Précision, Win Rate, Profit Factor
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.utils.risk_manager import RiskManager


class Backtester:
    """Système de backtesting avec gestion du risque"""

    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.0005, risk_config=None, timeframe='1d'):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.timeframe = timeframe
        self.trades = []
        self.equity_curve = []

        # Risk Manager intégré
        self.risk_manager = RiskManager(risk_config)

    def execute_trades(self, predictions: np.ndarray, actual_prices: pd.DataFrame):
        """
        Exécute les trades avec Risk Management
        predictions: array (n_samples, 4) VARIATIONS % [entry, tp1, tp2, sl]
        actual_prices: DataFrame avec ['open', 'high', 'low', 'close']
        """
        capital = self.initial_capital
        position = None

        # Pour limiter le sur-trading
        min_bars_between_trades = 5    # nombre minimum de bougies entre deux entrées
        last_entry_idx = -9999

        # CONVERSION : Variations % → Prix absolus
        print(f"\n🔄 Conversion variations → prix absolus...")
        print(f"   Variations range: [{predictions.min()*100:.2f}%, {predictions.max()*100:.2f}%]")

        predictions_prices = np.zeros_like(predictions)

        for i in range(len(predictions)):
            current_price = actual_prices.iloc[i]['close']

            # predictions[i,0] = variation entry (ex: 0.008 = +0.8%)
            predictions_prices[i, 0] = current_price * (1 + predictions[i, 0])  # entry
            predictions_prices[i, 1] = current_price * (1 + predictions[i, 1])  # tp1
            predictions_prices[i, 2] = current_price * (1 + predictions[i, 2])  # tp2
            predictions_prices[i, 3] = current_price * (1 + predictions[i, 3])  # sl

        predictions = predictions_prices

        print(f"   Prix convertis: [{predictions.min():.2f}, {predictions.max():.2f}]")

        for i in range(len(predictions)):
            current_price = actual_prices.iloc[i]['close']

            # Les prédictions sont maintenant des PRIX ABSOLUS
            entry_pred = predictions[i, 0]
            tp1_pred = predictions[i, 1]
            tp2_pred = predictions[i, 2]
            sl_pred = predictions[i, 3]

            # Calcule la variation prédite pour le signal
            entry_variation = (entry_pred - current_price) / current_price

            # Seuils par timeframe (déjà en place)
            if '1d' in self.timeframe or '1w' in self.timeframe:
                threshold = 0.0005  # 0.05%
            elif any(tf in self.timeframe for tf in ['4h', '6h', '12h']):
                threshold = 0.001   # 0.10%
            elif any(tf in self.timeframe for tf in ['1h', '2h']):
                threshold = 0.002   # 0.20%
            elif any(tf in self.timeframe for tf in ['15m', '30m']):
                threshold = 0.003   # 0.30%
            else:  # 1m, 5m
                threshold = 0.005   # 0.50%

            # Seuil global minimal pour éviter les signaux trop faibles
            global_min_threshold = 0.005  # 0.5 %
            effective_threshold = max(threshold, global_min_threshold)

            # Filtre simple de cooldown entre trades
            too_soon = (i - last_entry_idx) < min_bars_between_trades

            # (Optionnel) Filtre de tendance simple: n'autoriser les longs que si close > MA50
            # ma_window = 50
            # if i >= ma_window:
            #     ma50 = actual_prices['close'].iloc[i-ma_window+1:i+1].mean()
            #     trend_ok = current_price > ma50
            # else:
            #     trend_ok = True
            trend_ok = True  # laisser à True si tu ne veux pas de filtre de tendance pour l'instant

            # SIGNAL D'ENTRÉE
            if (
                position is None
                and entry_variation > effective_threshold
                and not too_soon
                and trend_ok
            ):
                # DEBUG: Limite affichage aux 3 premiers signaux
                if len(self.trades) < 3:
                    print(f"\n🔔 Signal détecté à l'index {i}")
                    print(f"   Prix actuel: {current_price:.2f}")
                    print(f"   Prix entry prédit: {entry_pred:.2f} ({entry_variation*100:+.2f}%)")
                    print(f"   Threshold effectif: {effective_threshold*100:.2f}%")
                    print(f"   TP1: {tp1_pred:.2f} | TP2: {tp2_pred:.2f} | SL: {sl_pred:.2f}")

                # Prix d'entrée réel avec slippage
                entry_price = current_price * (1 + self.slippage)

                # Utilise les niveaux prédits par le modèle
                tp1_target = tp1_pred
                tp2_target = tp2_pred
                sl_target = sl_pred

                # Vérifie si on peut ouvrir un trade
                can_trade, reason = self.risk_manager.can_open_trade()
                if len(self.trades) < 3:
                    print(f"   Can trade: {can_trade} | Raison: {reason}")
                if not can_trade:
                    continue

                # Valide le risk/reward
                valid, msg = self.risk_manager.validate_trade(entry_price, sl_target, tp1_target)
                if len(self.trades) < 3:
                    print(f"   Trade valide: {valid} | Message: {msg}")
                if not valid:
                    continue

                # Calcule position size avec Risk Manager
                position_size_units = self.risk_manager.calculate_position_size(
                    entry_price=entry_price,
                    stop_loss=sl_target,
                    win_rate=0.60
                )

                if len(self.trades) < 3:
                    print(f"   Position size: {position_size_units:.4f} units")

                if position_size_units == 0:
                    if len(self.trades) < 3:
                        print(f"   ❌ Position size = 0, skip")
                    continue

                position_size = position_size_units * entry_price

                # Vérifie qu'on a assez de capital
                if position_size > capital * 0.95:
                    position_size = capital * 0.95
                    position_size_units = position_size / entry_price

                shares = position_size / entry_price
                commission_cost = position_size * self.commission

                position = {
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'shares': shares,
                    'tp1': tp1_target,
                    'tp2': tp2_target,
                    'sl': sl_target,
                    'commission': commission_cost,
                    'id': len(self.trades)
                }

                # Enregistre la position ouverte dans le Risk Manager
                self.risk_manager.add_position(position)

                capital -= (position_size + commission_cost)

                last_entry_idx = i  # on vient d'entrer en position

                if len(self.trades) < 3:
                    print(f"   ✅ Trade ouvert #{position['id']}")

            # GESTION DE POSITION OUVERTE
            elif position is not None:
                high = actual_prices.iloc[i]['high']
                low = actual_prices.iloc[i]['low']

                exit_price = None
                exit_reason = None

                # Check TP2 (priorité max)
                if high >= position['tp2']:
                    exit_price = position['tp2'] * (1 - self.slippage)
                    exit_reason = 'TP2'

                # Check TP1
                elif high >= position['tp1']:
                    exit_price = position['tp1'] * (1 - self.slippage)
                    exit_reason = 'TP1'

                # Check SL
                elif low <= position['sl']:
                    exit_price = position['sl'] * (1 - self.slippage)
                    exit_reason = 'SL'

                # Timeout après 20 périodes
                elif i - position['entry_idx'] >= 20:
                    exit_price = current_price * (1 - self.slippage)
                    exit_reason = 'TIMEOUT'

                # FERMETURE DE POSITION
                if exit_price:
                    exit_value = position['shares'] * exit_price
                    commission_cost = exit_value * self.commission
                    entry_cost = position['shares'] * position['entry_price']
                    pnl = exit_value - entry_cost - commission_cost - position['commission']

                    capital += exit_value - commission_cost

                    # Met à jour le Risk Manager
                    self.risk_manager.update_capital(pnl)
                    self.risk_manager.close_position(position['id'])

                    self.trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'pnl_pct': (pnl / entry_cost) * 100,
                        'exit_reason': exit_reason,
                        'duration': i - position['entry_idx']
                    })

                    if len(self.trades) <= 3:
                        print(f"   ✅ Trade fermé: {exit_reason} | PNL: ${pnl:.2f} ({(pnl/entry_cost)*100:+.2f}%)")

                    position = None

            # Track equity
            if position:
                current_value = capital + (position['shares'] * current_price)
            else:
                current_value = capital

            self.equity_curve.append(current_value)

        return self.trades, self.equity_curve

    def calculate_metrics(self) -> Dict:
        """Calcule toutes les métriques de performance"""
        if not self.trades:
            return {'error': 'Aucun trade exécuté'}

        trades_df = pd.DataFrame(self.trades)
        equity = np.array(self.equity_curve)

        # PNL
        total_pnl = trades_df['pnl'].sum()
        pnl_pct = ((equity[-1] - self.initial_capital) / self.initial_capital) * 100

        # Win Rate
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(winning_trades) / len(trades_df) * 100

        # Profit Factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        cumulative = np.maximum.accumulate(equity)
        drawdown = (equity - cumulative) / cumulative * 100
        max_drawdown = drawdown.min()

        # Sharpe Ratio
        returns = np.diff(equity) / equity[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Précision
        tp1_hits = len(trades_df[trades_df['exit_reason'] == 'TP1'])
        tp2_hits = len(trades_df[trades_df['exit_reason'] == 'TP2'])
        sl_hits = len(trades_df[trades_df['exit_reason'] == 'SL'])
        precision = ((tp1_hits + tp2_hits) / len(trades_df)) * 100

        # Moyennes
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        avg_trade_duration = trades_df['duration'].mean()

        # Métriques Risk Manager
        risk_metrics = self.risk_manager.get_metrics()

        metrics = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),

            'total_pnl': round(total_pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'final_capital': round(equity[-1], 2),

            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'precision': round(precision, 2),

            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),

            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_trade_duration': round(avg_trade_duration, 2),

            'tp1_hits': tp1_hits,
            'tp2_hits': tp2_hits,
            'sl_hits': sl_hits,

            # Risk metrics
            'risk_drawdown': round(risk_metrics['current_drawdown'] * 100, 2),
            'peak_capital': round(risk_metrics['peak_capital'], 2)
        }

        return metrics

    def get_trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> pd.Series:
        return pd.Series(self.equity_curve)

    def print_report(self):
        """Affiche un rapport complet avec Risk Management"""
        metrics = self.calculate_metrics()

        if 'error' in metrics:
            print(metrics['error'])
            return

        print("=" * 60)
        print("RAPPORT DE BACKTESTING (avec Risk Management)")
        print("=" * 60)

        print(f"\n📊 RÉSUMÉ DES TRADES")
        print(f"Total trades: {metrics['total_trades']}")
        print(f"Gagnants: {metrics['winning_trades']} | Perdants: {metrics['losing_trades']}")
        print(f"TP1: {metrics['tp1_hits']} | TP2: {metrics['tp2_hits']} | SL: {metrics['sl_hits']}")

        print(f"\n💰 PROFIT & LOSS")
        print(f"PNL Total: ${metrics['total_pnl']} ({metrics['pnl_pct']}%)")
        print(f"Capital Initial: ${self.initial_capital}")
        print(f"Capital Final: ${metrics['final_capital']}")
        print(f"Peak Capital: ${metrics['peak_capital']}")

        print(f"\n📈 PERFORMANCE")
        print(f"Win Rate: {metrics['win_rate']}%")
        print(f"Profit Factor: {metrics['profit_factor']}")
        print(f"Précision: {metrics['precision']}%")

        print(f"\n⚠️ RISQUE")
        print(f"Max Drawdown: {metrics['max_drawdown']}%")
        print(f"Risk Manager Drawdown: {metrics['risk_drawdown']}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")

        print(f"\n📊 MOYENNES")
        print(f"Gain moyen: ${metrics['avg_win']}")
        print(f"Perte moyenne: ${metrics['avg_loss']}")
        print(f"Durée moyenne: {metrics['avg_trade_duration']} périodes")

        # Affiche config Risk Manager
        print(f"\n🛡️ CONFIGURATION RISK MANAGEMENT")
        print(f"Max risque/trade: {self.risk_manager.config['max_risk_per_trade']*100}%")
        print(f"Max position size: {self.risk_manager.config['max_position_size']*100}%")
        print(f"Max drawdown: {self.risk_manager.config['max_drawdown']*100}%")
        print(f"Min Risk/Reward: {self.risk_manager.config['min_risk_reward']}")

        print("=" * 60)
