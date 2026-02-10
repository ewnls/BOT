"""
Risk Management avancé pour trading ML
Position sizing, stop-loss dynamique, limite de drawdown
"""
import numpy as np
import pandas as pd

class RiskManager:
    """Gestion du risque professionnel"""
    
    def __init__(self, config=None):
        """
        Config par défaut (conservateur)
        """
        # Config par défaut
        default_config = {
            # Capital & Position Sizing
            'initial_capital': 10000,
            'max_risk_per_trade': 0.02,      # 2% max par trade
            'max_position_size': 0.20,        # 20% max du capital
            'max_daily_risk': 0.06,           # 6% max perdu par jour
            'max_drawdown': 0.15,             # Stop si -15% depuis peak
            
            # Stops & Targets
            'use_atr_stops': True,            # SL basé sur ATR (volatilité)
            'atr_multiplier_sl': 2.0,         # SL = 2× ATR
            'atr_multiplier_tp': 3.0,         # TP = 3× ATR (risk/reward 1.5)
            'min_risk_reward': 1.5,           # Ratio min RR
            
            # Limites globales
            'max_concurrent_trades': 3,       # Max 3 trades simultanés
            'max_correlated_trades': 2,       # Max 2 trades corrélés
            'min_time_between_trades': 60,    # 60 min entre trades
            
            # Kelly Criterion
            'use_kelly': False,               # Sizing Kelly (avancé)
            'kelly_fraction': 0.25,           # Kelly conservateur (25%)
        }
        
        # CORRECTION : Fusionne config custom avec defaults
        if config:
            self.config = {**default_config, **config}
        else:
            self.config = default_config
        
        self.capital = self.config['initial_capital']
        self.peak_capital = self.capital
        self.daily_pnl = 0
        self.daily_trades = []
        self.open_positions = []
        self.last_trade_time = None
    
    def calculate_position_size(self, entry_price, stop_loss, win_rate=None):
        """
        Calcule la taille de position optimale
        Méthodes: Fixed %, ATR-based, Kelly Criterion
        """
        # Méthode 1: Fixed Risk (2% du capital)
        risk_amount = self.capital * self.config['max_risk_per_trade']
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        
        # Limite par % capital max
        max_units = (self.capital * self.config['max_position_size']) / entry_price
        position_size = min(position_size, max_units)
        
        # Kelly Criterion (si activé et win_rate connu)
        if self.config['use_kelly'] and win_rate:
            kelly_size = self._kelly_criterion(win_rate, entry_price, stop_loss)
            position_size = min(position_size, kelly_size)
        
        return position_size
    
    def _kelly_criterion(self, win_rate, entry, sl, tp=None):
        """
        Kelly Criterion pour sizing optimal
        f* = (p × b - q) / b
        où p=win_rate, q=1-p, b=gain_moyen/perte_moyenne
        """
        if tp is None:
            tp = entry * 1.03  # Assume +3%
        
        p = win_rate  # Probabilité de gain
        q = 1 - p     # Probabilité de perte
        
        avg_win = abs(tp - entry)
        avg_loss = abs(entry - sl)
        
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss  # Ratio gain/perte
        
        kelly_pct = (p * b - q) / b
        kelly_pct = max(0, kelly_pct)  # Pas de short Kelly
        
        # Kelly conservateur (25% du Kelly)
        kelly_fraction = kelly_pct * self.config['kelly_fraction']
        
        position_size = (self.capital * kelly_fraction) / entry
        
        return position_size
    
    def calculate_atr_stops(self, df, current_price, atr_period=14):
        """
        Stop-loss et take-profit basés sur ATR (volatilité)
        Plus volatil = stops plus larges
        """
        # Calcule ATR
        high = df['high'].tail(atr_period)
        low = df['low'].tail(atr_period)
        close = df['close'].tail(atr_period)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.mean()
        
        # Calcule stops
        sl = current_price - (atr * self.config['atr_multiplier_sl'])
        tp = current_price + (atr * self.config['atr_multiplier_tp'])
        
        return sl, tp, atr
    
    def can_open_trade(self, signal_time=None):
        """
        Vérifie si un nouveau trade est autorisé
        """
        # Check nombre de trades ouverts
        if len(self.open_positions) >= self.config['max_concurrent_trades']:
            return False, "Max positions ouvertes atteint"
        
        # Check drawdown
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if current_drawdown >= self.config['max_drawdown']:
            return False, f"Max drawdown atteint ({current_drawdown:.1%})"
        
        # Check perte journalière
        if abs(self.daily_pnl) >= self.capital * self.config['max_daily_risk']:
            return False, "Limite quotidienne de perte atteinte"
        
        # Check temps entre trades
        if signal_time and self.last_trade_time:
            time_diff = (signal_time - self.last_trade_time).total_seconds() / 60
            if time_diff < self.config['min_time_between_trades']:
                return False, f"Attendre {self.config['min_time_between_trades']}min entre trades"
        
        return True, "OK"
    
    def validate_trade(self, entry, sl, tp):
        """
        Valide qu'un trade respecte les ratios risk/reward
        """
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        
        if risk == 0:
            return False, "Stop-loss invalide"
        
        risk_reward = reward / risk
        
        if risk_reward < self.config['min_risk_reward']:
            return False, f"Risk/Reward trop faible ({risk_reward:.2f} < {self.config['min_risk_reward']})"
        
        return True, f"RR: {risk_reward:.2f}"
    
    def update_capital(self, pnl, trade_time=None):
        """
        Met à jour le capital après un trade
        """
        self.capital += pnl
        self.daily_pnl += pnl
        
        # Update peak
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        if trade_time:
            self.last_trade_time = trade_time
    
    def add_position(self, position):
        """Ajoute une position ouverte"""
        self.open_positions.append(position)
    
    def close_position(self, position_id):
        """Ferme une position"""
        self.open_positions = [p for p in self.open_positions if p['id'] != position_id]
    
    def get_metrics(self):
        """Retourne les métriques de risque actuelles"""
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        
        return {
            'capital': self.capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': current_drawdown,
            'daily_pnl': self.daily_pnl,
            'open_positions': len(self.open_positions),
            'risk_used': len(self.open_positions) / self.config['max_concurrent_trades']
        }
    
    def reset_daily(self):
        """Reset les compteurs journaliers"""
        self.daily_pnl = 0
        self.daily_trades = []


# Configuration AGGRESSIVE (crypto)
AGGRESSIVE_CONFIG = {
    'initial_capital': 10000,
    'max_risk_per_trade': 0.10,      # 10% au lieu de 5%
    'max_position_size': 0.90,       # 90% au lieu de 50%
    'max_daily_risk': 0.50,          # 50% au lieu de 15%
    'max_drawdown': 0.80,            # 80% au lieu de 30%
    'max_concurrent_trades': 10,     # 10 au lieu de 5
    'min_risk_reward': 0.5,          # 0.5 au lieu de 1.2 ← CRITIQUE
}

# Configuration CONSERVATRICE (stock/fiat)
CONSERVATIVE_CONFIG = {
    'initial_capital': 10000,
    'max_risk_per_trade': 0.02,
    'max_position_size': 0.20,
    'max_daily_risk': 0.05,
    'max_drawdown': 0.20,
    'max_concurrent_trades': 5,
    'min_risk_reward': 0.5,  # ← BAISSÉ de 1.2 à 0.5
}
