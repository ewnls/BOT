# ML Crypto Trading Bot (XGBoost)

Bot de trading algorithmique pour cryptos (et potentiellement stocks/fiat), basé sur du Machine Learning (XGBoost) et un backtester avec gestion du risque intégrée.

## 1. Objectif du projet

Construire un bot de trading **robuste** sur plusieurs symboles / timeframes, capable de :

- Entraîner des modèles ML (XGBoost, LSTM, Transformer disponibles mais focus actuel sur XGBoost).
- Gérer un **pipeline unifié** : chargement des données, features, split temporel, training, backtest.
- Intégrer un **RiskManager** (taille de position, drawdown max, risk/reward).
- Produire des **backtests réalistes** sur différentes périodes de marché (bull/bear, 2017→2026, 2023→2026, etc.).
- Générer des **courbes d’equity** et des fichiers d’analyse (Sharpe, drawdown, win rate, etc.).

Le dataset actuel fait ~5 Go (220 fichiers CSV, ~93M de lignes) avec de l’historique crypto depuis 2017 (multi‑timeframes de 1m à 1w).

## 2. Structure du projet

Racine du repo (simplifié) :

```text
.
├── pipeline.py                  # Pipeline principal ML + backtesting
├── models/
│   └── ml_models.py             # XGBoostModel (+ LSTM/Transformer disponibles)
├── utils/
│   ├── preprocessing.py         # DataPreprocessor, DataPreprocessorSimple
│   ├── backtesting.py           # Backtester + intégration RiskManager
│   └── risk_manager.py          # RiskManager, configs de risque
├── data/
│   └── crypto_binance/          # CSV OHLCV (ex: BTCUSDT_1d.csv, ETHUSDT_1h.csv, etc.)
├── results/
│   ├── analysis_*.csv           # Résultats d’analyse massive (multi fichiers)
│   └── equity_*.csv             # Courbes d’equity
├── scripts/
│   └── analyze_all_5go.py       # Analyse massive / réapprentissage (train<2023, test≥2023)
├── README.md
└── requirements.txt (optionnel)
