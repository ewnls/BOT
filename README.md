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
BOT/
├── src/
│   ├── pipeline.py                  # Pipeline principal ML + backtesting
│   ├── models/
│   │   └── ml_models.py             # XGBoostModel, LSTMModel, TransformerModel
│   └── utils/
│       ├── preprocessing.py         # DataPreprocessor, DataPreprocessorSimple
│       ├── backtesting.py           # Backtester + intégration RiskManager
│       └── risk_manager.py          # RiskManager, configs de risque
├── data/
│   └── crypto_binance/              # CSV OHLCV (ex: BTCUSDT_1d.csv, ETHUSDT_1h.csv)
├── results/
│   ├── analysis_*.csv               # Résultats d'analyse massive
│   ├── top_50_setups_to_optimize.csv
│   └── grid_search_optimized_*.csv  # (futur) résultats grid search
├── scripts/
│   ├── analyze_all_DataSet.py       # Analyse massive (train<2023, test≥2023)
│   └── grid_search_optimize.py      # Grid Search hyperparameter optimization
├── README.md
└── requirements.txt
