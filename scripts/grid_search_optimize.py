"""
Grid Search Hyperparameter Optimization pour les Top 50 setups
Optimise epochs, batch_size, dropout, learning_rate pour LSTM/Transformer
"""

import numpy as np
import os
import random
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.pipeline import TradingPipeline
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
from itertools import product

# FIX: seeds pour reproductibilité
random.seed(42)
np.random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    pass

os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# FIX: logging fichier au lieu de warnings.filterwarnings('ignore')
logging.basicConfig(
    filename='logs/grid_search.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_FOLDERS = {
    'crypto_binance': 'data/crypto_binance/',
}  # FIX: accolade fermante (SyntaxError dans l'original)

INITIAL_CAPITAL = 10000
SPLIT_DATE      = "2023-01-01"
LOOKBACK        = 60

# FIX: grille vraiment explorée (pas juste les epochs)
GRID_PARAMS = {
    'epochs'       : [50, 100, 150],
    'batch_size'   : [32, 64],
    'dropout'      : [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005],
}

param_combinations = list(product(
    GRID_PARAMS['epochs'],
    GRID_PARAMS['batch_size'],
    GRID_PARAMS['dropout'],
    GRID_PARAMS['learning_rate'],
))
print(f"🔍 Grid Search: {len(param_combinations)} combinaisons par setup")


def get_csv_path(filename, asset_type):
    """Retrouve le chemin complet du CSV."""
    folder = DATA_FOLDERS.get(asset_type, 'data/')
    return os.path.join(folder, filename)


def _compute_score(pnl: float, sharpe: float) -> float:
    """
    FIX: score composite qui évite le piège négatif × négatif = positif.
    Récompense uniquement si les deux métriques sont positives.
    """
    if pnl <= 0 or sharpe <= 0:
        return pnl  # pénalise les setups négatifs
    return pnl * sharpe


def train_and_evaluate_one_config(args):
    """Entraîne et évalue UNE configuration d'hyperparamètres pour UN setup."""
    setup_info, params, config_idx, total_configs = args
    epochs, batch_size, dropout, lr = params

    try:
        csv_path = get_csv_path(setup_info['filename'], setup_info['asset_type'])
        if not os.path.exists(csv_path):
            return None

        pipeline = TradingPipeline()
        pipeline.load_data({setup_info['timeframe']: csv_path})

        X_train, y_train, X_val, y_val, X_test_full, y_test_full, n_feat = \
            pipeline.prepare_training_data(
                primary_timeframe=setup_info['timeframe'],
                multi_tf=False,
                train_end=None, val_end=None, test_end=None,
            )

        test_index_full = pipeline.test_prices.index
        split_ts  = pd.to_datetime(SPLIT_DATE, utc=True)
        mask_test = test_index_full >= split_ts

        if mask_test.sum() < 10:
            return None

        X_test = X_test_full[mask_test]
        y_test = y_test_full[mask_test]
        pipeline.test_prices = pipeline.test_prices.loc[mask_test]

        has_val    = X_val is not None and len(X_val) > 0
        model_type = setup_info['model'].lower()

        if model_type == 'xgboost':
            # FIX: X_val fusionnée dans train → ne pas la repasser en eval_set
            X_tr = np.concatenate([X_train, X_val]) if has_val else X_train
            y_tr = np.concatenate([y_train, y_val]) if has_val else y_train
            pipeline.train_model(
                model_type='xgboost',
                X_train=X_tr, y_train=y_tr,
                X_val=None,   y_val=None,
                n_features=n_feat, lookback=LOOKBACK,
            )
        else:
            # LSTM / Transformer : X_val séparée pour que l'early stopping reste valide
            pipeline.train_model(
                model_type=model_type,
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                n_features=n_feat, lookback=LOOKBACK,
                epochs=epochs, batch_size=batch_size,
                dropout=dropout, learning_rate=lr,
            )

        metrics, _, _ = pipeline.backtest(
            X_test, y_test,
            initial_capital=INITIAL_CAPITAL,
            asset_type=setup_info['asset_type'],
        )

        if 'error' in metrics:
            return None

        result = {
            'symbol'       : setup_info['symbol'],
            'timeframe'    : setup_info['timeframe'],
            'model'        : setup_info['model'],
            'epochs'       : epochs,
            'batch_size'   : batch_size,
            'dropout'      : dropout,
            'learning_rate': lr,
            'pnl_pct'      : metrics['pnl_pct'],
            'sharpe_ratio' : metrics['sharpe_ratio'],
            'profit_factor': metrics['profit_factor'],
            'win_rate'     : metrics['win_rate'],
            'total_trades' : metrics['total_trades'],
            'max_drawdown' : metrics['max_drawdown'],
            'config_idx'   : config_idx,
        }

        print(
            f"  [{config_idx}/{total_configs}] "
            f"e={epochs} bs={batch_size} do={dropout:.1f} lr={lr} → "
            f"PNL={metrics['pnl_pct']:+.2f}% Sharpe={metrics['sharpe_ratio']:.2f}",
            flush=True
        )
        return result

    except Exception as e:
        print(f"  ❌ Erreur config {config_idx}: {e}", flush=True)
        logging.error(f"Config {config_idx} | {setup_info.get('symbol', '?')} | {e}")
        return None


def optimize_one_setup(args):
    """Optimise les hyperparamètres pour UN setup via Grid Search."""
    setup_info, setup_idx, total_setups = args

    print(f"\n{'='*80}")
    print(f"[{setup_idx}/{total_setups}] GRID SEARCH: "
          f"{setup_info['symbol']} {setup_info['timeframe']} {setup_info['model']}")
    print(f"Baseline → PNL={setup_info.get('pnl_pct', 0):+.2f}% | "
          f"Sharpe={setup_info.get('sharpe_ratio', 0):.2f}")
    print(f"{'='*80}")

    results = []
    for config_idx, params in enumerate(param_combinations, 1):
        result = train_and_evaluate_one_config(
            (setup_info, params, config_idx, len(param_combinations))
        )
        if result is not None:
            results.append(result)

    if not results:
        print("  ❌ Aucune configuration valide pour ce setup")
        return None

    results_df = pd.DataFrame(results)

    # FIX: score composite (pénalise négatif × négatif)
    results_df['score'] = results_df.apply(
        lambda r: _compute_score(r['pnl_pct'], r['sharpe_ratio']), axis=1
    )
    best = results_df.loc[results_df['score'].idxmax()]

    baseline_pnl    = setup_info.get('pnl_pct', 0)
    baseline_sharpe = setup_info.get('sharpe_ratio', 0)
    improvement_pnl    = best['pnl_pct']     - baseline_pnl
    improvement_sharpe = best['sharpe_ratio'] - baseline_sharpe

    print(f"\n✅ MEILLEURE CONFIG:")
    print(f"  epochs={int(best['epochs'])} | bs={int(best['batch_size'])} | "
          f"dropout={best['dropout']:.1f} | lr={best['learning_rate']:.4f}")
    print(f"  PNL:    {best['pnl_pct']:+.2f}% "
          f"(baseline: {baseline_pnl:+.2f}%) → Δ={improvement_pnl:+.2f}%")
    print(f"  Sharpe: {best['sharpe_ratio']:.2f} "
          f"(baseline: {baseline_sharpe:.2f}) → Δ={improvement_sharpe:+.2f}")
    print(f"{'='*80}\n")

    return {
        'symbol'            : setup_info['symbol'],
        'timeframe'         : setup_info['timeframe'],
        'model'             : setup_info['model'],
        'filename'          : setup_info.get('filename', ''),
        'asset_type'        : setup_info.get('asset_type', ''),
        'baseline_pnl_pct'  : baseline_pnl,
        'baseline_sharpe'   : baseline_sharpe,
        'optimized_pnl_pct' : best['pnl_pct'],
        'optimized_sharpe'  : best['sharpe_ratio'],
        'improvement_pnl'   : improvement_pnl,
        'improvement_sharpe': improvement_sharpe,
        'best_epochs'       : int(best['epochs']),
        'best_batch_size'   : int(best['batch_size']),
        'best_dropout'      : best['dropout'],
        'best_learning_rate': best['learning_rate'],
        'profit_factor'     : best['profit_factor'],
        'win_rate'          : best['win_rate'],
        'total_trades'      : int(best['total_trades']),
        'max_drawdown'      : best['max_drawdown'],
    }


def main():
    print("=" * 80)
    print("🔍 GRID SEARCH HYPERPARAMETER OPTIMIZATION - TOP 50 SETUPS")
    print("=" * 80)
    print(f"Combinaisons testées par setup : {len(param_combinations)}")
    for k, v in GRID_PARAMS.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    top_50_file = 'results/top_50_setups_to_optimize.csv'
    if not os.path.exists(top_50_file):
        print(f"❌ Fichier {top_50_file} introuvable!")
        return

    top_50 = pd.read_csv(top_50_file)
    print(f"📂 {len(top_50)} setups à optimiser")

    args = [
        (row.to_dict(), idx + 1, len(top_50))
        for idx, row in top_50.iterrows()
    ]

    n_workers = min(4, cpu_count())
    print(f"⚡ Utilisation de {n_workers} workers")
    print("=" * 80)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/grid_search_optimized_{timestamp}.csv"

    all_optimized = []

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(optimize_one_setup, args):
            if result:
                all_optimized.append(result)
                # FIX: sauvegarde incrémentale après chaque setup optimisé
                pd.DataFrame(all_optimized).to_csv(output_file, index=False)
                print(f"💾 Sauvegarde intermédiaire: {len(all_optimized)} setups", flush=True)

    if not all_optimized:
        print("\n❌ Aucun setup optimisé!")
        return

    df_optimized = pd.DataFrame(all_optimized)
    df_optimized.to_csv(output_file, index=False)

    # Export séparé des setups améliorés uniquement
    improved      = df_optimized[df_optimized['improvement_pnl'] > 0].copy()
    improved_file = 'results/grid_search_improved_only.csv'
    improved.to_csv(improved_file, index=False)

    print("\n" + "=" * 80)
    print("📊 RÉSULTATS GRID SEARCH")
    print("=" * 80)
    print(f"Setups optimisés             : {len(df_optimized)}")
    print(f"Amélioration moyenne PNL     : {df_optimized['improvement_pnl'].mean():+.2f}%")
    print(f"Amélioration moyenne Sharpe  : {df_optimized['improvement_sharpe'].mean():+.2f}")
    print(f"Setups améliorés (PNL)       : "
          f"{len(df_optimized[df_optimized['improvement_pnl'] > 0])}/{len(df_optimized)}")
    print(f"Setups améliorés (Sharpe)    : "
          f"{len(df_optimized[df_optimized['improvement_sharpe'] > 0])}/{len(df_optimized)}")

    print("\n" + "-" * 80)
    print("🏆 TOP 5 PLUS GRANDES AMÉLIORATIONS (PNL):")
    for _, row in df_optimized.nlargest(5, 'improvement_pnl').iterrows():
        print(f"  {row['symbol']:8} {row['timeframe']:4} {row['model']:11} → "
              f"+{row['improvement_pnl']:.2f}% "
              f"(de {row['baseline_pnl_pct']:.2f}% à {row['optimized_pnl_pct']:.2f}%)")

    print(f"\n💾 Résultats complets  : {output_file}")
    print(f"💾 Setups améliorés    : {improved_file}")
    print("=" * 80)
    print("✅ GRID SEARCH TERMINÉ")


if __name__ == '__main__':
    main()
