"""
Walk-Forward Validation sur les meilleurs setups optimisés
Teste la robustesse sur 4 périodes temporelles différentes
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

random.seed(42)
np.random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    pass

os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/walk_forward.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ── Configuration ────────────────────────────────────────────────────────────
DATA_FOLDERS = {
    'crypto_binance': 'data/crypto_binance/',
}  # FIX: accolade fermante

INITIAL_CAPITAL = 10000
LOOKBACK        = 60

# FIX: val_end ajouté pour split propre sans data leakage
# train_end → val_end → test_start → test_end (fenêtres non chevauchantes)
PERIODS = [
    {
        'name'      : 'Period_1',
        'train_end' : '2021-06-30',
        'val_end'   : '2021-12-31',
        'test_start': '2022-01-01',
        'test_end'  : '2022-12-31',
    },
    {
        'name'      : 'Period_2',
        'train_end' : '2022-06-30',
        'val_end'   : '2022-12-31',
        'test_start': '2023-01-01',
        'test_end'  : '2023-12-31',
    },
    {
        'name'      : 'Period_3',
        'train_end' : '2023-06-30',
        'val_end'   : '2023-12-31',
        'test_start': '2024-01-01',
        'test_end'  : '2024-12-31',
    },
    {
        'name'      : 'Period_4',
        'train_end' : '2024-06-30',
        'val_end'   : '2024-12-31',
        'test_start': '2025-01-01',
        'test_end'  : '2025-12-31',
    },
]


def get_csv_path(filename, asset_type):
    folder = DATA_FOLDERS.get(asset_type, 'data/')
    return os.path.join(folder, filename)


def validate_one_setup_one_period(setup_info, period):
    """
    Valide UN setup sur UNE période.
    FIX: split par dates pour garantir qu'aucune donnée future
         ne se retrouve dans le train (leakage supprimé).
    """
    try:
        csv_path = get_csv_path(setup_info['filename'], setup_info['asset_type'])
        if not os.path.exists(csv_path):
            return None

        pipeline = TradingPipeline()
        pipeline.load_data({setup_info['timeframe']: csv_path})

        # FIX: split par dates (train_end / val_end / test_end) pour chaque période
        try:
            X_train, y_train, X_val, y_val, X_test, y_test, n_feat = \
                pipeline.prepare_training_data(
                    primary_timeframe=setup_info['timeframe'],
                    multi_tf=False,
                    train_end=period['train_end'],
                    val_end=period['val_end'],
                    test_end=period['test_end'],
                )
        except ValueError as e:
            logging.warning(f"Split vide pour {setup_info['symbol']} {period['name']}: {e}")
            return None

        # Filtre test à partir de test_start
        test_index  = pipeline.test_prices.index
        test_start  = pd.to_datetime(period['test_start'], utc=True)
        mask_test   = test_index >= test_start

        if mask_test.sum() < 10:
            return None

        X_test = X_test[mask_test]
        y_test = y_test[mask_test]
        pipeline.test_prices = pipeline.test_prices.loc[mask_test]

        # FIX: mask_train supprimé (dead code – le split par date s'en charge)

        pipeline.train_model(
            model_type=setup_info['model'].lower(),
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            n_features=n_feat,
            lookback=LOOKBACK,
            epochs=int(setup_info.get('best_epochs', 50)),
            batch_size=int(setup_info.get('best_batch_size', 32)),
            dropout=setup_info.get('best_dropout', 0.2),
            learning_rate=setup_info.get('best_learning_rate', 0.001),
        )

        metrics, _, _ = pipeline.backtest(
            X_test, y_test,
            initial_capital=INITIAL_CAPITAL,
            asset_type=setup_info['asset_type'],
        )

        if 'error' in metrics:
            return None

        return {
            'symbol'       : setup_info['symbol'],
            'timeframe'    : setup_info['timeframe'],
            'model'        : setup_info['model'],
            'period'       : period['name'],
            'test_start'   : period['test_start'],
            'test_end'     : period['test_end'],
            'pnl_pct'      : metrics['pnl_pct'],
            'sharpe_ratio' : metrics['sharpe_ratio'],
            'profit_factor': metrics['profit_factor'],
            'win_rate'     : metrics['win_rate'],
            'total_trades' : metrics['total_trades'],
            'max_drawdown' : metrics['max_drawdown'],
            'best_epochs'  : int(setup_info.get('best_epochs', 50)),
        }

    except Exception as e:
        print(f"  ❌ Erreur {period['name']}: {e}")
        logging.error(f"{setup_info['symbol']} {period['name']}: {e}")
        return None


def validate_one_setup(args):
    """Valide UN setup sur TOUTES les périodes Walk-Forward."""
    setup_info, setup_idx, total_setups, output_file_agg, output_file_det = args

    print(f"\n{'='*80}")
    print(f"[{setup_idx}/{total_setups}] {setup_info['symbol']} "
          f"{setup_info['timeframe']} - {setup_info['model']}")
    print(f"Optimized: PNL={setup_info.get('optimized_pnl_pct', 0):+.2f}% | "
          f"Sharpe={setup_info.get('optimized_sharpe', 0):.2f}")
    print(f"Best params: epochs={int(setup_info.get('best_epochs', 50))} | "
          f"bs={int(setup_info.get('best_batch_size', 32))}")
    print(f"{'='*80}")

    results = []
    for period in PERIODS:
        print(f"  → Testing {period['name']} "
              f"({period['test_start']} to {period['test_end']})...", flush=True)
        result = validate_one_setup_one_period(setup_info, period)
        if result:
            results.append(result)
            print(f"  ✅ PNL={result['pnl_pct']:+.2f}% | "
                  f"Sharpe={result['sharpe_ratio']:.2f} | "
                  f"Trades={result['total_trades']}", flush=True)
        else:
            print("  ⚠️ Pas assez de données pour cette période", flush=True)

    if not results:
        return None

    results_df = pd.DataFrame(results)
    aggregated = {
        'symbol'            : setup_info['symbol'],
        'timeframe'         : setup_info['timeframe'],
        'model'             : setup_info['model'],
        'filename'          : setup_info.get('filename', ''),
        'asset_type'        : setup_info.get('asset_type', ''),
        'optimized_pnl_pct' : setup_info.get('optimized_pnl_pct', 0),
        'optimized_sharpe'  : setup_info.get('optimized_sharpe', 0),
        'best_epochs'       : int(setup_info.get('best_epochs', 50)),
        'best_batch_size'   : int(setup_info.get('best_batch_size', 32)),
        'best_dropout'      : setup_info.get('best_dropout', 0.2),
        'best_learning_rate': setup_info.get('best_learning_rate', 0.001),
        'wf_avg_pnl'        : results_df['pnl_pct'].mean(),
        'wf_median_pnl'     : results_df['pnl_pct'].median(),
        'wf_std_pnl'        : results_df['pnl_pct'].std(),
        'wf_min_pnl'        : results_df['pnl_pct'].min(),
        'wf_max_pnl'        : results_df['pnl_pct'].max(),
        'wf_avg_sharpe'     : results_df['sharpe_ratio'].mean(),
        'wf_positive_periods': int((results_df['pnl_pct'] > 0).sum()),
        'wf_total_periods'  : len(results_df),
        'wf_consistency'    : (results_df['pnl_pct'] > 0).mean() * 100,
    }

    print(f"\n📊 RÉSUMÉ:")
    print(f"  Moyenne PNL: {aggregated['wf_avg_pnl']:+.2f}% "
          f"(médiane: {aggregated['wf_median_pnl']:+.2f}%)")
    print(f"  Périodes positives: {aggregated['wf_positive_periods']}/"
          f"{aggregated['wf_total_periods']} "
          f"({aggregated['wf_consistency']:.0f}%)")

    # FIX: sauvegarde intermédiaire
    _append_to_csv(aggregated, output_file_agg)
    for r in results:
        _append_to_csv(r, output_file_det)

    return aggregated, results


def _append_to_csv(row: dict, filepath: str):
    """Sauvegarde incrémentale (append)."""
    df = pd.DataFrame([row])
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', header=header, index=False)


def main():
    print("=" * 80)
    print("🔄 WALK-FORWARD VALIDATION - TOP SETUPS OPTIMISÉS")
    print("=" * 80)
    print(f"Périodes testées: {len(PERIODS)}")
    for p in PERIODS:
        print(f"  • {p['name']}: {p['test_start']} → {p['test_end']}")
    print("=" * 80)

    improved_file = 'results/grid_search_improved_only.csv'
    if not os.path.exists(improved_file):
        print(f"⚠️ {improved_file} introuvable, recherche alternative...")
        candidates = sorted(Path('results').glob('grid_search_optimized_*.csv'), reverse=True)
        if not candidates:
            print("❌ Aucun fichier de résultats grid search trouvé!")
            return
        improved_file = str(candidates[0])
        print(f"  → Utilisation de: {improved_file}")

    df = pd.read_csv(improved_file)
    top_10 = df.nlargest(10, 'optimized_pnl_pct')
    print(f"\n📂 {len(top_10)} setups à valider (Top 10)")
    print("=" * 80)

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_agg = f"results/walk_forward_aggregated_{timestamp}.csv"
    output_file_det = f"results/walk_forward_detailed_{timestamp}.csv"

    args = [
        (row.to_dict(), idx + 1, len(top_10), output_file_agg, output_file_det)
        for idx, row in top_10.iterrows()
    ]

    all_aggregated = []
    all_detailed   = []

    for arg in args:
        result = validate_one_setup(arg)
        if result:
            aggregated, detailed = result
            all_aggregated.append(aggregated)
            all_detailed.extend(detailed)

    if not all_aggregated:
        print("\n❌ Aucun setup validé!")
        return

    df_aggregated = pd.DataFrame(all_aggregated)
    df_detailed   = pd.DataFrame(all_detailed)
    df_aggregated.to_csv(output_file_agg, index=False)
    df_detailed.to_csv(output_file_det, index=False)

    robust = df_aggregated[df_aggregated['wf_consistency'] >= 75.0]
    print(f"\n✅ Setups robustes (>=75% périodes positives): {len(robust)}/{len(df_aggregated)}")
    if len(robust) > 0:
        print("\n🏆 SETUPS ROBUSTES:")
        for _, row in robust.iterrows():
            print(f"  {row['symbol']:10} {row['timeframe']:4} {row['model']:11} → "
                  f"Avg PNL={row['wf_avg_pnl']:+.2f}% | "
                  f"Consistency={row['wf_consistency']:.0f}% | "
                  f"Sharpe={row['wf_avg_sharpe']:.2f}")
        robust_file = f"results/walk_forward_robust_only_{timestamp}.csv"
        robust.to_csv(robust_file, index=False)
        print(f"\n💾 Setups robustes: {robust_file}")

    print(f"\n💾 Résultats agrégés : {output_file_agg}")
    print(f"💾 Résultats détaillés: {output_file_det}")
    print("=" * 80)
    print("✅ WALK-FORWARD VALIDATION TERMINÉE")


if __name__ == '__main__':
    main()
