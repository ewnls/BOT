"""
Analyse massive sur tout le dataset - XGBoost uniquement
Train < 2023, backtest sur 2023+ pour tous les fichiers
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

from pipeline import TradingPipeline
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Configuration
DATA_FOLDERS = {
    'crypto_binance': 'data/crypto_binance/',
}

MODELS = ['xgboost']
INITIAL_CAPITAL = 10000

SPLIT_DATE = "2023-01-01"  # tout avant = train, tout après = test


def get_timeframe_from_filename(filename: str) -> str:
    for tf in ['1w', '1d', '12h', '6h', '4h', '2h', '1h', '30m', '15m', '5m', '1m']:
        if tf in filename.lower():
            return tf
    return '1d'


def scan_all_files():
    all_files = []
    for asset_type, folder in DATA_FOLDERS.items():
        if not os.path.exists(folder):
            print(f"⚠️ Dossier inexistant: {folder}")
            continue
        for file in Path(folder).glob('*.csv'):
            timeframe = get_timeframe_from_filename(file.name)
            all_files.append({
                'filepath': str(file),
                'filename': file.name,
                'asset_type': asset_type,
                'timeframe': timeframe,
                'symbol': file.stem.replace(f'_{timeframe}', '')
            })
    print(f"📂 {len(all_files)} fichiers CSV détectés")
    return all_files


def train_and_backtest_one(args):
    """
    1 fichier, XGBoost uniquement.
    Train sur toute l'histoire avant SPLIT_DATE, test sur SPLIT_DATE+.
    """
    file_info, index, total = args
    results = []

    try:
        print(f"[{index}/{total}] {file_info['filename']} ({file_info['timeframe']}) [{file_info['asset_type']}]")

        pipeline = TradingPipeline()
        pipeline.load_data({file_info['timeframe']: file_info['filepath']})

        # On laisse pipeline.prepare_training_data faire un split par ratios
        # puis on recoupe manuellement train/test par dates sur la partie test.
        try:
            X_train, y_train, X_val, y_val, X_test_full, y_test_full, n_feat = pipeline.prepare_training_data(
                primary_timeframe=file_info['timeframe'],
                multi_tf=False,
                train_end=None,
                val_end=None,
                test_end=None,
            )
        except ValueError as e:
            print(f"   ❌ Erreur préparation données: {e}")
            return results

        # Index correspondant à X_test_full / y_test_full = pipeline.test_prices.index
        test_index_full = pipeline.test_prices.index

        split_ts = pd.to_datetime(SPLIT_DATE, utc=True)
        mask_test = test_index_full >= split_ts

        if mask_test.sum() < 10:
            print("   ⚠️ Trop peu de données en test après 2023-01-01, setup ignoré.")
            return results

        X_test = X_test_full[mask_test]
        y_test = y_test_full[mask_test]
        pipeline.test_prices = pipeline.test_prices.loc[mask_test]

        for model_type in MODELS:
            try:
                print(f"   → {model_type.upper()}...", flush=True)

                # Entraînement sur tout X_train+X_val (on concatène pour utiliser tout < 2023)
                X_train_all = X_train
                y_train_all = y_train
                if X_val is not None and len(X_val) > 0:
                    X_train_all = np.concatenate([X_train, X_val], axis=0)
                    y_train_all = np.concatenate([y_train, y_val], axis=0)

                pipeline.train_model(
                    model_type=model_type,
                    X_train=X_train_all,
                    y_train=y_train_all,
                    X_val=X_val,   # utilisé pour early stopping si le modèle le supporte
                    y_val=y_val,
                    n_features=n_feat,
                )

                metrics, trades, equity_series = pipeline.backtest(
                    X_test,
                    y_test,
                    initial_capital=INITIAL_CAPITAL,
                    asset_type=file_info['asset_type']
                )

                if 'error' in metrics:
                    print(f"      ❌ {model_type.upper()}: erreur metrics", flush=True)
                    continue

                metrics['filename'] = file_info['filename']
                metrics['symbol'] = file_info['symbol']
                metrics['timeframe'] = file_info['timeframe']
                metrics['asset_type'] = file_info['asset_type']
                metrics['model'] = model_type.upper()

                print(f"      ✅ {metrics['total_trades']} trades | PNL: {metrics['pnl_pct']:+.1f}% | Sharpe: {metrics['sharpe_ratio']:.2f}", flush=True)
                results.append(metrics)

            except Exception as e:
                print(f"      ❌ Erreur {model_type.upper()} sur {file_info['filename']}: {e}", flush=True)
                continue

    except Exception as e:
        print(f"   ❌ Erreur globale sur {file_info['filename']}: {e}", flush=True)

    return results


def main():
    print("="*80)
    print("🚀 RÉAPPRENTISSAGE COMPLET - XGBOOST - TRAIN<2023 / TEST≥2023 (split manuel)")
    print("="*80)
    print(f"Capital initial: ${INITIAL_CAPITAL}")
    print(f"Modèles: {', '.join([m.upper() for m in MODELS])}")
    print("="*80)

    all_files = scan_all_files()
    if len(all_files) == 0:
        print("❌ Aucun fichier trouvé!")
        return

    total_files = len(all_files)

    max_workers = 7
    n_workers = min(max_workers, cpu_count())
    print(f"⚡ Utilisation de {n_workers} workers (CPU logiques: {cpu_count()})")
    print("="*80)

    args = [(file_info, i + 1, total_files) for i, file_info in enumerate(all_files)]

    all_results = []

    with Pool(n_workers) as pool:
        for worker_result in pool.imap_unordered(train_and_backtest_one, args):
            if worker_result:
                all_results.extend(worker_result)
            print(f"📈 Résultats accumulés: {len(all_results)}", flush=True)

    if len(all_results) == 0:
        print("\n❌ Aucun résultat à analyser!")
        return

    df = pd.DataFrame(all_results)
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/analysis_retrain_2023plus_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Résultats sauvegardés: {output_file}")
    print("\n✅ ANALYSE TERMINÉE")


if __name__ == '__main__':
    main()
