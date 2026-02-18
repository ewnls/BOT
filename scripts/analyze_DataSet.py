"""
Analyse massive sur tout le dataset - XGBoost + LSTM + Transformer
Train < 2023, backtest sur 2023+ pour tous les fichiers
LSTM/Transformer uniquement sur timeframes >= 4h
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

# Seeds pour reproductibilité
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
    filename='logs/analyze_dataset.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ── Configuration ────────────────────────────────────────────────────────────
DATA_FOLDERS = {
    'crypto_binance': 'data/crypto_binance/',
}  # FIX: accolade fermante ajoutée

INITIAL_CAPITAL = 10000
SPLIT_DATE      = "2023-01-01"
EPOCHS          = 50
BATCH_SIZE      = 32
LOOKBACK        = 60
TF_LONG         = ['4h', '6h', '12h', '1d', '1w']


def get_timeframe_from_filename(filename: str) -> str:
    for tf in ['1w', '1d', '12h', '6h', '4h', '2h', '1h', '30m', '15m', '5m', '1m']:
        if tf in filename.lower():
            return tf
    # FIX: log au lieu de rater silencieusement
    logging.warning(f"Timeframe inconnu dans '{filename}', fallback '1d'")
    print(f"⚠️ Timeframe inconnu dans '{filename}', fallback '1d'")
    return '1d'


def scan_all_files():
    all_files = []
    for asset_type, folder in DATA_FOLDERS.items():
        if not os.path.exists(folder):
            print(f"⚠️ Dossier inexistant: {folder}")
            continue
        for file in Path(folder).glob('*.csv'):
            timeframe = get_timeframe_from_filename(file.name)
            # FIX: rsplit pour éviter les doublons de timeframe dans le nom
            symbol = file.stem.rsplit(f'_{timeframe}', 1)[0]
            all_files.append({
                'filepath' : str(file),
                'filename' : file.name,
                'asset_type': asset_type,
                'timeframe' : timeframe,
                'symbol'   : symbol,
            })
    print(f"📂 {len(all_files)} fichiers CSV détectés")
    return all_files


def _save_partial(results: list, output_file: str):
    """Sauvegarde incrémentale après chaque résultat."""
    if results:
        pd.DataFrame(results).to_csv(output_file, index=False)


def train_and_backtest_one(args):
    """
    1 fichier, modèles selon timeframe :
    - TF courts (5m-2h) : XGBoost seul
    - TF longs (>=4h)   : XGBoost + LSTM + Transformer
    """
    file_info, index, total, output_file = args
    results = []

    models_to_test = (['xgboost', 'lstm', 'transformer']
                      if file_info['timeframe'] in TF_LONG else ['xgboost'])
    suffix = " [3 modèles]" if file_info['timeframe'] in TF_LONG else " [XGBoost seul]"

    try:
        print(f"[{index}/{total}] {file_info['filename']} "
              f"({file_info['timeframe']}) [{file_info['asset_type']}]{suffix}")

        pipeline_data = TradingPipeline()
        pipeline_data.load_data({file_info['timeframe']: file_info['filepath']})

        try:
            X_train, y_train, X_val, y_val, X_test_full, y_test_full, n_feat = \
                pipeline_data.prepare_training_data(
                    primary_timeframe=file_info['timeframe'],
                    multi_tf=False,
                    train_end=None, val_end=None, test_end=None,
                )
        except ValueError as e:
            print(f"  ❌ Erreur préparation données: {e}")
            return results

        test_index_full = pipeline_data.test_prices.index
        split_ts  = pd.to_datetime(SPLIT_DATE, utc=True)
        mask_test = test_index_full >= split_ts

        if mask_test.sum() < 10:
            print("  ⚠️ Trop peu de données en test après 2023-01-01, ignoré.")
            return results

        X_test             = X_test_full[mask_test]
        y_test             = y_test_full[mask_test]
        test_prices_filtered = pipeline_data.test_prices.loc[mask_test]

        # Train complet pour XGBoost (pas d'early stopping avec val leaké)
        has_val = X_val is not None and len(X_val) > 0
        X_train_all = np.concatenate([X_train, X_val]) if has_val else X_train
        y_train_all = np.concatenate([y_train, y_val]) if has_val else y_train

        for model_type in models_to_test:
            try:
                print(f"  → {model_type.upper()}...", flush=True)

                pipeline = TradingPipeline()
                pipeline.load_data({file_info['timeframe']: file_info['filepath']})
                pipeline.scaler_X        = pipeline_data.scaler_X
                pipeline.scaler_y        = pipeline_data.scaler_y
                pipeline.test_prices     = test_prices_filtered.copy()
                pipeline.primary_timeframe = file_info['timeframe']

                if model_type == 'xgboost':
                    # FIX: X_val déjà dans X_train_all → ne pas la repasser en eval_set
                    pipeline.train_model(
                        model_type='xgboost',
                        X_train=X_train_all, y_train=y_train_all,
                        X_val=None, y_val=None,
                        n_features=n_feat, lookback=LOOKBACK,
                    )
                else:
                    # LSTM / Transformer : X_val NON incluse dans X_train_all
                    # pour que l'early stopping reste valide
                    pipeline.train_model(
                        model_type=model_type,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val,     y_val=y_val,
                        n_features=n_feat, lookback=LOOKBACK,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                    )

                metrics, _, _ = pipeline.backtest(
                    X_test, y_test,
                    initial_capital=INITIAL_CAPITAL,
                    asset_type=file_info['asset_type'],
                )

                if 'error' in metrics:
                    print(f"  ❌ {model_type.upper()}: erreur metrics", flush=True)
                    continue

                metrics.update({
                    'filename'  : file_info['filename'],
                    'symbol'    : file_info['symbol'],
                    'timeframe' : file_info['timeframe'],
                    'asset_type': file_info['asset_type'],
                    'model'     : model_type.upper(),
                })

                print(f"  ✅ {metrics['total_trades']} trades | "
                      f"PNL: {metrics['pnl_pct']:+.1f}% | "
                      f"Sharpe: {metrics['sharpe_ratio']:.2f}", flush=True)

                results.append(metrics)
                # FIX: sauvegarde incrémentale
                _save_partial(results, output_file)

            except Exception as e:
                print(f"  ❌ Erreur {model_type.upper()} sur {file_info['filename']}: {e}", flush=True)
                logging.error(f"{model_type.upper()} | {file_info['filename']} | {e}")
                continue

    except Exception as e:
        print(f"  ❌ Erreur globale sur {file_info['filename']}: {e}", flush=True)
        logging.error(f"Global | {file_info['filename']} | {e}")

    return results


def _run_xgboost_only(args):
    """Worker XGBoost pur (CPU) — safe pour multiprocessing."""
    file_info, index, total, output_file = args
    results = []

    print(f"[{index}/{total}] XGB {file_info['filename']}", flush=True)
    try:
        pipeline_data = TradingPipeline()
        pipeline_data.load_data({file_info['timeframe']: file_info['filepath']})

        try:
            X_train, y_train, X_val, y_val, X_test_full, y_test_full, n_feat = \
                pipeline_data.prepare_training_data(
                    primary_timeframe=file_info['timeframe'],
                    multi_tf=False,
                )
        except ValueError as e:
            print(f"  ❌ Prep: {e}")
            return results

        split_ts  = pd.to_datetime(SPLIT_DATE, utc=True)
        mask_test = pipeline_data.test_prices.index >= split_ts
        if mask_test.sum() < 10:
            return results

        X_test   = X_test_full[mask_test]
        y_test   = y_test_full[mask_test]
        has_val  = X_val is not None and len(X_val) > 0
        X_tr     = np.concatenate([X_train, X_val]) if has_val else X_train
        y_tr     = np.concatenate([y_train, y_val]) if has_val else y_train

        pipeline = TradingPipeline()
        pipeline.load_data({file_info['timeframe']: file_info['filepath']})
        pipeline.scaler_X          = pipeline_data.scaler_X
        pipeline.scaler_y          = pipeline_data.scaler_y
        pipeline.test_prices       = pipeline_data.test_prices.loc[mask_test].copy()
        pipeline.primary_timeframe = file_info['timeframe']

        pipeline.train_model(
            model_type='xgboost',
            X_train=X_tr, y_train=y_tr,
            X_val=None,   y_val=None,
            n_features=n_feat, lookback=LOOKBACK,
        )
        metrics, _, _ = pipeline.backtest(
            X_test, y_test,
            initial_capital=INITIAL_CAPITAL,
            asset_type=file_info['asset_type'],
        )
        if 'error' not in metrics:
            metrics.update({
                'filename'  : file_info['filename'],
                'symbol'    : file_info['symbol'],
                'timeframe' : file_info['timeframe'],
                'asset_type': file_info['asset_type'],
                'model'     : 'XGBOOST',
            })
            results.append(metrics)
            _save_partial(results, output_file)
            print(f"  ✅ {metrics['total_trades']} trades | "
                  f"PNL: {metrics['pnl_pct']:+.1f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f}", flush=True)
    except Exception as e:
        print(f"  ❌ {e}", flush=True)
        logging.error(f"{file_info['filename']} XGB: {e}")

    return results


def _run_deep_sequential(file_info, index, total, output_file):
    """
    LSTM + Transformer en séquentiel sur GPU DirectML.
    NE PAS mettre dans un Pool → TF/PyTorch + fork = crash.
    """
    results = []
    for model_type in ['lstm', 'transformer']:
        print(f"[{index}/{total}] {model_type.upper()} {file_info['filename']}", flush=True)
        try:
            pipeline_data = TradingPipeline()
            pipeline_data.load_data({file_info['timeframe']: file_info['filepath']})

            try:
                X_train, y_train, X_val, y_val, X_test_full, y_test_full, n_feat = \
                    pipeline_data.prepare_training_data(
                        primary_timeframe=file_info['timeframe'],
                        multi_tf=False,
                    )
            except ValueError as e:
                print(f"  ❌ Prep: {e}")
                continue

            split_ts  = pd.to_datetime(SPLIT_DATE, utc=True)
            mask_test = pipeline_data.test_prices.index >= split_ts
            if mask_test.sum() < 10:
                continue

            X_test = X_test_full[mask_test]
            y_test = y_test_full[mask_test]

            pipeline = TradingPipeline()
            pipeline.load_data({file_info['timeframe']: file_info['filepath']})
            pipeline.scaler_X          = pipeline_data.scaler_X
            pipeline.scaler_y          = pipeline_data.scaler_y
            pipeline.test_prices       = pipeline_data.test_prices.loc[mask_test].copy()
            pipeline.primary_timeframe = file_info['timeframe']

            pipeline.train_model(
                model_type=model_type,
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                n_features=n_feat, lookback=LOOKBACK,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
            )
            metrics, _, _ = pipeline.backtest(
                X_test, y_test,
                initial_capital=INITIAL_CAPITAL,
                asset_type=file_info['asset_type'],
            )
            if 'error' not in metrics:
                metrics.update({
                    'filename'  : file_info['filename'],
                    'symbol'    : file_info['symbol'],
                    'timeframe' : file_info['timeframe'],
                    'asset_type': file_info['asset_type'],
                    'model'     : model_type.upper(),
                })
                results.append(metrics)
                _save_partial(results, output_file)
                print(f"  ✅ {metrics['total_trades']} trades | "
                      f"PNL: {metrics['pnl_pct']:+.1f}% | "
                      f"Sharpe: {metrics['sharpe_ratio']:.2f}", flush=True)
        except Exception as e:
            print(f"  ❌ {model_type.upper()}: {e}", flush=True)
            logging.error(f"{file_info['filename']} {model_type}: {e}")

    return results


def main():
    all_files   = scan_all_files()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/analyze_dataset_{timestamp}.csv"
    all_results = []

    # ── Sépare les fichiers par type de modèle ────────────────────────────────
    files_xgb   = [f for f in all_files if f['timeframe'] not in TF_LONG]
    files_deep  = [f for f in all_files if f['timeframe'] in TF_LONG]
    total       = len(all_files)

    print(f"\n{'='*60}")
    print(f"📂 {total} fichiers | "
          f"{len(files_xgb)} XGBoost CPU | "
          f"{len(files_deep)} Deep GPU")
    print(f"{'='*60}\n")

    # ── Phase 1 : XGBoost en parallèle sur tous les cores ────────────────────
    if files_xgb:
        # Laisse 2 threads au système
        n_workers = max(1, cpu_count() - 2)
        print(f"⚡ Phase 1 — XGBoost ({n_workers} workers CPU)")

        args_xgb = [
            (f, i + 1, total, output_file)
            for i, f in enumerate(files_xgb)
        ]
        with Pool(n_workers) as pool:
            for batch in pool.imap_unordered(_run_xgboost_only, args_xgb):
                all_results.extend(batch)
                _save_partial(all_results, output_file)

    # ── Phase 2 : LSTM + Transformer séquentiel sur GPU ──────────────────────
    if files_deep:
        print(f"\n⚡ Phase 2 — LSTM + Transformer (GPU DirectML, séquentiel)")
        offset = len(files_xgb)

        for i, file_info in enumerate(files_deep):
            batch = _run_deep_sequential(
                file_info, offset + i + 1, total, output_file
            )
            all_results.extend(batch)
            _save_partial(all_results, output_file)

    # ── Rapport final ─────────────────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_file, index=False)
        print(f"\n✅ {len(df)} résultats sauvegardés → {output_file}")
        print(df.groupby('model')[['pnl_pct', 'sharpe_ratio', 'win_rate']]
                .mean().round(2).to_string())
    else:
        print("\n❌ Aucun résultat.")


if __name__ == '__main__':
    main()
