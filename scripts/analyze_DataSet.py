"""
Analyse de tous les datasets crypto avec XGBoost, LSTM et Transformer.
- Phase 1 : XGBoost     → CPU Pool (cpu_count - 2 workers)
- Phase 2+3 simultané   : LSTM CPU (4 workers × 4 threads)
                        + Transformer GPU DirectML (thread séparé)
"""

import os
import sys
import random
import logging
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.pipeline import TradingPipeline

random.seed(42)
np.random.seed(42)

os.makedirs('results', exist_ok=True)
os.makedirs('logs',    exist_ok=True)

logging.basicConfig(
    filename='logs/analyze_dataset.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s',
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_FOLDERS = {
    'crypto_binance': 'data/crypto_binance/',
}

INITIAL_CAPITAL = 10_000
SPLIT_DATE      = "2023-01-01"
LOOKBACK        = 60
EPOCHS          = 50
BATCH_SIZE      = 256

LSTM_ALLOWED_TF = {'1h', '2h', '4h', '6h', '12h', '1d', '1w'}

# Workers LSTM (CPU) : 4 workers × 4 threads = 16 threads = 100% du 9800X3D
LSTM_WORKERS            = 4
LSTM_THREADS_PER_WORKER = 4

# ─────────────────────────────────────────────────────────────────────────────
# Contrôle des phases
# ─────────────────────────────────────────────────────────────────────────────

RUN_XGBOOST     = True   # déjà fait → skip
RUN_LSTM        = True
RUN_TRANSFORMER = True

# Fichier de résultats XGBoost existant (utilisé si RUN_XGBOOST=False)
XGB_RESULTS_FILE = 'results/analyze_dataset_XXXXXXXX_XXXXXX.csv'   # ← à adapter


# ─────────────────────────────────────────────────────────────────────────────
# Scan des fichiers
# ─────────────────────────────────────────────────────────────────────────────

def scan_all_files() -> list:
    all_files = []
    for asset_type, folder in DATA_FOLDERS.items():
        if not os.path.exists(folder):
            print(f"⚠️  Dossier introuvable : {folder}")
            continue
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith('.csv'):
                continue
            parts = filename.replace('.csv', '').rsplit('_', 1)
            if len(parts) != 2:
                continue
            symbol, timeframe = parts
            all_files.append({
                'filename'  : filename,
                'filepath'  : os.path.join(folder, filename),
                'symbol'    : symbol,
                'timeframe' : timeframe,
                'asset_type': asset_type,
            })
    print(f"📂 {len(all_files)} fichiers trouvés")
    return all_files


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde incrémentale (thread-safe via lock externe)
# ─────────────────────────────────────────────────────────────────────────────

def _save_partial(results: list, filepath: str):
    if results:
        pd.DataFrame(results).to_csv(filepath, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Worker XGBoost (CPU Pool)
# ─────────────────────────────────────────────────────────────────────────────

def _run_xgboost_only(args) -> list:
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
            print(f"  ❌ Prep: {e}", flush=True)
            return results

        split_ts  = pd.to_datetime(SPLIT_DATE, utc=True)
        mask_test = pipeline_data.test_prices.index >= split_ts
        if mask_test.sum() < 10:
            print(f"  ❌ Pas assez de données post {SPLIT_DATE}", flush=True)
            return results

        X_test  = X_test_full[mask_test]
        y_test  = y_test_full[mask_test]
        has_val = X_val is not None and len(X_val) > 0
        X_tr    = np.concatenate([X_train, X_val]) if has_val else X_train
        y_tr    = np.concatenate([y_train, y_val]) if has_val else y_train

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
            print(
                f"  ✅ {metrics['total_trades']} trades | "
                f"PNL: {metrics['pnl_pct']:+.1f}% | "
                f"Sharpe: {metrics['sharpe_ratio']:.2f}",
                flush=True,
            )
        else:
            print(f"  ⚠️  Aucun trade exécuté", flush=True)

    except Exception as e:
        print(f"  ❌ {e}", flush=True)
        logging.error(f"{file_info['filename']} XGB: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Worker LSTM (CPU Pool — thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _run_lstm_worker(args) -> list:
    """
    Worker LSTM multiprocessing.
    Limite les threads PyTorch pour éviter la sur-souscription CPU.
    """
    file_info, index, total, output_file = args

    import torch
    torch.set_num_threads(LSTM_THREADS_PER_WORKER)

    return _run_deep_sequential(
        file_info, index, total,
        output_file, model_types=['lstm'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Worker LSTM / Transformer (séquentiel)
# ─────────────────────────────────────────────────────────────────────────────

def _run_deep_sequential(
    file_info   : dict,
    index       : int,
    total       : int,
    output_file : str,
    model_types : list = None,
) -> list:
    if model_types is None:
        model_types = ['lstm', 'transformer']

    results = []

    for model_type in model_types:
        label = model_type.upper()
        print(f"[{index}/{total}] {label} {file_info['filename']}", flush=True)

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
                print(f"  ❌ Prep: {e}", flush=True)
                continue

            split_ts  = pd.to_datetime(SPLIT_DATE, utc=True)
            mask_test = pipeline_data.test_prices.index >= split_ts
            if mask_test.sum() < 10:
                print(f"  ❌ Pas assez de données post {SPLIT_DATE}", flush=True)
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
                    'model'     : label,
                })
                results.append(metrics)
                print(
                    f"  ✅ {metrics['total_trades']} trades | "
                    f"PNL: {metrics['pnl_pct']:+.1f}% | "
                    f"Sharpe: {metrics['sharpe_ratio']:.2f}",
                    flush=True,
                )
            else:
                print(f"  ⚠️  Aucun trade exécuté", flush=True)

        except Exception as e:
            print(f"  ❌ {label}: {e}", flush=True)
            logging.error(f"{file_info['filename']} {label}: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Thread Transformer (GPU DirectML — tourne en parallèle du Pool LSTM)
# ─────────────────────────────────────────────────────────────────────────────

def _run_transformer_thread(
    files_trans : list,
    output_file : str,
    all_results : list,
    lock        : threading.Lock,
):
    """
    Thread dédié au Transformer sur GPU DirectML.
    Tourne en parallèle du Pool LSTM CPU.
    Le lock garantit l'accès thread-safe à all_results.
    """
    total = len(files_trans)
    print(f"\n  [TRANSFORMER THREAD] Démarrage — {total} fichiers sur GPU\n",
          flush=True)

    for i, file_info in enumerate(files_trans):
        batch = _run_deep_sequential(
            file_info, i + 1, total,
            output_file, model_types=['transformer'],
        )
        if batch:
            with lock:
                all_results.extend(batch)
                _save_partial(all_results, output_file)

    print(f"\n  [TRANSFORMER THREAD] Terminé ✅\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Récapitulatif final
# ─────────────────────────────────────────────────────────────────────────────

def _print_recap(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"📊 RÉCAPITULATIF PAR MODÈLE ({len(df)} résultats totaux)")
    print(f"{'='*60}")

    for model_name in ['XGBOOST', 'LSTM', 'TRANSFORMER']:
        sub = df[df['model'] == model_name]
        if sub.empty:
            continue
        profitable = sub[sub['pnl_pct'] > 0]

        print(f"\n── {model_name} ({len(sub)} setups) ──────────────────────")
        print(f"  PNL moyen          : {sub['pnl_pct'].mean():+.2f}%")
        print(f"  PNL médian         : {sub['pnl_pct'].median():+.2f}%")
        print(f"  PNL max            : {sub['pnl_pct'].max():+.2f}%")
        print(f"  Sharpe moyen       : {sub['sharpe_ratio'].mean():.2f}")
        print(f"  Sharpe max         : {sub['sharpe_ratio'].max():.2f}")
        print(f"  Win rate moyen     : {sub['win_rate'].mean():.1f}%")
        print(f"  Setups profitables : "
              f"{len(profitable)}/{len(sub)} "
              f"({len(profitable)/len(sub)*100:.0f}%)")

        top3 = sub.nlargest(3, 'sharpe_ratio')[
            ['symbol', 'timeframe', 'pnl_pct', 'sharpe_ratio', 'win_rate']
        ]
        print(f"  Top 3 Sharpe :")
        for _, r in top3.iterrows():
            print(f"    {r['symbol']:10} {r['timeframe']:4} → "
                  f"PNL {r['pnl_pct']:+.1f}% | "
                  f"Sharpe {r['sharpe_ratio']:.2f} | "
                  f"WR {r['win_rate']:.0f}%")

    print(f"\n{'='*60}")
    print("📈 COMPARAISON SYNTHÉTIQUE")
    print(f"{'='*60}")
    summary = df.groupby('model').agg(
        setups         = ('pnl_pct',      'count'),
        pnl_moy        = ('pnl_pct',      'mean'),
        pnl_max        = ('pnl_pct',      'max'),
        sharpe_moy     = ('sharpe_ratio', 'mean'),
        sharpe_max     = ('sharpe_ratio', 'max'),
        winrate_moy    = ('win_rate',     'mean'),
        pct_profitable = ('pnl_pct',      lambda x: round((x > 0).mean() * 100, 1)),
    ).round(2)
    print(summary.to_string())
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_files   = scan_all_files()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/analyze_dataset_{timestamp}.csv"
    all_results = []
    lock        = threading.Lock()

    files_xgb   = all_files
    files_lstm  = [f for f in all_files if f['timeframe'] in LSTM_ALLOWED_TF]
    files_trans = all_files
    n_workers   = max(1, cpu_count() - 2)

    print(f"\n{'='*60}")
    print(f"📂 {len(all_files)} fichiers")
    print(f"  XGBoost     : {len(files_xgb)} fichiers (tous TF)  — {n_workers} workers CPU")
    print(f"  LSTM        : {len(files_lstm)} fichiers (TF ≥ 1h) — {LSTM_WORKERS} workers CPU × {LSTM_THREADS_PER_WORKER} threads")
    print(f"  Transformer : {len(files_trans)} fichiers (tous TF) — GPU DirectML (thread séparé)")
    print(f"{'='*60}\n")

    # ── Phase 1 : XGBoost ────────────────────────────────────────────────────
    if RUN_XGBOOST:
        print(f"⚡ Phase 1/3 — XGBoost ({n_workers} workers CPU)\n")
        args_xgb = [
            (f, i + 1, len(files_xgb), output_file)
            for i, f in enumerate(files_xgb)
        ]
        with Pool(n_workers) as pool:
            for batch in pool.imap_unordered(_run_xgboost_only, args_xgb):
                with lock:
                    all_results.extend(batch)
                    _save_partial(all_results, output_file)
    else:
        print("⏭️  Phase 1/3 — XGBoost ignorée (RUN_XGBOOST=False)")
        if os.path.exists(XGB_RESULTS_FILE):
            existing    = pd.read_csv(XGB_RESULTS_FILE)
            xgb_only    = existing[existing['model'] == 'XGBOOST']
            all_results = xgb_only.to_dict('records')
            print(f"  📂 {len(all_results)} résultats XGBoost chargés "
                  f"depuis {XGB_RESULTS_FILE}\n")
        else:
            print(f"  ⚠️  Fichier XGBoost introuvable : {XGB_RESULTS_FILE}\n")

    # ── Phase 2+3 simultanée : LSTM CPU + Transformer GPU ────────────────────
    run_both = RUN_LSTM and RUN_TRANSFORMER
    run_one  = RUN_LSTM or RUN_TRANSFORMER

    if run_one:
        print(f"\n⚡ Phase 2+3 — LSTM CPU + Transformer GPU (simultané)\n")

    transformer_thread = None

    # Lance le Transformer dans un thread séparé
    if RUN_TRANSFORMER:
        transformer_thread = threading.Thread(
            target=_run_transformer_thread,
            args=(files_trans, output_file, all_results, lock),
            daemon=False,
        )
        transformer_thread.start()
        if run_both:
            print(f"  ▶ Transformer GPU lancé en arrière-plan "
                  f"({len(files_trans)} fichiers)", flush=True)

    # Lance le Pool LSTM en parallèle
    if RUN_LSTM:
        print(f"  ▶ LSTM CPU démarré "
              f"({LSTM_WORKERS} workers × {LSTM_THREADS_PER_WORKER} threads, "
              f"{len(files_lstm)} fichiers)\n",
              flush=True)

        args_lstm = [
            (f, i + 1, len(files_lstm), output_file)
            for i, f in enumerate(files_lstm)
        ]
        with Pool(LSTM_WORKERS) as pool:
            for batch in pool.imap_unordered(_run_lstm_worker, args_lstm):
                with lock:
                    all_results.extend(batch)
                    _save_partial(all_results, output_file)

        print(f"\n  ▶ LSTM CPU terminé ✅", flush=True)

    # Attend la fin du Transformer
    if transformer_thread is not None:
        print(f"\n  ⏳ En attente de la fin du Transformer GPU...", flush=True)
        transformer_thread.join()

    # ── Récapitulatif final ───────────────────────────────────────────────────
    if not all_results:
        print("\n❌ Aucun résultat.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)

    _print_recap(df)

    print(f"\n💾 Résultats complets → {output_file}")


if __name__ == '__main__':
    main()
