"""
Analyse massive sur tout le dataset - XGBoost + LSTM + Transformer
Train < 2023, backtest sur 2023+ pour tous les fichiers
LSTM/Transformer uniquement sur timeframes ≥ 4h
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Ajoute la racine du projet (dossier BOT) au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.pipeline import TradingPipeline
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Configuration
DATA_FOLDERS = {
    'crypto_binance': 'data/crypto_binance/',
}

INITIAL_CAPITAL = 10000
SPLIT_DATE = "2023-01-01"  # tout avant = train, tout après = test

# Hyperparamètres pour LSTM/Transformer
EPOCHS = 50
BATCH_SIZE = 32
LOOKBACK = 60

# Timeframes pour lesquels on teste les 3 modèles (sinon XGBoost seul)
TF_LONG = ['4h', '6h', '12h', '1d', '1w']

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
    1 fichier, modèles selon timeframe:
      - TF courts (5m-2h): XGBoost seul
      - TF longs (≥4h): XGBoost + LSTM + Transformer
    """
    file_info, index, total = args
    results = []

    # Adapter les modèles selon le timeframe
    if file_info['timeframe'] in TF_LONG:
        models_to_test = ['xgboost', 'lstm', 'transformer']
        suffix = " [3 modèles]"
    else:
        models_to_test = ['xgboost']
        suffix = " [XGBoost seul]"

    try:
        print(f"[{index}/{total}] {file_info['filename']} ({file_info['timeframe']}) [{file_info['asset_type']}]{suffix}")

        # On prépare les données une seule fois
        pipeline_data = TradingPipeline()
        pipeline_data.load_data({file_info['timeframe']: file_info['filepath']})

        try:
            X_train, y_train, X_val, y_val, X_test_full, y_test_full, n_feat = pipeline_data.prepare_training_data(
                primary_timeframe=file_info['timeframe'],
                multi_tf=False,
                train_end=None,
                val_end=None,
                test_end=None,
            )
        except ValueError as e:
            print(f"   ❌ Erreur préparation données: {e}")
            return results

        # Index correspondant à X_test_full / y_test_full = pipeline_data.test_prices.index
        test_index_full = pipeline_data.test_prices.index

        split_ts = pd.to_datetime(SPLIT_DATE, utc=True)
        mask_test = test_index_full >= split_ts

        if mask_test.sum() < 10:
            print("   ⚠️ Trop peu de données en test après 2023-01-01, setup ignoré.")
            return results

        X_test = X_test_full[mask_test]
        y_test = y_test_full[mask_test]
        test_prices_filtered = pipeline_data.test_prices.loc[mask_test]

        # Préparation train complet (train + val)
        X_train_all = X_train
        y_train_all = y_train
        if X_val is not None and len(X_val) > 0:
            X_train_all = np.concatenate([X_train, X_val], axis=0)
            y_train_all = np.concatenate([y_train, y_val], axis=0)

        # Boucle sur les modèles adaptés au timeframe
        for model_type in models_to_test:
            try:
                print(f"   → {model_type.upper()}...", flush=True)

                # Recréer un pipeline propre pour ce modèle
                pipeline = TradingPipeline()
                pipeline.load_data({file_info['timeframe']: file_info['filepath']})
                
                # On recopie les scalers et test_prices du pipeline_data
                pipeline.scaler_X = pipeline_data.scaler_X
                pipeline.scaler_y = pipeline_data.scaler_y
                pipeline.test_prices = test_prices_filtered.copy()
                pipeline.primary_timeframe = file_info['timeframe']

                # Entraînement selon le type de modèle
                if model_type == 'xgboost':
                    pipeline.train_model(
                        model_type='xgboost',
                        X_train=X_train_all,
                        y_train=y_train_all,
                        X_val=X_val,
                        y_val=y_val,
                        n_features=n_feat,
                        lookback=LOOKBACK,
                    )
                else:  # lstm ou transformer
                    pipeline.train_model(
                        model_type=model_type,
                        X_train=X_train_all,
                        y_train=y_train_all,
                        X_val=X_val,
                        y_val=y_val,
                        n_features=n_feat,
                        lookback=LOOKBACK,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
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
    print("🚀 RÉAPPRENTISSAGE COMPLET - Stratégie optimisée par timeframe")
    print("   • TF courts (5m-2h): XGBoost seul")
    print("   • TF longs (≥4h): XGBoost + LSTM + Transformer")
    print("="*80)
    print(f"Capital initial: ${INITIAL_CAPITAL}")
    print(f"Epochs (LSTM/Transformer): {EPOCHS}, Batch size: {BATCH_SIZE}")
    print("="*80)

    all_files = scan_all_files()
    print(f"📂 Avant filtre: {len(all_files)} fichiers")
    all_files = [f for f in all_files if "_1m.csv" not in f['filename']]
    print(f"📂 Après filtre (1m exclus): {len(all_files)} fichiers")
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
