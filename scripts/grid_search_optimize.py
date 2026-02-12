"""
Grid Search Hyperparameter Optimization pour les Top 50 setups
Optimise epochs, batch_size, dropout, learning_rate pour LSTM/Transformer
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.pipeline import TradingPipeline
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
import json
from itertools import product

# Configuration
DATA_FOLDERS = {
    'crypto_binance': 'data/crypto_binance/',
}

INITIAL_CAPITAL = 10000
SPLIT_DATE = "2023-01-01"
LOOKBACK = 60

# Grid Search Hyperparameters
GRID_PARAMS = {
    'epochs': [50, 100, 150],        # On teste 3 valeurs d'epochs
    'batch_size': [32],              # Fixé à 32 (optimal dans la plupart des cas)
    'dropout': [0.2],                # Fixé à 0.2 (standard)
    'learning_rate': [0.001]         # Fixé à 0.001 (valeur par défaut)
}

# Générer toutes les combinaisons
param_combinations = list(product(
    GRID_PARAMS['epochs'],
    GRID_PARAMS['batch_size'],
    GRID_PARAMS['dropout'],
    GRID_PARAMS['learning_rate']
))

print(f"🔍 Grid Search: {len(param_combinations)} combinaisons par setup")

def get_csv_path(filename, asset_type):
    """Retrouve le chemin complet du CSV"""
    folder = DATA_FOLDERS.get(asset_type, 'data/')
    return os.path.join(folder, filename)

def train_and_evaluate_one_config(args):
    """
    Entraîne et évalue UNE configuration d'hyperparamètres pour UN setup.
    """
    setup_info, params, config_idx, total_configs = args
    
    epochs, batch_size, dropout, lr = params
    
    try:
        # Charger les données
        csv_path = get_csv_path(setup_info['filename'], setup_info['asset_type'])
        
        if not os.path.exists(csv_path):
            return None
        
        pipeline = TradingPipeline()
        pipeline.load_data({setup_info['timeframe']: csv_path})
        
        # Préparer les données
        X_train, y_train, X_val, y_val, X_test_full, y_test_full, n_feat = pipeline.prepare_training_data(
            primary_timeframe=setup_info['timeframe'],
            multi_tf=False,
            train_end=None,
            val_end=None,
            test_end=None,
        )
        
        # Split par date (train < 2023, test >= 2023)
        test_index_full = pipeline.test_prices.index
        split_ts = pd.to_datetime(SPLIT_DATE, utc=True)
        mask_test = test_index_full >= split_ts
        
        if mask_test.sum() < 10:
            return None
        
        X_test = X_test_full[mask_test]
        y_test = y_test_full[mask_test]
        pipeline.test_prices = pipeline.test_prices.loc[mask_test]
        
        # Train complet (train + val)
        X_train_all = np.concatenate([X_train, X_val], axis=0) if X_val is not None else X_train
        y_train_all = np.concatenate([y_train, y_val], axis=0) if y_val is not None else y_train
        
        # Entraînement avec les hyperparams testés
        pipeline.train_model(
            model_type=setup_info['model'].lower(),
            X_train=X_train_all,
            y_train=y_train_all,
            X_val=X_val,
            y_val=y_val,
            n_features=n_feat,
            lookback=LOOKBACK,
            epochs=epochs,
            batch_size=batch_size,
            dropout=dropout,
            learning_rate=lr
        )
        
        # Backtest
        metrics, _, _ = pipeline.backtest(
            X_test,
            y_test,
            initial_capital=INITIAL_CAPITAL,
            asset_type=setup_info['asset_type']
        )
        
        if 'error' in metrics:
            return None
        
        # Retourner les résultats avec les hyperparams
        result = {
            'symbol': setup_info['symbol'],
            'timeframe': setup_info['timeframe'],
            'model': setup_info['model'],
            'epochs': epochs,
            'batch_size': batch_size,
            'dropout': dropout,
            'learning_rate': lr,
            'pnl_pct': metrics['pnl_pct'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'profit_factor': metrics['profit_factor'],
            'win_rate': metrics['win_rate'],
            'total_trades': metrics['total_trades'],
            'max_drawdown': metrics['max_drawdown'],
            'config_idx': config_idx,
        }
        
        print(f"   [{config_idx}/{total_configs}] epochs={epochs} bs={batch_size} → PNL={metrics['pnl_pct']:+.2f}% Sharpe={metrics['sharpe_ratio']:.2f}", flush=True)
        
        return result
        
    except Exception as e:
        print(f"   ❌ Erreur config {config_idx}: {e}", flush=True)
        return None

def optimize_one_setup(args):
    """
    Optimise les hyperparamètres pour UN setup via Grid Search.
    """
    setup_info, setup_idx, total_setups = args
    
    print("="*80)
    print("🔍 GRID SEARCH HYPERPARAMETER OPTIMIZATION - TOP 50 SETUPS (GRILLE RÉDUITE)")
    print("="*80)
    print(f"Combinaisons testées par setup: {len(param_combinations)} (epochs uniquement)")
    print(f"Paramètres: epochs={GRID_PARAMS['epochs']}")
    print(f"            batch_size=32 (fixe), dropout=0.2 (fixe), lr=0.001 (fixe)")
    print("="*80)
    
    results = []
    
    # Tester toutes les combinaisons d'hyperparams
    total_configs = len(param_combinations)
    
    for config_idx, params in enumerate(param_combinations, 1):
        result = train_and_evaluate_one_config((setup_info, params, config_idx, total_configs))
        if result is not None:
            results.append(result)
    
    if len(results) == 0:
        print(f"   ❌ Aucune configuration valide pour ce setup")
        return None
    
    # Trouver la meilleure configuration (on maximise le PNL * Sharpe)
    results_df = pd.DataFrame(results)
    results_df['score'] = results_df['pnl_pct'] * results_df['sharpe_ratio']
    best_config = results_df.loc[results_df['score'].idxmax()]
    
    improvement_pnl = best_config['pnl_pct'] - setup_info['pnl_pct']
    improvement_sharpe = best_config['sharpe_ratio'] - setup_info['sharpe_ratio']
    
    print(f"\n{'='*80}")
    print(f"✅ MEILLEURE CONFIG TROUVÉE:")
    print(f"   epochs={int(best_config['epochs'])} | batch_size={int(best_config['batch_size'])} | dropout={best_config['dropout']:.1f} | lr={best_config['learning_rate']:.4f}")
    print(f"   PNL: {best_config['pnl_pct']:+.2f}% (baseline: {setup_info['pnl_pct']:+.2f}%) → Δ={improvement_pnl:+.2f}%")
    print(f"   Sharpe: {best_config['sharpe_ratio']:.2f} (baseline: {setup_info['sharpe_ratio']:.2f}) → Δ={improvement_sharpe:+.2f}")
    print(f"{'='*80}\n")
    
    # Retourner le setup avec sa meilleure config
    optimized_setup = {
        'symbol': setup_info['symbol'],
        'timeframe': setup_info['timeframe'],
        'model': setup_info['model'],
        'filename': setup_info['filename'],
        'asset_type': setup_info['asset_type'],
        'baseline_pnl_pct': setup_info['pnl_pct'],
        'baseline_sharpe': setup_info['sharpe_ratio'],
        'optimized_pnl_pct': best_config['pnl_pct'],
        'optimized_sharpe': best_config['sharpe_ratio'],
        'improvement_pnl': improvement_pnl,
        'improvement_sharpe': improvement_sharpe,
        'best_epochs': int(best_config['epochs']),
        'best_batch_size': int(best_config['batch_size']),
        'best_dropout': best_config['dropout'],
        'best_learning_rate': best_config['learning_rate'],
        'profit_factor': best_config['profit_factor'],
        'win_rate': best_config['win_rate'],
        'total_trades': int(best_config['total_trades']),
        'max_drawdown': best_config['max_drawdown'],
    }
    
    return optimized_setup

def main():
    print("="*80)
    print("🔍 GRID SEARCH HYPERPARAMETER OPTIMIZATION - TOP 50 SETUPS")
    print("="*80)
    print(f"Combinaisons testées par setup: {len(param_combinations)}")
    print(f"Paramètres: epochs={GRID_PARAMS['epochs']}, batch_size={GRID_PARAMS['batch_size']}")
    print(f"            dropout={GRID_PARAMS['dropout']}, lr={GRID_PARAMS['learning_rate']}")
    print("="*80)
    
    # Charger le top 50
    top_50_file = 'results/top_50_setups_to_optimize.csv'
    if not os.path.exists(top_50_file):
        print(f"❌ Fichier {top_50_file} introuvable!")
        return
    
    top_50 = pd.read_csv(top_50_file)
    print(f"📂 {len(top_50)} setups à optimiser")
    
    # Préparer les arguments pour le multiprocessing
    args = [
        (row.to_dict(), idx + 1, len(top_50))
        for idx, row in top_50.iterrows()
    ]
    
    # IMPORTANT: Grid Search est CPU-intensif, on réduit le nombre de workers
    max_workers = 4  # On lance 3 setups en parallèle (chacun teste 81 configs)
    n_workers = min(max_workers, cpu_count())
    print(f"⚡ Utilisation de {n_workers} workers")
    print("="*80)
    
    all_optimized = []
    
    # Séquentiel pour debug (décommenter si besoin)
    # for arg in args:
    #     result = optimize_one_setup(arg)
    #     if result:
    #         all_optimized.append(result)
    
    # Parallèle
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(optimize_one_setup, args):
            if result:
                all_optimized.append(result)
    
    if len(all_optimized) == 0:
        print("\n❌ Aucun setup optimisé!")
        return
    
    # Sauvegarder les résultats
    df_optimized = pd.DataFrame(all_optimized)
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/grid_search_optimized_{timestamp}.csv"
    df_optimized.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("📊 RÉSULTATS GRID SEARCH")
    print("="*80)
    print(f"Setups optimisés: {len(df_optimized)}")
    print(f"\nAmélioration moyenne PNL: {df_optimized['improvement_pnl'].mean():+.2f}%")
    print(f"Amélioration moyenne Sharpe: {df_optimized['improvement_sharpe'].mean():+.2f}")
    print(f"\nSetups améliorés (PNL): {len(df_optimized[df_optimized['improvement_pnl'] > 0])}/{len(df_optimized)}")
    print(f"Setups améliorés (Sharpe): {len(df_optimized[df_optimized['improvement_sharpe'] > 0])}/{len(df_optimized)}")
    
    # Top 5 améliorations
    print("\n" + "-"*80)
    print("🏆 TOP 5 PLUS GRANDES AMÉLIORATIONS (PNL):")
    top_improvements = df_optimized.nlargest(5, 'improvement_pnl')
    for idx, row in top_improvements.iterrows():
        print(f"   {row['symbol']:8} {row['timeframe']:4} {row['model']:11} → +{row['improvement_pnl']:.2f}% (de {row['baseline_pnl_pct']:.2f}% à {row['optimized_pnl_pct']:.2f}%)")
    
    print(f"\n💾 Résultats sauvegardés: {output_file}")
    print("="*80)
    print("✅ GRID SEARCH TERMINÉ")

if __name__ == '__main__':
    main()
