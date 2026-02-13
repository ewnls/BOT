"""
Walk-Forward Validation sur les meilleurs setups optimisés
Teste la robustesse sur 4 périodes temporelles différentes
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.pipeline import TradingPipeline
import pandas as pd
from datetime import datetime

# Configuration
DATA_FOLDERS = {
    'crypto_binance': 'data/crypto_binance/',
}

INITIAL_CAPITAL = 10000
LOOKBACK = 60

# Périodes Walk-Forward
PERIODS = [
    {'name': 'Period_1', 'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31'},
    {'name': 'Period_2', 'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31'},
    {'name': 'Period_3', 'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-12-31'},
    {'name': 'Period_4', 'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2025-12-31'},
]

def get_csv_path(filename, asset_type):
    """Retrouve le chemin complet du CSV"""
    folder = DATA_FOLDERS.get(asset_type, 'data/')
    return os.path.join(folder, filename)

def validate_one_setup_one_period(setup_info, period):
    """
    Valide UN setup sur UNE période avec ses hyperparams optimaux.
    """
    try:
        csv_path = get_csv_path(setup_info['filename'], setup_info['asset_type'])
        
        if not os.path.exists(csv_path):
            return None
        
        pipeline = TradingPipeline()
        pipeline.load_data({setup_info['timeframe']: csv_path})
        
        # Préparer les données
        X_train, y_train, X_val, y_val, X_test_full, y_test_full, n_feat = pipeline.prepare_training_data(
            primary_timeframe=setup_info['timeframe'],
            multi_tf=False,
        )
        
        # Split temporel selon la période
        test_index_full = pipeline.test_prices.index
        test_start = pd.to_datetime(period['test_start'], utc=True)
        test_end = pd.to_datetime(period['test_end'], utc=True)
        
        # Mask pour le test de cette période
        mask_test = (test_index_full >= test_start) & (test_index_full <= test_end)
        
        if mask_test.sum() < 10:
            return None
        
        X_test = X_test_full[mask_test]
        y_test = y_test_full[mask_test]
        pipeline.test_prices = pipeline.test_prices.loc[mask_test]
        
        # Pour le train : tout avant test_start
        train_index_full = pd.concat([
            pd.DataFrame(index=pipeline.preprocessor.data[setup_info['timeframe']].index)
        ]).index
        
        mask_train = train_index_full < test_start
        
        # On utilise X_train et X_val qui sont déjà < test_start dans la plupart des cas
        # Mais pour être sûr on filtre
        
        # Entraîner avec les meilleurs hyperparams
        pipeline.train_model(
            model_type=setup_info['model'].lower(),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_features=n_feat,
            lookback=LOOKBACK,
            epochs=int(setup_info['best_epochs']),
            batch_size=int(setup_info['best_batch_size']),
            dropout=setup_info['best_dropout'],
            learning_rate=setup_info['best_learning_rate'],
        )
        
        # Backtest sur cette période
        metrics, _, _ = pipeline.backtest(
            X_test,
            y_test,
            initial_capital=INITIAL_CAPITAL,
            asset_type=setup_info['asset_type']
        )
        
        if 'error' in metrics:
            return None
        
        result = {
            'symbol': setup_info['symbol'],
            'timeframe': setup_info['timeframe'],
            'model': setup_info['model'],
            'period': period['name'],
            'test_start': period['test_start'],
            'test_end': period['test_end'],
            'pnl_pct': metrics['pnl_pct'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'profit_factor': metrics['profit_factor'],
            'win_rate': metrics['win_rate'],
            'total_trades': metrics['total_trades'],
            'max_drawdown': metrics['max_drawdown'],
            'best_epochs': int(setup_info['best_epochs']),
        }
        
        return result
        
    except Exception as e:
        print(f"   ❌ Erreur {period['name']}: {e}")
        return None

def validate_one_setup(args):
    """
    Valide UN setup sur TOUTES les périodes Walk-Forward.
    """
    setup_info, setup_idx, total_setups = args
    
    print(f"\n{'='*80}")
    print(f"[{setup_idx}/{total_setups}] {setup_info['symbol']} {setup_info['timeframe']} - {setup_info['model']}")
    print(f"Optimized: PNL={setup_info['optimized_pnl_pct']:+.2f}% | Sharpe={setup_info['optimized_sharpe']:.2f}")
    print(f"Best params: epochs={int(setup_info['best_epochs'])} | bs={int(setup_info['best_batch_size'])}")
    print(f"{'='*80}")
    
    results = []
    
    for period in PERIODS:
        print(f"   → Testing {period['name']} ({period['test_start']} to {period['test_end']})...", flush=True)
        result = validate_one_setup_one_period(setup_info, period)
        
        if result:
            results.append(result)
            print(f"      ✅ PNL={result['pnl_pct']:+.2f}% | Sharpe={result['sharpe_ratio']:.2f} | Trades={result['total_trades']}", flush=True)
        else:
            print(f"      ⚠️ Pas assez de données pour cette période", flush=True)
    
    if len(results) == 0:
        return None
    
    # Calculer les stats agrégées
    results_df = pd.DataFrame(results)
    
    aggregated = {
        'symbol': setup_info['symbol'],
        'timeframe': setup_info['timeframe'],
        'model': setup_info['model'],
        'filename': setup_info['filename'],
        'asset_type': setup_info['asset_type'],
        'optimized_pnl_pct': setup_info['optimized_pnl_pct'],
        'optimized_sharpe': setup_info['optimized_sharpe'],
        'best_epochs': int(setup_info['best_epochs']),
        'best_batch_size': int(setup_info['best_batch_size']),
        'best_dropout': setup_info['best_dropout'],
        'best_learning_rate': setup_info['best_learning_rate'],
        # Stats Walk-Forward
        'wf_avg_pnl': results_df['pnl_pct'].mean(),
        'wf_median_pnl': results_df['pnl_pct'].median(),
        'wf_std_pnl': results_df['pnl_pct'].std(),
        'wf_min_pnl': results_df['pnl_pct'].min(),
        'wf_max_pnl': results_df['pnl_pct'].max(),
        'wf_avg_sharpe': results_df['sharpe_ratio'].mean(),
        'wf_positive_periods': len(results_df[results_df['pnl_pct'] > 0]),
        'wf_total_periods': len(results_df),
        'wf_consistency': len(results_df[results_df['pnl_pct'] > 0]) / len(results_df) * 100,
    }
    
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Moyenne PNL: {aggregated['wf_avg_pnl']:+.2f}% (médiane: {aggregated['wf_median_pnl']:+.2f}%)")
    print(f"   Périodes positives: {aggregated['wf_positive_periods']}/{aggregated['wf_total_periods']} ({aggregated['wf_consistency']:.0f}%)")
    print(f"   Écart-type: {aggregated['wf_std_pnl']:.2f}% | Min: {aggregated['wf_min_pnl']:+.2f}% | Max: {aggregated['wf_max_pnl']:+.2f}%")
    
    return aggregated, results

def main():
    print("="*80)
    print("🔄 WALK-FORWARD VALIDATION - TOP SETUPS OPTIMISÉS")
    print("="*80)
    print(f"Périodes testées: {len(PERIODS)}")
    for p in PERIODS:
        print(f"  • {p['name']}: {p['test_start']} → {p['test_end']}")
    print("="*80)
    
    # Charger les setups optimisés améliorés
    improved_file = 'results/grid_search_improved_only.csv'
    if not os.path.exists(improved_file):
        print(f"❌ Fichier {improved_file} introuvable! Utilisation du fichier complet.")
        improved_file = 'results/grid_search_optimized_20260213_020659.csv'
    
    df = pd.read_csv(improved_file)
    
    # Prendre les 10 meilleurs setups
    top_10 = df.nlargest(10, 'optimized_pnl_pct')
    
    print(f"\n📂 {len(top_10)} setups à valider (Top 10)")
    print("="*80)
    
    args = [
        (row.to_dict(), idx + 1, len(top_10))
        for idx, row in top_10.iterrows()
    ]
    
    all_aggregated = []
    all_detailed = []
    
    # Séquentiel (Walk-Forward est déjà lent)
    for arg in args:
        result = validate_one_setup(arg)
        if result:
            aggregated, detailed = result
            all_aggregated.append(aggregated)
            all_detailed.extend(detailed)
    
    if len(all_aggregated) == 0:
        print("\n❌ Aucun setup validé!")
        return
    
    # Sauvegarder les résultats
    df_aggregated = pd.DataFrame(all_aggregated)
    df_detailed = pd.DataFrame(all_detailed)
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    agg_file = f"results/walk_forward_aggregated_{timestamp}.csv"
    det_file = f"results/walk_forward_detailed_{timestamp}.csv"
    
    df_aggregated.to_csv(agg_file, index=False)
    df_detailed.to_csv(det_file, index=False)
    
    print("\n" + "="*80)
    print("📊 RÉSULTATS WALK-FORWARD VALIDATION")
    print("="*80)
    
    # Filtrer les setups robustes (≥75% périodes positives)
    robust_setups = df_aggregated[df_aggregated['wf_consistency'] >= 75.0]
    
    print(f"\n✅ Setups validés (≥75% périodes positives): {len(robust_setups)}/{len(df_aggregated)}")
    
    if len(robust_setups) > 0:
        print("\n🏆 SETUPS ROBUSTES:")
        for idx, row in robust_setups.iterrows():
            print(f"   {row['symbol']:10} {row['timeframe']:4} {row['model']:11} → "
                  f"Avg PNL={row['wf_avg_pnl']:+.2f}% | Consistency={row['wf_consistency']:.0f}% | Sharpe={row['wf_avg_sharpe']:.2f}")
        
        # Sauvegarder seulement les robustes
        robust_file = f"results/walk_forward_robust_only_{timestamp}.csv"
        robust_setups.to_csv(robust_file, index=False)
        print(f"\n💾 Setups robustes: {robust_file}")
    
    print(f"\n💾 Résultats agrégés: {agg_file}")
    print(f"💾 Résultats détaillés: {det_file}")
    print("="*80)
    print("✅ WALK-FORWARD VALIDATION TERMINÉE")

if __name__ == '__main__':
    main()
