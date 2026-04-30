import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# 設定路徑
root = Path(r"c:\0_git workspace\Continual-Imbalance-Ensemble")
phase1_path = root / "results/phase1_baseline/xgb/rerun_20260406_gridplus_n24/tuned/bankruptcy_year_splits_xgb_raw.csv"
phase2_path = root / "results/phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv"
phase3_path = root / "results/phase3_feature/xgb_bankruptcy_fs_full_static.csv"

out_dir = root / "results/phase3_feature/plots"
out_dir.mkdir(parents=True, exist_ok=True)

SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")

def parse_split_order(split_id):
    m = SPLIT_RE.match(str(split_id).strip())
    if not m: return (99, 99)
    return (int(m.group(1)), int(m.group(2)))

def build_evolution_plot():
    # 1. 加載數據
    df1 = pd.read_csv(phase1_path)
    df2 = pd.read_csv(phase2_path)
    df3 = pd.read_csv(phase3_path)

    # 2. Phase 1: Best Single
    p1_best = df1.groupby('split').agg({'AUC': 'max', 'F1': 'max'}).reset_index()
    p1_best['label'] = 'Single XGB (Best)'

    # 3. Phase 2: Simple Ensemble (All_6)
    p2_ens = df2[(df2['ensemble'] == '6models') & (df2['type'] == 'k_subsets_mean')].copy()
    p2_ens = p2_ens[['split', 'AUC', 'F1']]
    p2_ens['label'] = 'XGB Ensemble (Simple)'

    # 4. Phase 3: FS Ensemble (Best FS All_6)
    # 我們取 All_6 中 AUC 最高的一組 FS 方法 (通常是 mutual_info_r80)
    p3_ens = df3[df3['ensemble'] == 'All_6'].copy()
    p3_best_fs = p3_ens.groupby('split').agg({'AUC': 'max', 'F1': 'max'}).reset_index()
    p3_best_fs['label'] = 'XGB Ensemble + FS (Best)'

    # 5. 統一 Split 排序
    splits = sorted(p1_best['split'].unique(), key=parse_split_order)
    p1_data = p1_best.set_index('split').loc[splits].reset_index()
    p2_data = p2_ens.set_index('split').loc[splits].reset_index()
    p3_data = p3_best_fs.set_index('split').loc[splits].reset_index()

    # 6. 繪圖
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    metrics = ['AUC', 'F1']
    colors = {
        'Single XGB (Best)': '#7F7F7F',        # 灰色
        'XGB Ensemble (Simple)': '#1F77B4',    # 藍色
        'XGB Ensemble + FS (Best)': '#D62728'  # 紅色
    }
    markers = {'Single XGB (Best)': 'x', 'XGB Ensemble (Simple)': 'o', 'XGB Ensemble + FS (Best)': 's'}
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        
        # Plot Single
        ax.plot(p1_data['split'], p1_data[m], label='Single XGB (Best Baseline)', 
                color=colors['Single XGB (Best)'], marker=markers['Single XGB (Best)'], linestyle='--', alpha=0.6)
        
        # Plot Simple Ensemble
        ax.plot(p2_data['split'], p2_data[m], label='XGB Ensemble (Simple Average)', 
                color=colors['XGB Ensemble (Simple)'], marker=markers['XGB Ensemble (Simple)'], linestyle=':', linewidth=1.5)
        
        # Plot FS Ensemble
        ax.plot(p3_data['split'], p3_data[m], label='XGB Ensemble + FS (Proposed)', 
                color=colors['XGB Ensemble + FS (Best)'], marker=markers['XGB Ensemble + FS (Best)'], linestyle='-', linewidth=2.5)
        
        ax.set_ylabel(m, fontsize=12, fontweight='bold')
        ax.set_title(f"Performance Evolution: Single vs Ensemble vs FS-Ensemble ({m})", fontsize=15, fontweight='bold', pad=15)
        ax.legend(loc='lower left', frameon=True, shadow=True)
        
        # 在最後一個點標註最終效能
        ax.annotate(f"{p3_data.iloc[-1][m]:.4f}", xy=(len(splits)-1, p3_data.iloc[-1][m]), 
                    xytext=(10, 0), textcoords='offset points', color=colors['XGB Ensemble + FS (Best)'], fontweight='bold')

    plt.xticks(range(len(splits)), splits, rotation=40, ha='right')
    plt.xlabel("Incremental Year Splits", fontsize=12, fontweight='bold')
    
    # 加入浮水印或標註
    fig.text(0.99, 0.01, 'Phase 3 Feature Selection Evolution', ha='right', fontsize=10, alpha=0.5)
    
    plt.tight_layout()
    save_path = out_dir / "xgb_comparison_evolution.png"
    plt.savefig(save_path, dpi=200)
    print(f"Evolution plot saved to: {save_path}")

if __name__ == "__main__":
    build_evolution_plot()
