import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# 設定路徑
root = Path(r"c:\0_git workspace\Continual-Imbalance-Ensemble")
phase1_path = root / "results/phase1_baseline/xgb/rerun_20260406_gridplus_n24/tuned/bankruptcy_year_splits_xgb_raw.csv"
phase2_path = root / "results/phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv"
phase3_static_path = root / "results/phase3_feature/xgb_bankruptcy_fs_full_static.csv"
phase3_des_path = root / "results/phase3_feature/xgb_bankruptcy_fs_full_des.csv"

out_dir = root / "results/phase3_feature/plots"
out_dir.mkdir(parents=True, exist_ok=True)

SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")

def parse_split_order(split_id):
    m = SPLIT_RE.match(str(split_id).strip())
    if not m: return (99, 99)
    return (int(m.group(1)), int(m.group(2)))

def build_complete_evolution_plot():
    # 1. 加載數據
    df1 = pd.read_csv(phase1_path)
    df2 = pd.read_csv(phase2_path)
    df3_s = pd.read_csv(phase3_static_path)
    df3_d = pd.read_csv(phase3_des_path)

    # 2. Phase 1: Best Single
    p1_best = df1.groupby('split').agg({'AUC': 'max', 'F1': 'max'}).reset_index()

    # 3. Phase 2: Simple Static Ensemble (All_6)
    p2_ens = df2[(df2['ensemble'] == '6models') & (df2['type'] == 'k_subsets_mean')].copy()

    # 4. Phase 3: FS Static Ensemble (Best FS All_6)
    p3_static = df3_s[df3_s['ensemble'] == 'All_6'].groupby('split').agg({'AUC': 'max', 'F1': 'max'}).reset_index()

    # 5. Phase 3: FS Dynamic Ensemble (Best DES)
    # 我們取所有 DES 方法中的最高值
    p3_des = df3_d.groupby('split').agg({'AUC': 'max', 'F1': 'max'}).reset_index()

    # 6. 統一 Split 排序
    splits = sorted(p1_best['split'].unique(), key=parse_split_order)
    p1_data = p1_best.set_index('split').loc[splits].reset_index()
    p2_data = p2_ens.set_index('split').loc[splits].reset_index()
    p3_s_data = p3_static.set_index('split').loc[splits].reset_index()
    p3_d_data = p3_des.set_index('split').loc[splits].reset_index()

    # 7. 繪圖
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    metrics = ['AUC', 'F1']
    colors = {
        'Single XGB (Best)': '#7F7F7F',
        'XGB Ensemble (Simple Static)': '#1F77B4',
        'XGB Ensemble + FS (Static)': '#FF7F0E',
        'XGB Ensemble + FS (Dynamic/DES)': '#D62728'
    }
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        
        # Plot Single
        ax.plot(p1_data['split'], p1_data[m], label='Single XGB (Best Baseline)', 
                color=colors['Single XGB (Best)'], marker='x', linestyle='--', alpha=0.5)
        
        # Plot Simple Static
        ax.plot(p2_data['split'], p2_data[m], label='XGB Ensemble (Simple Static)', 
                color=colors['XGB Ensemble (Simple Static)'], marker='o', linestyle=':', alpha=0.7)
        
        # Plot FS Static
        ax.plot(p3_s_data['split'], p3_s_data[m], label='XGB Ensemble + FS (Static)', 
                color=colors['XGB Ensemble + FS (Static)'], marker='^', linestyle='-.')
        
        # Plot FS Dynamic (The Final Proposed)
        ax.plot(p3_d_data['split'], p3_d_data[m], label='XGB Ensemble + FS (Dynamic DES)', 
                color=colors['XGB Ensemble + FS (Dynamic/DES)'], marker='s', linestyle='-', linewidth=3)
        
        ax.set_ylabel(m, fontsize=12, fontweight='bold')
        ax.set_title(f"Complete Evolution: Single vs Static vs Dynamic Ensemble ({m})", fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=10)
        
        # 標註 DES 的最高點
        max_val = p3_d_data[m].max()
        max_idx = p3_d_data[m].idxmax()
        ax.annotate(f"Peak: {max_val:.4f}", xy=(max_idx, max_val), xytext=(0, 20),
                    textcoords='offset points', ha='center', color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))

    plt.xticks(range(len(splits)), splits, rotation=45, ha='right')
    plt.xlabel("Incremental Year Splits", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = out_dir / "xgb_complete_comparison_evolution.png"
    plt.savefig(save_path, dpi=200)
    print(f"Complete evolution plot saved to: {save_path}")

if __name__ == "__main__":
    build_complete_evolution_plot()
