import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# 設定路徑
root = Path(r"c:\0_git workspace\Continual-Imbalance-Ensemble")
phase1_path = root / "results/phase1_baseline/xgb/rerun_20260406_gridplus_n24/tuned/bankruptcy_year_splits_xgb_raw.csv"
phase2_path = root / "results/phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv"
out_dir = root / "results/phase2_ensemble/plots"
out_dir.mkdir(parents=True, exist_ok=True)

SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")

def parse_split_order(split_id):
    m = SPLIT_RE.match(str(split_id).strip())
    if not m: return (99, 99)
    return (int(m.group(1)), int(m.group(2)))

def build_plot():
    # 1. 加載數據
    df1 = pd.read_csv(phase1_path)
    df2 = pd.read_csv(phase2_path)

    # 2. 處理 Phase 1 (取每個 split 的最佳單一模型)
    p1_best = df1.groupby('split').agg({'AUC': 'max', 'F1': 'max'}).reset_index()
    p1_best['label'] = 'Single XGB (Best of P1)'

    # 3. 處理 Phase 2 (取 All_6 集成，通常是 6models k_subsets_mean)
    # 注意：在原始 CSV 中，All_6 對應 ensemble='6models' 且 type='k_subsets_mean'
    p2_ens = df2[(df2['ensemble'] == '6models') & (df2['type'] == 'k_subsets_mean')].copy()
    p2_ens = p2_ens[['split', 'AUC', 'F1']]
    p2_ens['label'] = 'XGB Ensemble (All_6)'

    # 4. 合併並排序
    splits = sorted(p1_best['split'].unique(), key=parse_split_order)
    p1_best = p1_best.set_index('split').loc[splits].reset_index()
    p2_ens = p2_ens.set_index('split').loc[splits].reset_index()

    # 5. 繪圖
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    metrics = ['AUC', 'F1']
    colors = {'Single XGB (Best of P1)': '#888888', 'XGB Ensemble (All_6)': '#D62728'}
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        # Plot Single
        ax.plot(p1_best['split'], p1_best[m], marker='o', linestyle='--', color=colors['Single XGB (Best of P1)'], label='Single XGB (Best of P1)', alpha=0.7)
        # Plot Ensemble
        ax.plot(p2_ens['split'], p2_ens[m], marker='s', linestyle='-', color=colors['XGB Ensemble (All_6)'], label='XGB Ensemble (All_6)', linewidth=2.5)
        
        ax.set_ylabel(m, fontsize=12)
        ax.set_title(f"XGB Single vs Ensemble Comparison: {m}", fontsize=14, fontweight='bold')
        ax.legend()
        
        # 標註差距
        for idx, split in enumerate(splits):
            diff = p2_ens.iloc[idx][m] - p1_best.iloc[idx][m]
            if diff > 0:
                ax.annotate(f"+{diff:.3f}", (idx, p2_ens.iloc[idx][m]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='red', fontweight='bold')

    plt.xticks(rotation=45)
    plt.xlabel("Year Splits (Old + New)", fontsize=12)
    plt.tight_layout()
    
    save_path = out_dir / "xgb_single_vs_ensemble_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    build_plot()
