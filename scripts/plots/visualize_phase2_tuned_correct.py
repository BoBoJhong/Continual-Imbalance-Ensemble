import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# 設定路徑 - 使用正確的 Tuned 數據
root = Path(r"c:\0_git workspace\Continual-Imbalance-Ensemble")
phase1_path = root / "results/phase1_baseline/xgb/rerun_20260406_gridplus_n24/tuned/bankruptcy_year_splits_xgb_raw.csv"
phase2_static_path = root / "results/phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv"
phase2_tuned_des_path = root / "results/phase2_ensemble/dynamic/des/tuned_des_rerun_20260406/xgb_oldnew_ensemble_des_thesis_format_bankruptcy.csv"
phase2_dcs_path = root / "results/phase2_ensemble/dynamic/dcs/xgb_oldnew_ensemble_dcs_thesis_format_bankruptcy.csv"

out_dir = root / "results/phase2_ensemble/plots"
out_dir.mkdir(parents=True, exist_ok=True)

SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")

def parse_split_order(split_id):
    m = SPLIT_RE.match(str(split_id).strip())
    if not m: return (99, 99)
    return (int(m.group(1)), int(m.group(2)))

def build_phase2_correct_plot():
    # 1. 加載數據
    df1 = pd.read_csv(phase1_path)
    df_static = pd.read_csv(phase2_static_path)
    df_des = pd.read_csv(phase2_tuned_des_path)
    df_dcs = pd.read_csv(phase2_dcs_path)

    # 2. 統一 Splits 排序
    splits = sorted(df1['split'].unique(), key=parse_split_order)

    # 3. 提取基準線
    p1_best = df1.groupby('split').agg({'AUC': 'max', 'F1': 'max'}).reindex(splits).reset_index()
    p2_static = df_static[(df_static['ensemble'] == '6models') & (df_static['type'] == 'k_subsets_mean')].set_index('split').reindex(splits).reset_index()

    # 4. 提取 Tuned DES 與 DCS
    # DES 檔案列名是 split_id, AUC
    p2_des = df_des.rename(columns={'split_id': 'split'}).set_index('split').reindex(splits).reset_index()
    p2_dcs = df_dcs.rename(columns={'split_id': 'split'}).set_index('split').reindex(splits).reset_index()

    # 5. 繪圖
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

    metrics = ['AUC', 'F1']
    colors = {
        'Single XGB (Best)': '#7F7F7F',
        'Simple Static': '#1F77B4',
        'Tuned Dynamic DES': '#D62728', # 紅色
        'Dynamic DCS (OLA/LCA)': '#2CA02C' # 綠色
    }

    for i, m in enumerate(metrics):
        ax = axes[i]
        
        # Plot Single
        ax.plot(splits, p1_best[m], label='Single XGB (Best)', color=colors['Single XGB (Best)'], marker='x', linestyle='--', alpha=0.6)
        
        # Plot Static
        ax.plot(splits, p2_static[m], label='Simple Static Ensemble', color=colors['Simple Static'], marker='o', linestyle=':', alpha=0.8)
        
        # Plot Tuned DES
        ax.plot(splits, p2_des[m], label='Tuned Dynamic DES', color=colors['Tuned Dynamic DES'], marker='s', linestyle='-', linewidth=2.5)
        
        # Plot DCS
        ax.plot(splits, p2_dcs[m], label='Dynamic DCS', color=colors['Dynamic DCS (OLA/LCA)'], marker='^', linestyle='-.', linewidth=2)
        
        ax.set_ylabel(m, fontsize=12, fontweight='bold')
        ax.set_title(f"Study I Corrected: Single vs Tuned Ensembles ({m})", fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=11)
        
        # 標註贏過單模型的地方
        for idx in range(len(splits)):
            if p2_des.iloc[idx][m] > p1_best.iloc[idx][m]:
                ax.scatter(splits[idx], p2_des.iloc[idx][m], color='gold', s=150, edgecolors='black', zorder=5, label='Beat Single' if idx==0 else "")

    plt.xticks(range(len(splits)), splits, rotation=40, ha='right')
    plt.xlabel("Year Splits", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = out_dir / "xgb_phase2_tuned_comparison.png"
    plt.savefig(save_path, dpi=200)
    print(f"Corrected Phase 2 plot saved to: {save_path}")

if __name__ == "__main__":
    build_phase2_correct_plot()
