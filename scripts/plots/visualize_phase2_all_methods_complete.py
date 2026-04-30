import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# 設定路徑
root = Path(r"c:\0_git workspace\Continual-Imbalance-Ensemble")
phase1_path = root / "results/phase1_baseline/xgb/rerun_20260406_gridplus_n24/tuned/bankruptcy_year_splits_xgb_raw.csv"
phase2_static_path = root / "results/phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv"
phase2_des_path = root / "results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_by_sampling_raw_bankruptcy.csv"

out_dir = root / "results/phase2_ensemble/plots"
out_dir.mkdir(parents=True, exist_ok=True)

SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")

def parse_split_order(split_id):
    m = SPLIT_RE.match(str(split_id).strip())
    if not m: return (99, 99)
    return (int(m.group(1)), int(m.group(2)))

def build_phase2_comprehensive_plot():
    # 1. 加載數據
    df1 = pd.read_csv(phase1_path)
    df2_s = pd.read_csv(phase2_static_path)
    df2_d = pd.read_csv(phase2_des_path)

    # 2. 獲取所有 Splits 並排序
    splits = sorted(df1['split'].unique(), key=parse_split_order)

    # 3. 提取基準線
    p1_best = df1.groupby('split').agg({'AUC': 'max', 'F1': 'max'}).reindex(splits).reset_index()
    p2_static = df2_s[(df2_s['ensemble'] == '6models') & (df2_s['type'] == 'k_subsets_mean')].set_index('split').reindex(splits).reset_index()

    # 4. 提取動態方法
    des_methods = ['Dynamic_KNORA_E', 'Dynamic_KNORA_U', 'Dynamic_DES_KNN']
    des_data = {}
    for mth in des_methods:
        subset = df2_d[(df2_d['ensemble'] == mth) & (df2_d['type'] == 'all6')].set_index('split').reindex(splits).reset_index()
        des_data[mth] = subset

    # 5. 繪圖
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

    metrics = ['AUC', 'F1']
    colors = {
        'Single XGB (Best)': '#7F7F7F',
        'Simple Static': '#1F77B4',
        'Dynamic_KNORA_E': '#2CA02C',
        'Dynamic_KNORA_U': '#9467BD',
        'Dynamic_DES_KNN': '#D62728'
    }

    for i, m in enumerate(metrics):
        ax = axes[i]
        
        # 繪製基準線 (確保 dropna 以便連線)
        ax.plot(splits, p1_best[m], label='Single XGB (Best)', color=colors['Single XGB (Best)'], marker='x', linestyle='--', alpha=0.6)
        
        p2_plot_data = p2_static.dropna(subset=[m])
        ax.plot(p2_plot_data['split'], p2_plot_data[m], label='Simple Static Ensemble', color=colors['Simple Static'], marker='o', linestyle=':', alpha=0.8)
        
        # 繪製動態方法 (關鍵：使用 dropna 讓點與點之間連線)
        for mth in des_methods:
            plot_df = des_data[mth].dropna(subset=[m])
            if not plot_df.empty:
                ax.plot(plot_df['split'], plot_df[m], label=mth, color=colors[mth], marker='s', linestyle='-', linewidth=2)
        
        ax.set_ylabel(m, fontsize=12, fontweight='bold')
        ax.set_title(f"Study I Comparison: Single vs All Ensembles ({m})", fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', frameon=True, shadow=True, ncol=2, fontsize=10)

    plt.xticks(range(len(splits)), splits, rotation=40, ha='right')
    plt.xlabel("Year Splits", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = out_dir / "xgb_phase2_all_methods_comparison.png"
    plt.savefig(save_path, dpi=200)
    print(f"Phase 2 comparison plot saved to: {save_path}")

if __name__ == "__main__":
    build_phase2_comprehensive_plot()
