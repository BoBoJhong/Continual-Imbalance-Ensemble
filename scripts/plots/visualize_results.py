"""
結果視覺化腳本
讀取現有 results/ CSV 產出 4 張論文用圖表到 results/visualizations/。

圖表：
  Fig 1 — Bankruptcy 各方法 AUC 比較 (grouped bar)
  Fig 2 — 三資料集最佳 AUC 對比 (grouped bar)
  Fig 3 — Study II：有/無特徵選擇 AUC 差異 (paired bar)
  Fig 4 — Ensemble 模型數量 vs AUC 趨勢 (line chart)

使用方式：
    python scripts/plots/visualize_results.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

RESULTS_DIR = project_root / "results"
VIZ_DIR     = RESULTS_DIR / "visualizations"

# ─── matplotlib 設定 ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # 無頭模式，適合伺服器/CI
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams.update({
        "figure.dpi":       150,
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib 未安裝，請執行: pip install matplotlib")


# ─── 顏色主題 ────────────────────────────────────────────────────────────
PALETTE = {
    "baseline":  "#5B8DB8",
    "ensemble":  "#E07B54",
    "des":       "#6DBF8E",
    "feature":   "#9B6DBF",
    "no_fs":     "#5B8DB8",
    "with_fs":   "#E07B54",
}


# ─── helper ──────────────────────────────────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [SKIP] 找不到 {path.relative_to(project_root)}")
        return None
    return pd.read_csv(path, index_col=0)


def save_fig(fig, name: str):
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = VIZ_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out.relative_to(project_root)}")


# ─── Fig 1: Bankruptcy 各方法 AUC ────────────────────────────────────────
def fig1_bankruptcy_auc():
    # 合併 baseline + ensemble + DES
    frames = {}
    for csv, tag in [
        (RESULTS_DIR / "baseline" / "bankruptcy_baseline_results.csv",  "Baseline"),
        (RESULTS_DIR / "ensemble" / "bankruptcy_ensemble_results.csv",   "Ensemble"),
        (RESULTS_DIR / "des"      / "bankruptcy_des_results.csv",         "DES"),
    ]:
        df = load_csv(csv)
        if df is not None and "AUC" in df.columns:
            frames[tag] = df["AUC"]

    if not frames:
        print("  [SKIP] Fig1: 無資料")
        return

    # 取 Ensemble 最佳 3 個以避免長條太多
    all_methods, all_auc, all_colors = [], [], []
    color_map = {"Baseline": PALETTE["baseline"], "Ensemble": PALETTE["ensemble"], "DES": PALETTE["des"]}
    for tag, series in frames.items():
        if tag == "Ensemble":
            series = series.nlargest(5)
        for method, auc in series.items():
            all_methods.append(f"{method}\n({tag})")
            all_auc.append(auc)
            all_colors.append(color_map[tag])

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(all_methods)), all_auc, color=all_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(all_methods)))
    ax.set_xticklabels(all_methods, rotation=35, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Bankruptcy — AUC Comparison Across Methods")
    ax.set_ylim(max(0, min(all_auc) - 0.05), min(1.0, max(all_auc) + 0.05))
    for bar, val in zip(bars, all_auc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    # legend
    legend_handles = [mpatches.Patch(color=c, label=t) for t, c in color_map.items()]
    ax.legend(handles=legend_handles, loc="lower right")
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="AUC=0.9")
    save_fig(fig, "bankruptcy_auc_comparison.png")


# ─── Fig 2: 三資料集 best AUC ─────────────────────────────────────────────
def fig2_all_datasets():
    summary_path = RESULTS_DIR / "summary_all_datasets.csv"
    df = load_csv(summary_path)
    if df is None:
        # 嘗試從 index 重建
        print("  [SKIP] Fig2: 找不到 summary_all_datasets.csv，請先執行 scripts/analysis/compare_all_results.py")
        return
    if "dataset" not in df.columns:
        df = df.reset_index()

    datasets     = ["bankruptcy", "stock", "medical"]
    method_types = ["retrain", "ensemble_all_6", "DES_KNORAE"]
    labels       = ["Re-train", "Ensemble (All-6)", "DES KNORA-E"]
    colors       = [PALETTE["baseline"], PALETTE["ensemble"], PALETTE["des"]]
    alphas       = [1.0, 1.0, 1.0]

    x = np.arange(len(datasets))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (method, label, color, alpha) in enumerate(zip(method_types, labels, colors, alphas)):
        auc_vals = []
        for ds in datasets:
            sub = df[df["dataset"] == ds] if "dataset" in df.columns else df
            row = sub[sub.index == method] if method in sub.index else sub[sub["method"] == method] if "method" in sub.columns else pd.DataFrame()
            auc_vals.append(float(row["AUC"].values[0]) if len(row) > 0 and "AUC" in row.columns else 0.0)
        offset = (i - len(method_types)/2 + 0.5) * width
        bars = ax.bar(x + offset, auc_vals, width, label=label, color=color, alpha=alpha, edgecolor="white")
        for bar, val in zip(bars, auc_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC Comparison Across Datasets and Methods")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    save_fig(fig, "all_datasets_comparison.png")


# ─── Fig 3: Study II 有/無特徵選擇 ─────────────────────────────────────────
def fig3_feature_selection():
    fs_path = RESULTS_DIR / "feature_study" / "bankruptcy_fs_comparison.csv"
    df = load_csv(fs_path)
    if df is None:
        return
    if "combination" not in df.columns:
        df = df.reset_index().rename(columns={"index": "combination"})

    if "AUC_no_fs" not in df.columns or "AUC_with_fs" not in df.columns:
        print("  [SKIP] Fig3: CSV 格式不符，請確認 Study II 已執行")
        return

    combos = df["combination"].tolist()
    auc_no = df["AUC_no_fs"].tolist()
    auc_fs = df["AUC_with_fs"].tolist()
    x = np.arange(len(combos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width/2, auc_no, width, label="Without FS", color=PALETTE["no_fs"],  edgecolor="white")
    b2 = ax.bar(x + width/2, auc_fs, width, label="With FS",    color=PALETTE["with_fs"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("ensemble_","").replace("_"," ") for c in combos], rotation=35, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Study II — Feature Selection Impact on Ensemble AUC (Bankruptcy)")
    y_min = min(min(auc_no), min(auc_fs)) - 0.01
    ax.set_ylim(max(0.0, y_min), min(1.0, max(max(auc_no), max(auc_fs)) + 0.04))
    ax.legend()

    # 差異標記
    for xi, (no, fs) in enumerate(zip(auc_no, auc_fs)):
        diff = fs - no
        color = "#2ecc71" if diff > 0.003 else ("#e74c3c" if diff < -0.003 else "gray")
        ax.annotate(f"{diff:+.3f}", xy=(xi, max(no, fs) + 0.008),
                    ha="center", va="bottom", fontsize=7, color=color,
                    fontweight="bold" if abs(diff) > 0.003 else "normal")

    save_fig(fig, "feature_selection_impact.png")


# ─── Fig 4: Ensemble 模型數量 vs AUC ────────────────────────────────────────
def fig4_ensemble_size_trend():
    ens_path = RESULTS_DIR / "ensemble" / "bankruptcy_ensemble_results.csv"
    df = load_csv(ens_path)
    if df is None or "AUC" not in df.columns:
        return

    # 從 index 推算模型數量
    size_auc = {}
    for idx in df.index:
        s = str(idx)
        for size in [2, 3, 4, 5, 6]:
            if f"_{size}_" in s or s.endswith(f"_{size}") or f"ensemble_{size}" in s:
                if size not in size_auc:
                    size_auc[size] = []
                size_auc[size].append(df.loc[idx, "AUC"])
                break
        else:
            # 嘗試從名稱推算
            for size in [2, 3, 4, 5, 6]:
                if str(size) in s.split("_"):
                    if size not in size_auc:
                        size_auc[size] = []
                    size_auc[size].append(df.loc[idx, "AUC"])
                    break

    if not size_auc:
        # 備援：直接按 AUC 排序顯示 top-k
        print("  [NOTE] Fig4: 無法解析模型數量，改為全部方法排序圖")
        fig, ax = plt.subplots(figsize=(10, 4))
        sorted_df = df.sort_values("AUC")
        ax.barh(range(len(sorted_df)), sorted_df["AUC"], color=PALETTE["ensemble"])
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels([str(i) for i in sorted_df.index], fontsize=7)
        ax.set_xlabel("AUC-ROC")
        ax.set_title("Bankruptcy — Ensemble Combinations AUC Ranking")
        save_fig(fig, "ensemble_size_trend.png")
        return

    sizes = sorted(size_auc.keys())
    mean_auc = [np.mean(size_auc[s]) for s in sizes]
    max_auc  = [np.max(size_auc[s])  for s in sizes]
    min_auc  = [np.min(size_auc[s])  for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, mean_auc, "o-", color=PALETTE["ensemble"], linewidth=2, markersize=7, label="Mean AUC")
    ax.fill_between(sizes, min_auc, max_auc, alpha=0.15, color=PALETTE["ensemble"], label="Min–Max range")
    ax.plot(sizes, max_auc, "^--", color=PALETTE["des"], linewidth=1, markersize=5, label="Best AUC")
    ax.set_xticks(sizes)
    ax.set_xlabel("Number of Models in Ensemble")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Ensemble Size vs AUC (Bankruptcy)")
    ax.legend()
    for s, m in zip(sizes, mean_auc):
        ax.annotate(f"{m:.4f}", (s, m), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=PALETTE["ensemble"])
    save_fig(fig, "ensemble_size_trend.png")


# ─── main ────────────────────────────────────────────────────────────────
def main():
    if not HAS_MPL:
        print("請先安裝 matplotlib: pip install matplotlib")
        return

    print("產出視覺化圖表...")
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Fig 1] Bankruptcy AUC 比較...")
    fig1_bankruptcy_auc()

    print("\n[Fig 2] 三資料集比較...")
    fig2_all_datasets()

    print("\n[Fig 3] Study II 特徵選擇效果...")
    fig3_feature_selection()

    print("\n[Fig 4] Ensemble 大小 vs AUC 趨勢...")
    fig4_ensemble_size_trend()

    print(f"\n完成！所有圖表儲存於 results/visualizations/")
    saved = list(VIZ_DIR.glob("*.png"))
    for f in saved:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
