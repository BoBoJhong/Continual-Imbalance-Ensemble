import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
phase1_path = root / "results/phase1_baseline/xgb/rerun_20260406_gridplus_n24/tuned/bankruptcy_year_splits_xgb_raw.csv"
phase2_static_path = root / "results/phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv"
phase2_tuned_path = root / "results/phase2_ensemble/dynamic/des/tuned_des_rerun_20260406/xgb_oldnew_ensemble_des_thesis_format_bankruptcy.csv"
phase2_dcs_path = root / "results/phase2_ensemble/dynamic/dcs/xgb_oldnew_ensemble_dcs_thesis_format_bankruptcy.csv"
phase3_static_path = root / "results/phase3_feature/xgb_bankruptcy_fs_full_static.csv"
phase3_des_path = root / "results/phase3_feature/xgb_bankruptcy_fs_full_des.csv"
phase3_adv_path = root / "results/phase3_feature/xgb_bankruptcy_fs_advanced_raw.csv"

out_dir = root / "results/phase3_feature/plots"
out_dir.mkdir(parents=True, exist_ok=True)

SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")

def parse_split_order(split_id):
    m = SPLIT_RE.match(str(split_id).strip())
    if not m: return (99, 99)
    return (int(m.group(1)), int(m.group(2)))

def p1_val_selected_by_tune(df1, splits, metrics):
    """One row per split: max tune_val_auc; return test AUC/F1 for that config."""
    g = df1.dropna(subset=["tune_val_auc"])
    idx = g.groupby("split", sort=False)["tune_val_auc"].idxmax()
    p1 = g.loc[idx, ["split"] + metrics].set_index("split").reindex(splits).reset_index()
    return p1

def build_all_methods_evolution_plot():
    df1 = pd.read_csv(phase1_path)
    df2_s = pd.read_csv(phase2_static_path)
    df2_t = pd.read_csv(phase2_tuned_path)
    df2_dcs = pd.read_csv(phase2_dcs_path)
    df3_s = pd.read_csv(phase3_static_path)
    df3_d = pd.read_csv(phase3_des_path)
    df3_adv = pd.read_csv(phase3_adv_path)

    splits = sorted(df1["split"].unique(), key=parse_split_order)
    metrics = ["AUC", "F1"]

    p1_best = df1.groupby("split").agg({m: "max" for m in metrics}).reindex(splits).reset_index()
    p1_val = p1_val_selected_by_tune(df1, splits, metrics)

    p2_static = (
        df2_s[(df2_s["ensemble"] == "6models") & (df2_s["type"] == "k_subsets_mean")]
        .set_index("split")
        .reindex(splits)
        .reset_index()
    )
    p2_tuned_des = df2_t.rename(columns={"split_id": "split"}).set_index("split").reindex(splits).reset_index()
    p2_dcs = df2_dcs.rename(columns={"split_id": "split"}).set_index("split").reindex(splits).reset_index()

    p3_static_best = df3_adv.groupby("split").agg({m: "max" for m in metrics}).reindex(splits).reset_index()
    p3_dynamic_best = df3_d.groupby("split").agg({m: "max" for m in metrics}).reindex(splits).reset_index()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

    colors = {
        "P1 max over grid (per metric)": "#7F7F7F",
        "P1 val-selected (tune_val_auc)": "#4A4A4A",
        "Simple Static Ensemble": "#1F77B4",
        "Tuned Dynamic DES (P2)": "#D62728",
        "Dynamic DCS (P2)": "#2CA02C",
        "FS + Best Static (P3)": "#FF7F0E",
        "FS + Best Dynamic (P3)": "#9467BD",
    }

    for i, m in enumerate(metrics):
        ax = axes[i]

        ax.plot(
            splits,
            p1_best[m],
            label="P1 max over grid (per metric)",
            color=colors["P1 max over grid (per metric)"],
            marker="x",
            linestyle="--",
            alpha=0.6,
        )

        p1v = p1_val.dropna(subset=[m])
        ax.plot(
            p1v["split"],
            p1v[m],
            label="P1 val-selected (tune_val_auc)",
            color=colors["P1 val-selected (tune_val_auc)"],
            marker="+",
            linestyle=":",
            linewidth=2,
            alpha=0.9,
        )

        p2_s_plot = p2_static.dropna(subset=[m])
        ax.plot(
            p2_s_plot["split"],
            p2_s_plot[m],
            label="Simple Static Ensemble",
            color=colors["Simple Static Ensemble"],
            marker="o",
            linestyle=":",
            alpha=0.8,
        )

        p2_t_plot = p2_tuned_des.dropna(subset=[m])
        ax.plot(
            p2_t_plot["split"],
            p2_t_plot[m],
            label="Tuned Dynamic DES (P2)",
            color=colors["Tuned Dynamic DES (P2)"],
            marker="D",
            linestyle="-",
            linewidth=2,
        )

        p2_dcs_plot = p2_dcs.dropna(subset=[m])
        ax.plot(
            p2_dcs_plot["split"],
            p2_dcs_plot[m],
            label="Dynamic DCS (P2)",
            color=colors["Dynamic DCS (P2)"],
            marker="p",
            linestyle="-",
            linewidth=2,
        )

        p3_s_plot = p3_static_best.dropna(subset=[m])
        ax.plot(
            p3_s_plot["split"],
            p3_s_plot[m],
            label="FS + Best Static (P3)",
            color=colors["FS + Best Static (P3)"],
            marker="v",
            linestyle="-.",
            linewidth=2,
        )

        p3_d_plot = p3_dynamic_best.dropna(subset=[m])
        ax.plot(
            p3_d_plot["split"],
            p3_d_plot[m],
            label="FS + Best Dynamic (P3)",
            color=colors["FS + Best Dynamic (P3)"],
            marker="s",
            linestyle="-",
            linewidth=3,
        )

        ax.set_ylabel(m, fontsize=12, fontweight="bold")
        ax.set_title(
            f"Comprehensive Study II: Impact of FS on Static vs Dynamic Ensembles ({m})",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(loc="lower left", frameon=True, shadow=True, ncol=2, fontsize=8)

    fig.text(
        0.5,
        0.01,
        "P1 'max over grid' = per-metric max (AUC and F1 may come from different cells). "
        "P1 'val-selected' = test metrics from the row with max tune_val_auc per split.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    plt.xticks(range(len(splits)), splits, rotation=40, ha="right")
    plt.xlabel("Year Splits", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0.04, 1, 0.99))

    save_path = out_dir / "xgb_all_methods_comparison.png"
    plt.savefig(save_path, dpi=200)
    print(f"All methods plot saved to: {save_path}")

if __name__ == "__main__":
    build_all_methods_evolution_plot()
