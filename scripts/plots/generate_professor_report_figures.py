"""Generate presentation-ready figures for the professor progress report.

The script reads existing raw/result artifacts and writes figures to
``results/report_figures``. It never alters raw data or metric tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "report_figures"

BLUE = "#2F5D8A"
BLUE_LIGHT = "#91B3D7"
GOLD = "#C8952E"
ORANGE = "#D97745"
OLIVE = "#718355"
INK = "#24313D"
GREY = "#AAB3BB"
GRID = "#DDE3E8"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"],
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "savefig.facecolor": "white",
        }
    )


def finish(fig: plt.Figure, filename: str, source: str) -> None:
    fig.text(0.01, 0.01, f"Source: {source}", fontsize=8, color="#5E6B75")
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pooled_performance() -> None:
    source = "results/phase_flexible/rolling_bankruptcy/rolling_pooled_summary.csv"
    df = pd.read_csv(ROOT / source).sort_values("AUC", ascending=True)
    labels = df["method"].str.replace("_", " ")
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, metric, color in zip(axes, ["AUC", "F1"], [BLUE, GOLD]):
        values = df[metric]
        bars = ax.barh(y, values, color=color, edgecolor=INK, linewidth=0.6)
        ax.set_xlim(0, 0.9 if metric == "AUC" else 0.38)
        ax.set_title(metric, fontsize=13, weight="bold")
        ax.set_xlabel("Score (pooled out-of-sample predictions)")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels)
    fig.suptitle("Rolling Walk-forward Performance by Method", fontsize=16, weight="bold")
    fig.text(0.5, 0.925, "Test years 2009–2018; 33,636 pooled predictions per method", ha="center", fontsize=10)
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.20, right=0.97, wspace=0.10)
    finish(fig, "01_rolling_pooled_performance.png", source)


def annual_auc() -> None:
    source = "results/phase_flexible/rolling_bankruptcy/rolling_by_year.csv"
    df = pd.read_csv(ROOT / source)
    methods = ["New_under", "AdaptiveChoice_AUC", "DAWCE_AUC", "Equal6"]
    styles = {
        "New_under": (BLUE, "o", "-"),
        "AdaptiveChoice_AUC": (GOLD, "s", "--"),
        "DAWCE_AUC": (ORANGE, "^", "-."),
        "Equal6": (GREY, "D", ":"),
    }

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for method in methods:
        part = df[df["method"] == method].sort_values("test_year")
        color, marker, linestyle = styles[method]
        ax.plot(
            part["test_year"],
            part["AUC"],
            label=method.replace("_", " "),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
        )
    fig.suptitle("Annual AUC in Rolling Walk-forward Evaluation", fontsize=16, weight="bold", y=0.98)
    fig.text(0.5, 0.925, "Each year is tested once after validation on the preceding year", ha="center", fontsize=10)
    ax.set_xlabel("Test year")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(sorted(df["test_year"].unique()))
    ax.set_ylim(0.70, 0.93)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.10, right=0.97)
    finish(fig, "02_rolling_annual_auc.png", source)


def fair_ablation() -> None:
    source = "results/phase5_weighted/bk_fair_ablation_summary.csv"
    df = pd.read_csv(ROOT / source)
    keep = ["New_under", "AdaptiveChoice_AUC", "DAWCE_AUC", "DAWCE_F1", "Equal6"]
    df = df[(df["fs_variant"] == "no_fs") & df["method"].isin(keep)].copy()
    df = df.sort_values("test_AUC", ascending=True)
    labels = df["method"].str.replace("_", " ")
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, metric, title, color, xmax in [
        (axes[0], "test_AUC", "Test AUC", BLUE, 0.9),
        (axes[1], "test_F1", "Test F1", GOLD, 0.24),
    ]:
        colors = [color if m == "New_under" else BLUE_LIGHT if m != "Equal6" else GREY for m in df["method"]]
        bars = ax.barh(y, df[metric], color=colors, edgecolor=INK, linewidth=0.6)
        ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Mean across 15 dependent temporal splits")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in df[metric]], padding=3, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels)
    fig.suptitle("Fair Ablation without Feature Selection", fontsize=16, weight="bold")
    fig.text(0.5, 0.925, "Same model pool and preprocessing; splits share the 2015–2018 test period", ha="center", fontsize=10)
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.22, right=0.97, wspace=0.10)
    finish(fig, "03_fair_ablation_no_fs.png", source)


def feature_stability() -> None:
    source = "results/phase3_feature/stability/bankruptcy_feature_stability_summary.csv"
    df = pd.read_csv(ROOT / source)
    scopes = ["old", "new", "old_new"]
    methods = ["mutual_info", "shap", "rfe"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)

    for ax, scope in zip(axes, scopes):
        part = df[df["train_scope"] == scope]
        x = np.arange(len(methods))
        r80 = [part[(part["fs_method"] == m) & (part["fs_ratio"] == 0.8)]["jaccard_mean"].iloc[0] for m in methods]
        r50 = [part[(part["fs_method"] == m) & (part["fs_ratio"] == 0.5)]["jaccard_mean"].iloc[0] for m in methods]
        width = 0.36
        b1 = ax.bar(x - width / 2, r80, width, label="r80", color=BLUE, edgecolor=INK, linewidth=0.5)
        b2 = ax.bar(x + width / 2, r50, width, label="r50", color=GOLD, edgecolor=INK, linewidth=0.5)
        ax.set_title(scope.replace("_", " + ").title(), fontsize=12, weight="bold")
        ax.set_xticks(x, ["MI", "SHAP", "RFE"])
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.bar_label(b1, labels=[f"{v:.2f}" for v in r80], padding=2, fontsize=8, rotation=90)
        ax.bar_label(b2, labels=[f"{v:.2f}" for v in r50], padding=2, fontsize=8, rotation=90)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean pairwise Jaccard similarity")
    axes[1].legend(ncol=2, frameon=False, loc="upper center")
    fig.suptitle("Feature-selection Stability: r80 vs r50", fontsize=16, weight="bold")
    fig.text(0.5, 0.925, "Descriptive means across 105 dependent pairwise comparisons per setting", ha="center", fontsize=10)
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.98, wspace=0.12)
    finish(fig, "04_feature_stability.png", source)


def class_imbalance_by_year() -> None:
    source = "data/raw/bankruptcy/american_bankruptcy_dataset.csv"
    df = pd.read_csv(ROOT / source)
    failed = df["status_label"].astype(str).str.lower().eq("failed")
    annual = (
        pd.DataFrame({"year": df["fyear"], "failed": failed.astype(int)})
        .groupby("year", as_index=False)
        .agg(observations=("failed", "size"), failed=("failed", "sum"))
    )
    annual["failure_rate"] = annual["failed"] / annual["observations"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    axes[0].plot(annual["year"], annual["failure_rate"] * 100, color=BLUE, marker="o", linewidth=2)
    axes[0].axvspan(2015, 2018, color=GOLD, alpha=0.13, label="Main test period")
    axes[0].set_ylabel("Failure rate (%)")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].spines[["top", "right"]].set_visible(False)

    bars = axes[1].bar(annual["year"], annual["failed"], color=BLUE_LIGHT, edgecolor=INK, linewidth=0.4)
    axes[1].axvspan(2015, 2018, color=GOLD, alpha=0.13)
    axes[1].set_ylabel("Failed observations")
    axes[1].set_xlabel("Fiscal year")
    axes[1].set_xticks(annual["year"])
    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].grid(axis="y", color=GRID, linewidth=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].bar_label(bars, labels=[str(v) if i in {0, len(bars) - 1} else "" for i, v in enumerate(annual["failed"])], padding=2, fontsize=8)
    for ax in axes:
        ax.set_axisbelow(True)

    fig.suptitle("Bankruptcy Class Imbalance by Fiscal Year", fontsize=16, weight="bold")
    fig.text(0.5, 0.94, "78,682 firm-year observations, 1999–2018", ha="center", fontsize=10)
    fig.subplots_adjust(top=0.87, bottom=0.14, left=0.10, right=0.97, hspace=0.18)
    finish(fig, "05_class_imbalance_by_year.png", source)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    pooled_performance()
    annual_auc()
    fair_ablation()
    feature_stability()
    class_imbalance_by_year()
    print(f"Generated 5 figures in {OUT}")


if __name__ == "__main__":
    main()
