"""
Phase 1 baseline（XGB / Torch MLP）共用繪圖：年份切割折線圖 + compact 熱力圖。
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")
SAMPLING_ORDER = ["none", "undersampling", "oversampling", "hybrid"]

METHOD_ORDER_XGB = ["Old", "New", "Retrain", "Finetune"]
METHOD_COLORS_XGB = {
    "Old": "#5B8DB8",
    "New": "#E07B54",
    "Retrain": "#6DBF8E",
    "Finetune": "#9B6DBF",
}

# Torch MLP bankruptcy 實驗與 XGB 相同四策略（見 bankruptcy_year_splits_torch_mlp.py）
METHOD_ORDER_MLP = ["Old", "New", "Retrain", "Finetune"]
METHOD_COLORS_MLP = {
    "Old": "#5B8DB8",
    "New": "#E07B54",
    "Retrain": "#6DBF8E",
    "Finetune": "#9B6DBF",
    # 舊版 sklearn MLP 腳本或過期 raw 若仍含「Old+New」：
    "Old+New": "#3D9970",
}


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


_style()


def parse_new_years(split_id: str) -> int | None:
    m = SPLIT_RE.match(str(split_id).strip())
    if not m:
        return None
    return int(m.group(2))


def dataset_label_from_raw_stem(stem: str, raw_suffix: str) -> str:
    if stem.startswith("bankruptcy"):
        return "Bankruptcy"
    if stem.startswith("medical"):
        return "Medical"
    if stem.startswith("stock"):
        return "Stock"
    return stem.replace(raw_suffix, "").replace("_", " ").title()


def plot_year_split_lines(
    df: pd.DataFrame,
    dataset_title: str,
    metrics: list[str],
    out_path: Path,
    *,
    model_title: str,
    method_order: list[str],
    method_colors: dict[str, str],
) -> None:
    df = df.copy()
    df["_ny"] = df["split"].map(parse_new_years)
    df = df.dropna(subset=["_ny"])
    splits_sorted = df.sort_values("_ny", ascending=False)["split"].unique()

    x_labels = []
    x_pos = np.arange(len(splits_sorted))
    split_to_x = {}
    for i, s in enumerate(splits_sorted):
        split_to_x[s] = i
        ny = parse_new_years(s)
        x_labels.append(f"{ny}" if ny is not None else str(s))

    n_met = len(metrics)
    n_samp = len(SAMPLING_ORDER)
    fig, axes = plt.subplots(
        n_samp,
        n_met,
        figsize=(3.6 * n_met + 1, 2.8 * n_samp + 0.5),
        squeeze=False,
        sharex=True,
    )

    for si, sampling in enumerate(SAMPLING_ORDER):
        sub_s = df[df["sampling"] == sampling]
        for mi, metric in enumerate(metrics):
            ax = axes[si, mi]
            if metric not in sub_s.columns:
                ax.set_visible(False)
                continue
            for method in method_order:
                msub = sub_s[sub_s["method"] == method].drop_duplicates(subset=["split"])
                if msub.empty:
                    continue
                xs, ys = [], []
                for _, row in msub.iterrows():
                    s = row["split"]
                    if s in split_to_x:
                        xs.append(split_to_x[s])
                        ys.append(float(row[metric]))
                if not xs:
                    continue
                order = np.argsort(xs)
                xs = np.array(xs)[order]
                ys = np.array(ys)[order]
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    color=method_colors.get(method, "#333333"),
                    linewidth=1.6,
                    markersize=4,
                    label=method,
                )
            ax.set_xticks(x_pos)
            if si == n_samp - 1:
                ax.set_xticklabels(x_labels, rotation=0)
                ax.set_xlabel("New-window span (years)")
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(metric)
            ax.set_title(f"{sampling}")
            if si == 0 and mi == n_met - 1:
                ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    fig.suptitle(
        f"Phase 1 {model_title} — {dataset_title} (year splits)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_compact_heatmaps(
    summary_path: Path,
    out_prefix: Path,
    *,
    model_title: str,
    method_order: list[str],
    project_root: Path,
) -> None:
    df = pd.read_csv(summary_path)
    if not {"metric", "method"}.issubset(df.columns):
        return
    sampling_cols = [c for c in df.columns if c not in ("metric", "method")]
    metrics = df["metric"].unique()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1, 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        sub = df[df["metric"] == metric].set_index("method")[sampling_cols]
        sub = sub.reindex([m for m in method_order if m in sub.index])
        col_order = [c for c in ["hybrid", "none", "oversampling", "undersampling"] if c in sub.columns]
        sub = sub[col_order]
        sns.heatmap(
            sub.astype(float),
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0 if metric not in ("Type1_Error", "Type2_Error") else None,
            ax=ax,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(metric)
        ax.set_xlabel("Sampling")
        ax.set_ylabel("Method")
    fig.suptitle(
        f"Phase 1 {model_title} — mean over splits (compact summary)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out = out_prefix.parent / f"{out_prefix.name}_compact_heatmaps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out.relative_to(project_root)}")
