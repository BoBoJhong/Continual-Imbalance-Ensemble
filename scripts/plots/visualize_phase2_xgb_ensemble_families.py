"""
Compare Phase 2 XGB ensemble families for bankruptcy year splits.

Design goal:
  - one clean figure comparing Static / DES / DCS families
  - for each split and each metric, select the best row within each family
  - preserve the chosen ensemble / sampling in a companion CSV

Usage:
  python scripts/plots/visualize_phase2_xgb_ensemble_families.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("This plot needs matplotlib. Run: pip install matplotlib") from e

import numpy as np
import pandas as pd


SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")
METRICS = ["AUC", "F1"]
FAMILY_ORDER = ["Static", "DES", "DCS"]
FAMILY_COLORS = {
    "Static": "#2C6BAA",
    "DES": "#2E8B57",
    "DCS": "#C04A3A",
}
FAMILY_MARKERS = {
    "Static": "o",
    "DES": "s",
    "DCS": "D",
}
FAMILY_OFFSETS = {
    "Static": -0.18,
    "DES": 0.0,
    "DCS": 0.18,
}


def parse_split_order(split_id: str) -> tuple[int, int]:
    m = SPLIT_RE.match(str(split_id).strip())
    if not m:
        return (10**9, 10**9)
    return (int(m.group(1)), int(m.group(2)))


def split_tick_label(split_id: str) -> str:
    m = SPLIT_RE.match(str(split_id).strip())
    if not m:
        return str(split_id)
    return f"{int(m.group(1))}y\n{int(m.group(2))}y"


def style_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _pick_best_metric_row(g: pd.DataFrame, metric: str) -> pd.Series | None:
    g = g.dropna(subset=[metric])
    if g.empty:
        return None

    sort_cols = [metric]
    asc = [False]
    for extra in ("AUC", "F1", "Recall"):
        if extra != metric and extra in g.columns:
            sort_cols.append(extra)
            asc.append(False)
    return g.sort_values(sort_cols, ascending=asc).iloc[0]


def pick_best_family_rows(df: pd.DataFrame, family_name: str) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        for split, g in df.groupby("split"):
            best = _pick_best_metric_row(g, metric)
            if best is None:
                continue
            sampling = ""
            if "sampling_col" in best.index:
                sampling = str(best["sampling_col"])
            elif "type" in best.index:
                sampling = str(best["type"])
            rows.append(
                {
                    "family": family_name,
                    "metric": metric,
                    "split": str(split),
                    "value": float(best[metric]),
                    "ensemble": str(best.get("ensemble", "")),
                    "sampling": sampling,
                    "dataset": str(best.get("dataset", "")),
                }
            )
    return pd.DataFrame(rows)


def build_plot(best_df: pd.DataFrame, out_path: Path, *, connect_lines: bool) -> None:
    style_plot()

    splits = sorted(best_df["split"].unique(), key=parse_split_order)
    x_pos = np.arange(len(splits))
    split_to_x = {s: i for i, s in enumerate(splits)}

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.4), sharex=True)

    for ax, metric in zip(axes, METRICS):
        metric_df = best_df[best_df["metric"] == metric].copy()
        for family in FAMILY_ORDER:
            sub = metric_df[metric_df["family"] == family].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("split", key=lambda s: s.map(parse_split_order))
            xs = np.array([split_to_x[s] for s in sub["split"]], dtype=float) + FAMILY_OFFSETS[family]
            ys = sub["value"].to_numpy(dtype=float)

            if connect_lines and len(xs) > 1:
                ax.plot(
                    xs,
                    ys,
                    color=FAMILY_COLORS[family],
                    linewidth=1.2,
                    alpha=0.45,
                    zorder=1,
                )

            ax.scatter(
                xs,
                ys,
                s=72,
                color=FAMILY_COLORS[family],
                marker=FAMILY_MARKERS[family],
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
                label=family,
            )

        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.set_title(f"Best-per-split {metric}")

    axes[-1].set_xticks(x_pos)
    axes[-1].set_xticklabels([split_tick_label(s) for s in splits])
    axes[-1].set_xlabel("Old/New split sizes (years)")
    axes[-1].text(
        -0.055,
        -0.075,
        "Old",
        transform=axes[-1].transAxes,
        ha="right",
        va="center",
        fontsize=10,
    )
    axes[-1].text(
        -0.055,
        -0.125,
        "New",
        transform=axes[-1].transAxes,
        ha="right",
        va="center",
        fontsize=10,
    )

    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            color=FAMILY_COLORS[family],
            linestyle="-",
            linewidth=1.2,
            markersize=7,
            label=family,
        )
        for family in FAMILY_ORDER
        if family in set(best_df["family"].astype(str))
    ]
    axes[0].legend(handles=family_handles, title="Ensemble family", loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.suptitle(
        "Phase 2 Bankruptcy XGB Ensembles: Static vs DES vs DCS\n"
        "Each point selects the best row within the family for that split and metric",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.86, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--static-raw",
        default=str(
            project_root / "results" / "phase2_ensemble" / "static" / "xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv"
        ),
    )
    ap.add_argument(
        "--des-raw",
        default=str(
            project_root
            / "results"
            / "phase2_ensemble"
            / "dynamic"
            / "des"
            / "tuned_des_rerun_20260406"
            / "xgb_oldnew_ensemble_des_by_sampling_raw_bankruptcy.csv"
        ),
    )
    ap.add_argument(
        "--dcs-raw",
        default=str(
            project_root / "results" / "phase2_ensemble" / "dynamic" / "dcs" / "xgb_oldnew_ensemble_dcs_by_sampling_raw_bankruptcy.csv"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(project_root / "results" / "phase2_ensemble" / "model_comparison" / "plots"),
    )
    ap.add_argument(
        "--no-lines",
        action="store_true",
        help="Only draw points; disable helper lines within each family.",
    )
    args = ap.parse_args()

    static_df = pd.read_csv(args.static_raw)
    des_df = pd.read_csv(args.des_raw)
    dcs_df = pd.read_csv(args.dcs_raw)

    best_df = pd.concat(
        [
            pick_best_family_rows(static_df, "Static"),
            pick_best_family_rows(des_df, "DES"),
            pick_best_family_rows(dcs_df, "DCS"),
        ],
        ignore_index=True,
    )
    if best_df.empty:
        raise SystemExit("No phase2 ensemble rows found to plot.")

    best_df = best_df.sort_values(["metric", "split", "family"]).reset_index(drop=True)

    out_dir = Path(args.output_dir)
    csv_path = out_dir / "bankruptcy_phase2_xgb_static_des_dcs_points.csv"
    fig_path = out_dir / "bankruptcy_phase2_xgb_static_des_dcs_points.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_df.to_csv(csv_path, index=False, float_format="%.6f")
    build_plot(best_df, fig_path, connect_lines=not args.no_lines)

    print(f"[SAVED] {csv_path.relative_to(project_root)}")
    print(f"[SAVED] {fig_path.relative_to(project_root)}")
    print("Static splits:", ", ".join(sorted(static_df["split"].astype(str).unique(), key=parse_split_order)))
    print("DES splits:", ", ".join(sorted(des_df["split"].astype(str).unique(), key=parse_split_order)))
    print("DCS splits:", ", ".join(sorted(dcs_df["split"].astype(str).unique(), key=parse_split_order)))


if __name__ == "__main__":
    main()
