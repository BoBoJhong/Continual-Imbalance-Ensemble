"""
Compare best-per-split performance for XGB / RF / TabM.

Design goal:
  - scientific-style point plot instead of an overloaded all-settings chart
  - for each split and each metric, select the best row within each model family
  - preserve the actual chosen method / sampling in a companion CSV

Default inputs use the strongest currently available bankruptcy runs:
  - XGB tuned rerun
  - RF tuned
  - TabM tuned_light full run (complete raw)

Usage:
  python scripts/plots/visualize_phase1_best_family_pointplot.py
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
    raise ImportError(
        "This plot needs matplotlib. Run: pip install matplotlib"
    ) from e

import numpy as np
import pandas as pd


SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")
METRICS = ["AUC", "F1"]
TRAIN_START_YEAR = 1999
TRAIN_END_YEAR = 2014
MODEL_ORDER = ["XGB", "RF", "TabM"]
MODEL_COLORS = {
    "XGB": "#2C6BAA",
    "RF": "#2E8B57",
    "TabM": "#C04A3A",
}
MODEL_OFFSETS = {
    "XGB": -0.18,
    "RF": 0.00,
    "TabM": 0.18,
}
METHOD_STYLES = {
    "Old": {"marker": "o", "label": "OLD"},
    "New": {"marker": "s", "label": "NEW"},
    "Retrain": {"marker": "D", "label": "RETRAIN(ALL)"},
}
POINT_METHOD_ORDER = ["Old", "New"]


def parse_split_order(split_id: str) -> tuple[int, int]:
    m = SPLIT_RE.match(str(split_id).strip())
    if not m:
        return (10**9, 10**9)
    return (int(m.group(1)), int(m.group(2)))


def split_tick_label(split_id: str) -> str:
    m = SPLIT_RE.match(str(split_id).strip())
    if not m:
        return str(split_id)
    old_years = int(m.group(1))
    new_years = int(m.group(2))
    return f"{old_years}y\n{new_years}y"


def detect_completed_splits(df: pd.DataFrame) -> list[str]:
    completed: list[str] = []
    for split, g in df.groupby("split"):
        methods = set(g["method"])
        n_rows = len(g)
        ok = n_rows >= 8 and methods >= {"Old", "New"}
        if ok:
            completed.append(split)
    return sorted(completed, key=parse_split_order)


def _pick_best_metric_row(g: pd.DataFrame, metric: str) -> pd.Series | None:
    g = g.dropna(subset=[metric])
    if g.empty:
        return None

    # Tie-break with AUC then F1 for a stable, performance-oriented pick.
    sort_cols = [metric]
    asc = [False]
    for extra in ("AUC", "F1"):
        if extra != metric and extra in g.columns:
            sort_cols.append(extra)
            asc.append(False)
    return g.sort_values(sort_cols, ascending=asc).iloc[0]


def pick_best_split_rows(df: pd.DataFrame, model_name: str, splits: list[str]) -> pd.DataFrame:
    rows = []
    sub = df[df["split"].isin(splits) & df["method"].isin(POINT_METHOD_ORDER)].copy()
    for metric in METRICS:
        if metric not in sub.columns:
            continue
        for method in POINT_METHOD_ORDER:
            method_sub = sub[sub["method"] == method].copy()
            for split, g in method_sub.groupby("split"):
                best = _pick_best_metric_row(g, metric)
                if best is None:
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "split": split,
                        "value": float(best[metric]),
                        "method": str(best["method"]),
                        "sampling": str(best["sampling"]),
                        "plot_role": "point",
                    }
                )
    return pd.DataFrame(rows)


def pick_best_retrain_rows(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    sub = df[df["method"] == "Retrain"].copy()
    if sub.empty:
        return pd.DataFrame(rows)

    for metric in METRICS:
        if metric not in sub.columns:
            continue
        best = _pick_best_metric_row(sub, metric)
        if best is None:
            continue
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "split": "ALL",
                "value": float(best[metric]),
                "method": "Retrain",
                "sampling": str(best["sampling"]),
                "plot_role": "reference_line",
            }
        )
    return pd.DataFrame(rows)


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


def build_plot(best_df: pd.DataFrame, retrain_df: pd.DataFrame, out_path: Path, *, connect_lines: bool) -> None:
    style_plot()

    splits = sorted(best_df["split"].unique(), key=parse_split_order)
    x_pos = np.arange(len(splits))
    split_to_x = {s: i for i, s in enumerate(splits)}

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)

    for ax, metric in zip(axes, METRICS):
        metric_df = best_df[best_df["metric"] == metric].copy()
        retrain_metric_df = retrain_df[retrain_df["metric"] == metric].copy()
        for model in MODEL_ORDER:
            sub = metric_df[metric_df["model"] == model].copy()
            if sub.empty:
                continue
            for method in POINT_METHOD_ORDER:
                method_sub = sub[sub["method"] == method].copy()
                if method_sub.empty:
                    continue
                method_sub = method_sub.sort_values("split", key=lambda s: s.map(parse_split_order))
                xs = np.array([split_to_x[s] for s in method_sub["split"]], dtype=float) + MODEL_OFFSETS[model]
                ys = method_sub["value"].to_numpy(dtype=float)

                if connect_lines and len(xs) > 1:
                    ax.plot(
                        xs,
                        ys,
                        color=MODEL_COLORS[model],
                        linewidth=1.0,
                        alpha=0.25 if method == "Old" else 0.40,
                        linestyle=(0, (2, 2)) if method == "Old" else "-",
                        zorder=1,
                    )

                ax.scatter(
                    xs,
                    ys,
                    s=70,
                    color=MODEL_COLORS[model],
                    marker=METHOD_STYLES.get(method, METHOD_STYLES["Old"])["marker"],
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )

            retrain_row = retrain_metric_df[retrain_metric_df["model"] == model]
            if not retrain_row.empty:
                retrain_y = float(retrain_row.iloc[0]["value"])
                ax.hlines(
                    retrain_y,
                    xmin=x_pos.min() - 0.28,
                    xmax=x_pos.max() + 0.28,
                    colors=MODEL_COLORS[model],
                    linestyles=(0, (4, 2)),
                    linewidth=1.4,
                    alpha=0.8,
                    zorder=2,
                )

        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.set_title(f"Best-per-split {metric}: OLD vs NEW with Retrain(ALL) reference")

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

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MODEL_COLORS[m],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            label=m,
        )
        for m in MODEL_ORDER
    ]
    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=METHOD_STYLES[method]["marker"],
            color="#555555",
            linestyle="none",
            markersize=7,
            label=METHOD_STYLES[method]["label"],
        )
        for method in POINT_METHOD_ORDER
    ]
    if not retrain_df.empty:
        method_handles.append(
            plt.Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=(0, (4, 2)),
                linewidth=1.4,
                label=METHOD_STYLES["Retrain"]["label"],
            )
        )

    leg1 = axes[0].legend(
        handles=model_handles,
        title="Model family",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    axes[0].add_artist(leg1)
    axes[0].legend(
        handles=method_handles,
        title="Best row method",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.48),
    )

    fig.suptitle(
        "Phase 1 Bankruptcy Baselines: best-per-split comparison\n"
        "Each split shows the best OLD row and best NEW row; dashed lines show RETRAIN(ALL) as a global reference",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.84, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xgb-raw",
        default=str(
            project_root
            / "results"
            / "phase1_baseline"
            / "xgb"
            / "rerun_20260406_gridplus_n24"
            / "tuned"
            / "bankruptcy_year_splits_xgb_raw.csv"
        ),
    )
    ap.add_argument(
        "--rf-raw",
        default=str(
            project_root
            / "results"
            / "phase1_baseline"
            / "random_forest"
            / "tuned"
            / "bankruptcy_year_splits_rf_raw.csv"
        ),
    )
    ap.add_argument(
        "--tabm-raw",
        default=str(
            project_root
            / "results"
            / "phase1_baseline"
            / "tabm"
            / "tuned_light_full_20260410"
            / "tuned"
            / "bankruptcy_year_splits_tabm_raw.csv"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(project_root / "results" / "phase1_baseline" / "model_comparison" / "plots"),
    )
    ap.add_argument(
        "--no-lines",
        action="store_true",
        help="Only draw points; disable the faint helper lines connecting same-model splits.",
    )
    args = ap.parse_args()

    xgb_df = pd.read_csv(args.xgb_raw)
    rf_df = pd.read_csv(args.rf_raw)
    tabm_df = pd.read_csv(args.tabm_raw)

    xgb_splits = detect_completed_splits(xgb_df)
    rf_splits = detect_completed_splits(rf_df)
    tabm_splits = detect_completed_splits(tabm_df)

    plotted_splits = sorted(set(xgb_splits) | set(rf_splits) | set(tabm_splits), key=parse_split_order)

    if not plotted_splits:
        raise SystemExit("No completed splits found in the selected raw files.")

    best_df = pd.concat(
        [
            pick_best_split_rows(xgb_df, "XGB", xgb_splits),
            pick_best_split_rows(rf_df, "RF", rf_splits),
            pick_best_split_rows(tabm_df, "TabM", tabm_splits),
        ],
        ignore_index=True,
    )
    retrain_df = pd.concat(
        [
            pick_best_retrain_rows(xgb_df, "XGB"),
            pick_best_retrain_rows(rf_df, "RF"),
            pick_best_retrain_rows(tabm_df, "TabM"),
        ],
        ignore_index=True,
    )

    out_dir = Path(args.output_dir)
    csv_path = out_dir / "bankruptcy_best_per_split_xgb_rf_tabm_points.csv"
    fig_path = out_dir / "bankruptcy_best_per_split_xgb_rf_tabm_points.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_df = pd.concat([best_df, retrain_df], ignore_index=True)
    export_df = export_df.sort_values(["plot_role", "metric", "split", "model"]).reset_index(drop=True)
    export_df.to_csv(csv_path, index=False, float_format="%.6f")
    build_plot(best_df, retrain_df, fig_path, connect_lines=not args.no_lines)

    print(f"[SAVED] {csv_path.relative_to(project_root)}")
    print(f"[SAVED] {fig_path.relative_to(project_root)}")
    print("Plotted splits:", ", ".join(plotted_splits))
    print("XGB splits:", ", ".join(xgb_splits))
    print("RF splits:", ", ".join(rf_splits))
    print("TabM splits:", ", ".join(tabm_splits))


if __name__ == "__main__":
    main()
