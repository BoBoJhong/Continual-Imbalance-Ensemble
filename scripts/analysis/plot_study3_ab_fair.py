"""Render report figures from a completed Study 3 A/B fair-comparison run."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "B_model_FIFO3_equal": "B-model FIFO3",
    "B_data_equal": "B-data",
    "A_validation_boundary_equal": "A validation boundary",
    "Recent5y": "Recent 5y",
    "Recent3y": "Recent 3y",
    "Recent1y": "Recent 1y",
    "FullHistory": "Full history",
}
BLUE = "#2F6B9A"
ORANGE = "#D97706"
GREY = "#8A949E"
INK = "#263238"
GRID = "#D9DEE3"


def render(run_dir: Path) -> Path:
    seed_dir = run_dir / "seed_42"
    pooled = pd.read_csv(seed_dir / "fair_pooled_summary.csv")
    annual = pd.read_csv(seed_dir / "fair_by_year.csv")
    pooled["label"] = pooled.method.map(METHOD_LABELS)
    pooled = pooled.sort_values("PR_AUC", ascending=True).reset_index(drop=True)
    colors = [
        BLUE if method == "B_model_FIFO3_equal"
        else ORANGE if method == "A_validation_boundary_equal"
        else GREY
        for method in pooled.method
    ]

    fig = plt.figure(figsize=(15, 10), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.35)
    panels = [
        ("PR_AUC", "Average precision", (0.0, 0.14)),
        ("AUC", "ROC-AUC (focused scale)", (0.75, 0.88)),
        ("F1", "F1 at annual validation threshold", (0.0, 0.22)),
    ]
    for index, (column, title, limits) in enumerate(panels):
        axis = fig.add_subplot(grid[0, index])
        y = range(len(pooled))
        axis.hlines(y, limits[0], pooled[column], color=GRID, linewidth=1.2)
        axis.scatter(pooled[column], y, c=colors, s=65, edgecolor=INK, linewidth=0.5, zorder=3)
        for row_index, value in enumerate(pooled[column]):
            axis.annotate(
                f"{value:.3f}", (value, row_index), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK,
            )
        axis.set_xlim(*limits)
        axis.set_yticks(list(y), pooled.label if index == 0 else [""] * len(pooled))
        axis.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        axis.grid(axis="x", color=GRID, linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="both", colors=INK, labelsize=9)

    axis = fig.add_subplot(grid[1, :])
    styles = {
        "B_model_FIFO3_equal": (BLUE, "-", "o", "B-model FIFO3"),
        "A_validation_boundary_equal": (ORANGE, "--", "s", "A validation boundary"),
        "Recent3y": (INK, ":", "^", "Recent 3y"),
    }
    for method, (color, line_style, marker, label) in styles.items():
        frame = annual[annual.method == method].sort_values("test_feature_year")
        axis.plot(
            frame.test_feature_year, frame.PR_AUC, color=color, linestyle=line_style,
            marker=marker, linewidth=2.0, markersize=5, label=label,
        )
    axis.set_title("Annual average precision", loc="left", fontsize=11, color=INK, pad=10)
    axis.set_xlabel("Test feature year", color=INK)
    axis.set_ylabel("Average precision", color=INK)
    axis.set_xticks(sorted(annual.test_feature_year.unique()))
    axis.set_ylim(bottom=0)
    axis.grid(color=GRID, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK)
    axis.legend(frameon=False, ncol=3, loc="upper left")

    fig.suptitle(
        "Study 3 A/B fair comparison", x=0.06, y=0.98, ha="left",
        fontsize=17, fontweight="bold", color=INK,
    )
    fig.text(
        0.06, 0.945,
        "American Bankruptcy, test feature years 2009–2018; 33,636 company-years and 289 events",
        ha="left", fontsize=10, color="#56616B",
    )
    fig.text(
        0.06, 0.015,
        "All thresholds are selected on the prior validation year. The AUC panel uses a focused dot-plot scale; other panels start at zero.",
        ha="left", fontsize=9, color="#56616B",
    )
    output_dir = run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "study3_ab_fair_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    output = render(args.run_dir.resolve())
    print(output)


if __name__ == "__main__":
    main()
