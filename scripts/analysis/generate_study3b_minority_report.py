"""Generate auditable Study 3B minority-class tables and static thesis figures.

The script reads saved out-of-sample predictions only; it does not refit a model
or select a new configuration on the test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results/phase_flexible/study3_ab_fair/runs/20260909T122839561377Z"
    / "seed_42/fair_predictions.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results/phase_flexible/study3b_minority_analysis/analysis"
)
METHOD = "B_model_FIFO3_equal"
BLUE = "#2F6B9A"
ORANGE = "#D97706"
INK = "#263238"
GREY = "#8A949E"
GRID = "#D9DEE3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}_tp": int(tp),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tn": int(tn),
        f"{prefix}_recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        f"{prefix}_precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        f"{prefix}_fpr": float(fp / (fp + tn)) if fp + tn else float("nan"),
        f"{prefix}_alerts": int(tp + fp),
    }


def _summarize(group: pd.DataFrame) -> dict[str, float | int]:
    y_true = group.y_true.to_numpy(dtype=int)
    result: dict[str, float | int] = {
        "n_test": int(len(group)),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, group.y_proba)),
        "roc_auc": float(roc_auc_score(y_true, group.y_proba)),
    }
    result.update(_confusion(y_true, group.y_pred_f1.to_numpy(dtype=int), "f1_threshold"))
    result.update(_confusion(y_true, group.y_pred_fpr5.to_numpy(dtype=int), "fpr_budget_threshold"))
    return result


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK)


def _render_performance(annual: pd.DataFrame, output: Path) -> None:
    years = annual.test_feature_year.to_numpy()
    labels = [f"{year}\nn+={positive}" for year, positive in zip(years, annual.n_positive)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, facecolor="white")

    axes[0].plot(
        years, annual.average_precision, color=BLUE, marker="o", linewidth=2.2,
        markersize=6,
    )
    for year, value in zip(years, annual.average_precision):
        axes[0].annotate(f"{value:.3f}", (year, value), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=8, color=INK)
    axes[0].set_ylabel("Average precision", color=INK)
    axes[0].set_ylim(0, max(0.32, annual.average_precision.max() * 1.20))
    axes[0].set_title("Annual minority-class ranking", loc="left", color=INK, fontsize=12)
    _style_axis(axes[0])

    axes[1].plot(
        years, annual.f1_threshold_recall, color=BLUE, marker="o", linewidth=2.2,
        label="Recall",
    )
    axes[1].plot(
        years, annual.f1_threshold_precision, color=ORANGE, marker="s",
        linestyle="--", linewidth=2.0, label="Precision",
    )
    axes[1].set_ylabel("Rate", color=INK)
    axes[1].set_ylim(0, max(0.55, annual[["f1_threshold_recall", "f1_threshold_precision"]].max().max() * 1.15))
    axes[1].set_xticks(years, labels)
    axes[1].set_xlabel("Test feature year and number of positive events", color=INK)
    axes[1].set_title("Minority detection at the validation-selected F1 threshold", loc="left", color=INK, fontsize=12)
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    _style_axis(axes[1])

    fig.suptitle("Study 3B annual minority-class performance", x=0.08, y=0.99,
                 ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(
        0.08, 0.955,
        "B-model FIFO3; 2009–2018 out-of-sample feature years; thresholds selected on the prior validation year",
        ha="left", fontsize=9.5, color="#56616B",
    )
    fig.text(
        0.08, 0.012,
        "Source: saved fair_predictions.csv. AP is threshold-free; recall and precision use each year's validation-selected F1 threshold.",
        ha="left", fontsize=8.5, color="#56616B",
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.99, 0.93), h_pad=2.0)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_alert_burden(annual: pd.DataFrame, output: Path) -> None:
    years = annual.test_feature_year.to_numpy()
    tp = annual.fpr_budget_threshold_tp.to_numpy()
    fp = annual.fpr_budget_threshold_fp.to_numpy()
    alerts = tp + fp

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, facecolor="white")
    axes[0].bar(years, fp, color="#D9DEE3", edgecolor=INK, linewidth=0.5, label="False alerts")
    axes[0].bar(years, tp, bottom=fp, color=BLUE, edgecolor=INK, linewidth=0.5, label="Detected events")
    for year, total, true_positive in zip(years, alerts, tp):
        axes[0].annotate(f"{true_positive}/{total}", (year, total), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=8, color=INK)
    axes[0].set_ylabel("Number of alerts", color=INK)
    axes[0].set_title("Alert composition", loc="left", color=INK, fontsize=12)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    _style_axis(axes[0])

    axes[1].plot(years, annual.fpr_budget_threshold_recall, color=BLUE, marker="o",
                 linewidth=2.2, label="Test recall")
    axes[1].plot(years, annual.fpr_budget_threshold_fpr, color=ORANGE, marker="s",
                 linestyle="--", linewidth=2.0, label="Realized test FPR")
    axes[1].axhline(0.05, color=INK, linewidth=1.1, linestyle=":",
                    label="5% validation budget reference")
    axes[1].set_ylabel("Rate", color=INK)
    axes[1].set_ylim(0, max(0.70, annual.fpr_budget_threshold_recall.max() * 1.12))
    axes[1].set_xticks(years)
    axes[1].set_xlabel("Test feature year", color=INK)
    axes[1].set_title("Detection and realized false-positive rate", loc="left", color=INK, fontsize=12)
    axes[1].legend(frameon=False, ncol=3, loc="upper left")
    _style_axis(axes[1])

    fig.suptitle("Study 3B annual alert burden", x=0.08, y=0.99,
                 ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(
        0.08, 0.955,
        "Threshold maximizes validation recall subject to validation FPR ≤ 5%; the constraint is not guaranteed on future test years",
        ha="left", fontsize=9.5, color="#56616B",
    )
    fig.text(
        0.08, 0.012,
        "Labels above bars show detected events / total alerts. Source: saved fair_predictions.csv.",
        ha="left", fontsize=8.5, color="#56616B",
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.99, 0.93), h_pad=2.0)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate(predictions_path: Path, output_root: Path) -> Path:
    predictions = pd.read_csv(predictions_path)
    required = {
        "test_feature_year", "test_event_year", "method", "source_row_id",
        "company_name", "y_true", "y_proba", "y_pred_f1", "y_pred_fpr5",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    frame = predictions.loc[predictions.method == METHOD].copy()
    if frame.empty:
        raise ValueError(f"No rows for {METHOD}")
    keys = ["test_feature_year", "source_row_id", "company_name"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate B-model prediction keys: {keys}")

    annual_rows = []
    for year, group in frame.groupby("test_feature_year", sort=True):
        row = {
            "test_feature_year": int(year),
            "test_event_year": int(group.test_event_year.iloc[0]),
        }
        row.update(_summarize(group))
        annual_rows.append(row)
    annual = pd.DataFrame(annual_rows)
    pooled = pd.DataFrame([_summarize(frame)])
    pooled.insert(0, "test_feature_years", f"{annual.test_feature_year.min()}-{annual.test_feature_year.max()}")
    pooled.insert(1, "test_event_years", f"{annual.test_event_year.min()}-{annual.test_event_year.max()}")
    if int(annual.n_test.sum()) != int(pooled.n_test.iloc[0]):
        raise AssertionError("Annual rows do not reconcile to pooled n_test")
    if int(annual.n_positive.sum()) != int(pooled.n_positive.iloc[0]):
        raise AssertionError("Annual rows do not reconcile to pooled positives")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = output_root / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    annual_path = output_dir / "study3b_bmodel_operating_by_year.csv"
    pooled_path = output_dir / "study3b_bmodel_operating_pooled.csv"
    performance_path = output_dir / "study3b_bmodel_yearly_minority_performance.png"
    burden_path = output_dir / "study3b_bmodel_yearly_alert_burden.png"
    annual.to_csv(annual_path, index=False, float_format="%.12g")
    pooled.to_csv(pooled_path, index=False, float_format="%.12g")
    _render_performance(annual, performance_path)
    _render_alert_burden(annual, burden_path)

    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "source_predictions": str(predictions_path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_predictions_sha256": _sha256(predictions_path),
        "method": METHOD,
        "selection_note": "No model fitting or test-set configuration selection; saved predictions summarized only.",
        "outputs": {path.name: _sha256(path) for path in (annual_path, pooled_path, performance_path, burden_path)},
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(generate(args.predictions.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
