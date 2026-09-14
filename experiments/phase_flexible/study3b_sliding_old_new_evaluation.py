"""Evaluate fully sliding Old2+New1 models, FIFO ensembles, and model aging.

For a window model ending in feature year ``e``:

* Old data: ``e-2, e-1``;
* New data: ``e``;
* validation-only: ``e+1``;
* primary one-step-ahead test: ``e+2``.

The origin advances one year at a time from the earliest feasible window to
the latest.  A FIFO ensemble contains the latest one, two, then at most three
window models.  Each frozen model is additionally evaluated on every later
feature year using the threshold selected once on its own validation year.
Those aging evaluations are diagnostic and are never used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._shared.common_bankruptcy import US_CSV  # noqa: E402
from experiments.phase_flexible.rolling_bankruptcy_overlap_ensemble import (  # noqa: E402
    DEFAULT_IMBALANCE_METHOD,
    FittedWindowModel,
    _fit_window_model,
    _select_f1_threshold,
    load_event_data,
)
from src.evaluation import compute_metrics  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402

PROTOCOL_VERSION = "study3b_sliding_old2_new1_fifo3_aging_v1_exploratory"
OUTPUT_ROOT = PROJECT_ROOT / "results/phase_flexible/study3b_sliding_old_new/runs"
WINDOW_WIDTH = 3
OLD_WIDTH = 2
POOL_SIZE = 3
BLUE = "#2F6B9A"
ORANGE = "#D97706"
INK = "#263238"
GREY = "#8A949E"
GRID = "#D9DEE3"
HEATMAP = LinearSegmentedColormap.from_list("research_blue", ["#F4F7FA", "#9DBBD0", BLUE])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *arguments],
        cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sliding_rounds(min_feature_year: int, max_feature_year: int) -> list[dict]:
    """Return all feasible Old2+New1/validation/test rolling origins."""
    first_new_year = min_feature_year + OLD_WIDTH
    last_new_year = max_feature_year - 2
    return [
        {
            "round": index + 1,
            "model_id": f"M{new_year}",
            "old_start": new_year - OLD_WIDTH,
            "old_end": new_year - 1,
            "old_years": f"{new_year - OLD_WIDTH},{new_year - 1}",
            "new_year": new_year,
            "train_years": f"{new_year - OLD_WIDTH},{new_year - 1},{new_year}",
            "validation_feature_year": new_year + 1,
            "primary_test_feature_year": new_year + 2,
            "primary_test_event_year": new_year + 3,
            "test_used_for_selection": False,
        }
        for index, new_year in enumerate(range(first_new_year, last_new_year + 1))
    ]


def active_model_years(new_year: int, first_new_year: int, pool_size: int = POOL_SIZE) -> tuple[int, ...]:
    """Return the warm-up/FIFO pool available for the current rolling origin."""
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    return tuple(range(max(first_new_year, new_year - pool_size + 1), new_year + 1))


def _metric_row(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        **compute_metrics(y_true, proba, y_pred=y_pred),
        "threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "alerts": int(tp + fp),
        "prevalence": float(np.mean(y_true)),
    }


def _prediction_rows(
    frame: pd.DataFrame,
    *,
    method: str,
    model_id: str,
    test_year: int,
    proba: np.ndarray,
    threshold: float,
) -> list[dict]:
    rows = []
    for source_row_id, (_, sample), probability in zip(
        frame.index, frame.iterrows(), proba, strict=True
    ):
        rows.append({
            "method": method,
            "model_id": model_id,
            "source_row_id": int(source_row_id),
            "company_name": sample.company_name,
            "test_feature_year": int(test_year),
            "test_event_year": int(test_year + 1),
            "y_true": int(sample.target),
            "y_proba": float(probability),
            "threshold": float(threshold),
            "y_pred": int(probability >= threshold),
        })
    return rows


def _pooled_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in predictions.groupby("method", sort=False):
        row = _metric_row(
            group.y_true.to_numpy(dtype=int),
            group.y_proba.to_numpy(dtype=float),
            threshold=0.5,
        )
        # Classification metrics must use the validation-selected annual decisions.
        selected = compute_metrics(
            group.y_true.to_numpy(dtype=int),
            group.y_proba.to_numpy(dtype=float),
            y_pred=group.y_pred.to_numpy(dtype=int),
        )
        tn, fp, fn, tp = confusion_matrix(
            group.y_true, group.y_pred, labels=[0, 1]
        ).ravel()
        row.update(selected)
        row.update({"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                    "alerts": int(tp + fp), "threshold": "annual_validation_selected"})
        rows.append({"method": method, "n_predictions": len(group),
                     "n_positive": int(group.y_true.sum()), **row})
    return pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)


def _render_progression(primary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, facecolor="white")
    styles = {
        "SlidingNewestModel": (ORANGE, "--", "s", "Newest Old2+New1 model"),
        "SlidingFIFO3": (BLUE, "-", "o", "FIFO ensemble (up to 3)"),
    }
    for method, (color, linestyle, marker, label) in styles.items():
        frame = primary[primary.method == method].sort_values("test_feature_year")
        axes[0].plot(frame.test_feature_year, frame.PR_AUC, color=color,
                     linestyle=linestyle, marker=marker, linewidth=2.1, label=label)
        axes[1].plot(frame.test_feature_year, frame.Recall, color=color,
                     linestyle=linestyle, marker=marker, linewidth=2.1, label=label)
    axes[0].set_title("Average precision by rolling test year", loc="left", color=INK)
    axes[0].set_ylabel("Average precision", color=INK)
    axes[1].set_title("Recall at the validation-selected F1 threshold", loc="left", color=INK)
    axes[1].set_ylabel("Recall", color=INK)
    axes[1].set_xlabel("Test feature year", color=INK)
    years = sorted(primary.test_feature_year.unique())
    axes[1].set_xticks(years)
    for axis in axes:
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=INK)
        axis.legend(frameon=False, loc="upper left")
    fig.suptitle("Study 3B fully sliding Old/New evaluation", x=0.07, y=0.99,
                 ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(
        0.07, 0.952,
        "Old=two years, New=one year, next year validation, following year test; 1999–2018 feature data",
        ha="left", fontsize=9.5, color="#56616B",
    )
    fig.text(
        0.07, 0.012,
        "The FIFO pool warms up with one and two models, then retains the latest three. Test data never select thresholds or models.",
        ha="left", fontsize=8.5, color="#56616B",
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.99, 0.93), h_pad=2.0)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_aging_heatmap(aging: pd.DataFrame, output: Path) -> None:
    matrix = aging.pivot(index="new_year", columns="test_feature_year", values="PR_AUC")
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    fig, axis = plt.subplots(figsize=(14, 9), facecolor="white")
    image = axis.imshow(values, aspect="auto", cmap=HEATMAP, vmin=0, vmax=max(0.30, float(np.nanmax(values))))
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(matrix.index)), [f"M{year}" for year in matrix.index])
    axis.set_xlabel("Future test feature year", color=INK)
    axis.set_ylabel("Frozen model (New year / window end)", color=INK)
    axis.set_title("Average precision of each frozen model on later years", loc="left", color=INK, pad=12)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            if not values.mask[row_index, column_index]:
                value = float(values[row_index, column_index])
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center",
                          fontsize=7, color="white" if value >= 0.18 else INK)
    colorbar = fig.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Average precision", color=INK)
    axis.tick_params(colors=INK)
    fig.suptitle("Study 3B frozen-model aging matrix", x=0.08, y=0.99,
                 ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(
        0.08, 0.952,
        "Each M-year model uses the preceding two Old years plus its New year; its threshold is fixed on the immediately following validation year",
        ha="left", fontsize=9.3, color="#56616B",
    )
    fig.text(
        0.08, 0.012,
        "Blank cells precede a model's first eligible test year. Later test years are diagnostic and are not used to select or retire models.",
        ha="left", fontsize=8.5, color="#56616B",
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.99, 0.93))
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_experiment(*, seed: int, output_root: Path) -> Path:
    set_seed(seed)
    logger = get_logger("Study3BSlidingOldNew", console=True, file=False)
    data, feature_cols, label_audit = load_event_data()
    min_year, max_year = int(data.fyear.min()), int(data.fyear.max())
    protocol = pd.DataFrame(sliding_rounds(min_year, max_year))
    first_new_year = int(protocol.new_year.min())

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    source_paths = [
        Path(__file__),
        PROJECT_ROOT / "experiments/phase_flexible/rolling_bankruptcy_overlap_ensemble.py",
        PROJECT_ROOT / "config/model_config.yaml",
        PROJECT_ROOT / "config/sampling_config.yaml",
    ]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "exploratory_reconstructed_event_target",
        "target_definition": "failed company final observed feature year predicts event in fyear+1",
        "label_availability": "exact filing/report dates unavailable",
        "command": sys.argv,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_at_start": _git_output("status", "--porcelain"),
        "data_sha256": _sha256(US_CSV),
        "source_hashes": {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in source_paths},
        "seed": seed,
        "window_contract": "Old=e-2,e-1; New=e; Validation=e+1; primary Test=e+2",
        "pool_contract": "warm-up then latest three frozen models, FIFO, equal probability mean",
        "aging_contract": "each frozen model evaluated on e+2..2018 using threshold selected only on e+1",
        "test_used_for_selection": False,
        "feature_columns": feature_cols,
        "imbalance_method": DEFAULT_IMBALANCE_METHOD,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "matplotlib")},
        "chart_map": [
            {
                "file": "sliding_primary_performance.png",
                "question": "Does the sliding FIFO ensemble outperform the newest rolling-window model across time?",
                "family": "Trend / highlighted two-series line",
                "grain": "16 primary out-of-time test feature years",
                "metrics": ["PR_AUC", "Recall"],
                "palette": "hard two-root blue/orange plus neutrals",
            },
            {
                "file": "frozen_model_aging_ap_heatmap.png",
                "question": "How does each frozen Old2+New1 model rank events as it ages?",
                "family": "Matrix & Cohort / triangular heatmap",
                "grain": "model window end by eligible future test feature year",
                "metric": "PR_AUC",
                "palette": "single-root blue plus neutrals",
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        models: dict[int, FittedWindowModel] = {}
        thresholds: dict[int, float] = {}
        model_audits = []
        for row in protocol.itertuples(index=False):
            years = tuple(range(row.old_start, row.new_year + 1))
            logger.info(f"Fit {row.model_id}: Old={row.old_years}, New={row.new_year}")
            fitted = _fit_window_model(
                data, feature_cols, years, row.model_id, logger,
                seed=seed, imbalance_method=DEFAULT_IMBALANCE_METHOD,
            )
            validation = data[data.fyear == row.validation_feature_year]
            validation_proba = fitted.predict_proba(validation, feature_cols)
            thresholds[row.new_year] = _select_f1_threshold(
                validation.target.to_numpy(dtype=int), validation_proba
            )
            models[row.new_year] = fitted
            model_audits.append({
                **fitted.audit,
                "old_years": row.old_years,
                "new_year": row.new_year,
                "validation_feature_year": row.validation_feature_year,
                "primary_test_feature_year": row.primary_test_feature_year,
                "model_validation_threshold": thresholds[row.new_year],
            })

        primary_rows: list[dict] = []
        primary_predictions: list[dict] = []
        aging_rows: list[dict] = []
        aging_predictions: list[dict] = []

        for row in protocol.itertuples(index=False):
            test_year = row.primary_test_feature_year
            validation = data[data.fyear == row.validation_feature_year]
            test = data[data.fyear == test_year]
            newest = models[row.new_year]
            newest_validation = newest.predict_proba(validation, feature_cols)
            newest_test = newest.predict_proba(test, feature_cols)
            newest_threshold = _select_f1_threshold(validation.target, newest_validation)
            newest_metrics = _metric_row(test.target.to_numpy(dtype=int), newest_test, newest_threshold)
            primary_rows.append({
                "round": row.round, "method": "SlidingNewestModel", "model_id": row.model_id,
                "old_years": row.old_years, "new_year": row.new_year,
                "validation_feature_year": row.validation_feature_year,
                "test_feature_year": test_year, "test_event_year": test_year + 1,
                "active_models": row.model_id, "pool_size_actual": 1,
                "n_test": len(test), "n_test_positive": int(test.target.sum()), **newest_metrics,
            })
            primary_predictions.extend(_prediction_rows(
                test, method="SlidingNewestModel", model_id=row.model_id,
                test_year=test_year, proba=newest_test, threshold=newest_threshold,
            ))

            active_years = active_model_years(row.new_year, first_new_year)
            active = [models[year] for year in active_years]
            ensemble_validation = np.mean(
                [model.predict_proba(validation, feature_cols) for model in active], axis=0
            )
            ensemble_test = np.mean(
                [model.predict_proba(test, feature_cols) for model in active], axis=0
            )
            ensemble_threshold = _select_f1_threshold(validation.target, ensemble_validation)
            ensemble_metrics = _metric_row(test.target.to_numpy(dtype=int), ensemble_test, ensemble_threshold)
            active_ids = ",".join(f"M{year}" for year in active_years)
            primary_rows.append({
                "round": row.round, "method": "SlidingFIFO3", "model_id": f"FIFO[{active_ids}]",
                "old_years": row.old_years, "new_year": row.new_year,
                "validation_feature_year": row.validation_feature_year,
                "test_feature_year": test_year, "test_event_year": test_year + 1,
                "active_models": active_ids, "pool_size_actual": len(active),
                "n_test": len(test), "n_test_positive": int(test.target.sum()), **ensemble_metrics,
            })
            primary_predictions.extend(_prediction_rows(
                test, method="SlidingFIFO3", model_id=f"FIFO[{active_ids}]",
                test_year=test_year, proba=ensemble_test, threshold=ensemble_threshold,
            ))

            for future_test_year in range(row.primary_test_feature_year, max_year + 1):
                future = data[data.fyear == future_test_year]
                proba = newest.predict_proba(future, feature_cols)
                metric = _metric_row(
                    future.target.to_numpy(dtype=int), proba, thresholds[row.new_year]
                )
                aging_rows.append({
                    "model_id": row.model_id, "old_years": row.old_years,
                    "new_year": row.new_year, "train_years": row.train_years,
                    "validation_feature_year": row.validation_feature_year,
                    "test_feature_year": future_test_year,
                    "test_event_year": future_test_year + 1,
                    "model_age_at_test": future_test_year - row.new_year,
                    "n_test": len(future), "n_test_positive": int(future.target.sum()), **metric,
                })
                aging_predictions.extend(_prediction_rows(
                    future, method="FrozenModelAging", model_id=row.model_id,
                    test_year=future_test_year, proba=proba,
                    threshold=thresholds[row.new_year],
                ))

        primary = pd.DataFrame(primary_rows)
        primary_prediction_frame = pd.DataFrame(primary_predictions)
        aging = pd.DataFrame(aging_rows)
        aging_prediction_frame = pd.DataFrame(aging_predictions)
        pooled = _pooled_summary(primary_prediction_frame)

        if len(protocol) != 16 or protocol.primary_test_feature_year.tolist() != list(range(2003, 2019)):
            raise AssertionError("Expected 16 complete rolling origins with tests 2003..2018")
        if primary.duplicated(["method", "test_feature_year"]).any():
            raise AssertionError("Duplicate primary method/year rows")
        if primary_prediction_frame.duplicated(["method", "test_feature_year", "source_row_id"]).any():
            raise AssertionError("Duplicate primary prediction keys")
        if aging_prediction_frame.duplicated(["model_id", "test_feature_year", "source_row_id"]).any():
            raise AssertionError("Duplicate aging prediction keys")
        if primary.test_feature_year.max() >= data.fyear.max() + 1:
            raise AssertionError("A primary test year is outside observed features")

        outputs = {
            "sliding_protocol.csv": protocol,
            "sliding_model_audit.csv": pd.DataFrame(model_audits),
            "sliding_primary_by_year.csv": primary,
            "sliding_primary_predictions.csv": primary_prediction_frame,
            "sliding_primary_pooled.csv": pooled,
            "frozen_model_aging_by_year.csv": aging,
            "frozen_model_aging_predictions.csv": aging_prediction_frame,
            "label_audit.csv": label_audit,
        }
        for filename, frame in outputs.items():
            frame.assign(seed=seed, protocol_version=PROTOCOL_VERSION).to_csv(
                run_dir / filename, index=False, float_format="%.12g"
            )
        _render_progression(primary, run_dir / "sliding_primary_performance.png")
        _render_aging_heatmap(aging, run_dir / "frozen_model_aging_ap_heatmap.png")
        manifest["status"] = "completed"
        manifest["n_rounds"] = len(protocol)
        manifest["primary_test_feature_years"] = [int(protocol.primary_test_feature_year.min()),
                                                   int(protocol.primary_test_feature_year.max())]
        manifest["n_aging_model_year_pairs"] = len(aging)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["output_hashes"] = {
            path.relative_to(run_dir).as_posix(): _sha256(path)
            for path in sorted(run_dir.iterdir()) if path.is_file() and path != manifest_path
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved fully sliding Study 3B run: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be in [0, 2**32)")
    print(run_experiment(seed=args.seed, output_root=args.output_root.resolve()))


if __name__ == "__main__":
    main()
