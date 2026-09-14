"""Post-hoc uncertainty and stability analysis for the fully sliding Study 3B run.

This script never refits models or selects a threshold from Test.  It consumes
the frozen predictions of one completed run and reports paired uncertainty for
FIFO3 versus the newest three-year-window model, plus descriptive model-aging
summaries.  Company is the resampling unit so repeated company-years remain in
the same bootstrap cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase_flexible.study3_ab_fair_comparison import (  # noqa: E402
    paired_company_cluster_bootstrap,
)
from experiments.phase_flexible.study3b_sliding_old_new_evaluation import (  # noqa: E402
    _pooled_summary,
)

PROTOCOL_VERSION = "study3b_sliding_old_new_posthoc_v1"
REFERENCE_METHOD = "SlidingFIFO3"
COMPARISON_METHOD = "SlidingNewestModel"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scope_predictions(predictions: pd.DataFrame, minimum_year: int) -> pd.DataFrame:
    scoped = predictions[predictions.test_feature_year >= minimum_year].copy()
    expected = {REFERENCE_METHOD, COMPARISON_METHOD}
    if set(scoped.method) != expected:
        raise ValueError(f"Expected methods {sorted(expected)}")
    return scoped


def _paired_bootstrap(
    predictions: pd.DataFrame,
    *,
    scope: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    compatible = predictions.rename(columns={"y_pred": "y_pred_f1"})
    result = paired_company_cluster_bootstrap(
        compatible,
        reference_method=REFERENCE_METHOD,
        replicates=replicates,
        seed=seed,
    )
    result.insert(0, "scope", scope)
    return result


def _annual_differences(by_year: pd.DataFrame) -> pd.DataFrame:
    metrics = ["PR_AUC", "AUC", "F1", "Recall", "Precision"]
    index = ["round", "test_feature_year", "test_event_year"]
    fifo = (
        by_year[by_year.method == REFERENCE_METHOD]
        .set_index(index)
        .sort_index()
    )
    newest = (
        by_year[by_year.method == COMPARISON_METHOD]
        .set_index(index)
        .sort_index()
    )
    if not fifo.index.equals(newest.index):
        raise ValueError("FIFO and newest-model annual rows are not paired")
    rows: list[dict] = []
    for key in fifo.index:
        row = dict(zip(index, key, strict=True))
        row["pool_size_actual"] = int(fifo.loc[key, "pool_size_actual"])
        for metric in metrics:
            row[f"FIFO_minus_Newest_{metric}"] = float(
                fifo.loc[key, metric] - newest.loc[key, metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _aging_summary(aging: pd.DataFrame) -> pd.DataFrame:
    return (
        aging.groupby("model_age_at_test", as_index=False)
        .agg(
            n_model_year_pairs=("model_id", "size"),
            n_origin_models=("model_id", "nunique"),
            mean_PR_AUC=("PR_AUC", "mean"),
            median_PR_AUC=("PR_AUC", "median"),
            mean_AUC=("AUC", "mean"),
            mean_F1=("F1", "mean"),
            mean_Recall=("Recall", "mean"),
            mean_Precision=("Precision", "mean"),
        )
        .sort_values("model_age_at_test")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args(argv)
    if args.replicates < 1:
        parser.error("--replicates must be positive")

    run_dir = args.run_dir.resolve()
    prediction_path = run_dir / "sliding_primary_predictions.csv"
    by_year_path = run_dir / "sliding_primary_by_year.csv"
    aging_path = run_dir / "frozen_model_aging_by_year.csv"
    for path in (prediction_path, by_year_path, aging_path):
        if not path.is_file():
            parser.error(f"Missing required input: {path}")

    predictions = pd.read_csv(prediction_path)
    by_year = pd.read_csv(by_year_path)
    aging = pd.read_csv(aging_path)
    if predictions.duplicated(["method", "test_feature_year", "source_row_id"]).any():
        raise ValueError("Duplicate paired prediction keys")

    all_years = _scope_predictions(predictions, int(predictions.test_feature_year.min()))
    full_pool = _scope_predictions(predictions, 2005)
    bootstrap = pd.concat(
        [
            _paired_bootstrap(
                all_years,
                scope="all_rounds_2003_2018_including_FIFO_warmup",
                replicates=args.replicates,
                seed=args.seed,
            ),
            _paired_bootstrap(
                full_pool,
                scope="full_FIFO3_pool_2005_2018",
                replicates=args.replicates,
                seed=args.seed,
            ),
        ],
        ignore_index=True,
    )
    pooled = pd.concat(
        [
            _pooled_summary(all_years).assign(
                scope="all_rounds_2003_2018_including_FIFO_warmup"
            ),
            _pooled_summary(full_pool).assign(scope="full_FIFO3_pool_2005_2018"),
        ],
        ignore_index=True,
    )
    annual = _annual_differences(by_year)
    aging_summary = _aging_summary(aging)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = run_dir / "analysis" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "paired_company_cluster_bootstrap.csv": bootstrap,
        "pooled_by_scope.csv": pooled,
        "annual_FIFO_minus_Newest.csv": annual,
        "frozen_model_aging_by_age.csv": aging_summary,
    }
    for filename, frame in outputs.items():
        frame["analysis_protocol_version"] = PROTOCOL_VERSION
        frame.to_csv(output_dir / filename, index=False, float_format="%.12g")

    manifest = {
        "status": "completed",
        "analysis_protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir.relative_to(PROJECT_ROOT)),
        "source_files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in (prediction_path, by_year_path, aging_path)
        },
        "bootstrap": {
            "unit": "company_name",
            "paired": True,
            "replicates": args.replicates,
            "seed": args.seed,
            "ci": "percentile_95",
            "reference_minus_comparison": f"{REFERENCE_METHOD}-{COMPARISON_METHOD}",
        },
        "interpretation_limits": [
            "Bootstrap intervals quantify sampling uncertainty under company-cluster resampling; they do not establish universal or causal superiority.",
            "Aging-by-age summaries are cohort-confounded because fewer, older origin models contribute at large ages.",
            "All thresholds were frozen from each round's validation year; Test was not used for selection.",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
