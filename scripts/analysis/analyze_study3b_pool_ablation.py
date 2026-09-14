"""Consolidate Study 3B FIFO pool-size runs and quantify paired uncertainty."""
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

POOL_SIZES = (1, 2, 3, 5)
REFERENCE_POOL_SIZE = 3
METRICS = (
    "PR_AUC",
    "AUC",
    "F1",
    "G_Mean",
    "Recall",
    "Precision",
    "Balanced_Accuracy",
    "Recall_at_5pctFPR",
    "Precision_at_5pctFPR",
    "Realized_FPR_at_5pctFPR",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_run(value: str) -> tuple[int, Path]:
    try:
        raw_size, raw_path = value.split("=", 1)
        pool_size = int(raw_size)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--run must use POOL_SIZE=RUN_DIR") from error
    if pool_size not in POOL_SIZES:
        raise argparse.ArgumentTypeError(f"unexpected pool size {pool_size}")
    path = Path(raw_path).resolve()
    if not (path / "run_manifest.json").is_file():
        raise argparse.ArgumentTypeError(f"missing run_manifest.json under {path}")
    return pool_size, path


def _load_seed_file(run_dir: Path, filename: str) -> pd.DataFrame:
    seed_dirs = sorted(path for path in run_dir.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise ValueError(f"No seed directories found under {run_dir}")
    return pd.concat(
        [pd.read_csv(path / filename) for path in seed_dirs], ignore_index=True
    )


def _bmodel_rows(frame: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    expected_method = f"B_model_FIFO{pool_size}_equal"
    selected = frame.loc[frame.method == expected_method].copy()
    if selected.empty:
        raise ValueError(f"Missing {expected_method}")
    selected["pool_size"] = pool_size
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "phase_flexible"
        / "study3b_pool_ablation"
        / "analysis",
    )
    args = parser.parse_args(argv)
    runs = dict(args.run)
    if set(runs) != set(POOL_SIZES) or len(args.run) != len(POOL_SIZES):
        parser.error(f"Provide each pool size exactly once: {POOL_SIZES}")
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")

    pooled_frames = []
    yearly_frames = []
    prediction_frames = []
    audit_frames = []
    manifests = {}
    for pool_size in POOL_SIZES:
        run_dir = runs[pool_size]
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("status") != "completed":
            raise ValueError(f"Run {run_dir} is not completed")
        if manifest.get("pool_size") != pool_size:
            raise ValueError(f"Run {run_dir} does not declare pool size {pool_size}")
        manifests[pool_size] = manifest
        pooled_frames.append(
            _bmodel_rows(
                _load_seed_file(run_dir, "study3b_pooled_summary.csv"), pool_size
            )
        )
        yearly_frames.append(
            _bmodel_rows(_load_seed_file(run_dir, "study3b_by_year.csv"), pool_size)
        )
        prediction_frames.append(
            _bmodel_rows(
                _load_seed_file(run_dir, "study3b_predictions.csv"), pool_size
            )
        )
        audit = _load_seed_file(run_dir, "study3b_window_audit.csv")
        audit_frames.append(
            audit.loc[(audit.route == "B_model") & (audit.pool_action == "added")]
            .assign(pool_size=pool_size)
            .copy()
        )

    controlled_fields = (
        "protocol_version",
        "target_definition",
        "data_sha256",
        "feature_columns",
        "sampling",
        "imbalance_method",
        "window_years",
        "ensemble_weighting",
        "validation_fpr_budget",
        "test_years",
    )
    for field in controlled_fields:
        values = {
            json.dumps(manifest[field], sort_keys=True)
            for manifest in manifests.values()
        }
        if len(values) != 1:
            raise ValueError(f"Runs differ on controlled field {field}: {values}")

    pooled = pd.concat(pooled_frames, ignore_index=True).sort_values("pool_size")
    yearly = pd.concat(yearly_frames, ignore_index=True).sort_values(
        ["pool_size", "test_feature_year"]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    audits = pd.concat(audit_frames, ignore_index=True)

    pool_one = pooled.loc[pooled.pool_size == 1, [*METRICS]].iloc[0]
    deltas = pooled[["pool_size", "method", *METRICS]].copy()
    for metric in METRICS:
        deltas[f"delta_{metric}_vs_pool1"] = deltas[metric] - pool_one[metric]

    predictions["method"] = "B_model_FIFO" + predictions.pool_size.astype(str) + "_equal"
    bootstrap = paired_company_cluster_bootstrap(
        predictions,
        reference_method=f"B_model_FIFO{REFERENCE_POOL_SIZE}_equal",
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )

    audit_summary = audits.groupby("pool_size", as_index=False).agg(
        n_unique_fitted_models=("model_id", "nunique"),
        earliest_train_start=("train_start", "min"),
        latest_train_end=("train_end", "max"),
        total_fit_rows_before_sampling=("n_train", "sum"),
        total_positive_before=("n_positive_before", "sum"),
        total_negative_before=("n_negative_before", "sum"),
    )
    audit_summary["models_added_after_initialization"] = (
        audit_summary.n_unique_fitted_models - audit_summary.pool_size
    )

    output_dir = args.output_root / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "pool_ablation_pooled_summary.csv": pooled,
        "pool_ablation_by_year.csv": yearly,
        "pool_ablation_deltas_vs_pool1.csv": deltas,
        "pool_ablation_company_bootstrap.csv": bootstrap,
        "pool_ablation_training_audit.csv": audit_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.12g")

    manifest = {
        "analysis": "study3b_pool_ablation_v1",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv if argv is None else [__file__, *argv]),
        "pool_sizes": list(POOL_SIZES),
        "reference_pool_size": REFERENCE_POOL_SIZE,
        "controlled_fields": list(controlled_fields),
        "source_runs": {str(size): str(path) for size, path in runs.items()},
        "source_run_manifests_sha256": {
            str(size): _sha256(path / "run_manifest.json")
            for size, path in runs.items()
        },
        "analysis_source_sha256": _sha256(Path(__file__)),
        "bootstrap": {
            "unit": "company_name",
            "paired": True,
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "scope_note": "Captures company-cluster row uncertainty, not future-year variation.",
        },
        "output_hashes": {
            filename: _sha256(output_dir / filename) for filename in outputs
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
