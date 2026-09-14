"""Consolidate Study 3B imbalance-treatment runs and quantify paired uncertainty."""
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

TREATMENTS = ("none", "tomek", "scale_pos_weight")
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


def _parse_run(value: str) -> tuple[str, Path]:
    try:
        treatment, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--run must use treatment=RUN_DIR") from error
    if treatment not in TREATMENTS:
        raise argparse.ArgumentTypeError(f"unknown treatment {treatment!r}")
    path = Path(raw_path).resolve()
    if not (path / "run_manifest.json").is_file():
        raise argparse.ArgumentTypeError(f"missing run_manifest.json under {path}")
    return treatment, path


def _load_seed_file(run_dir: Path, filename: str) -> pd.DataFrame:
    seed_dirs = sorted(path for path in run_dir.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise ValueError(f"No seed directories found under {run_dir}")
    frames = [pd.read_csv(path / filename) for path in seed_dirs]
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase_flexible" / "study3b_sampling_ablation" / "analysis",
    )
    args = parser.parse_args(argv)
    runs = dict(args.run)
    if set(runs) != set(TREATMENTS) or len(args.run) != len(TREATMENTS):
        parser.error(f"Provide each treatment exactly once: {TREATMENTS}")
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")

    pooled_frames = []
    yearly_frames = []
    prediction_frames = []
    audit_frames = []
    manifests = {}
    for treatment in TREATMENTS:
        run_dir = runs[treatment]
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"Run {run_dir} is not completed")
        if manifest.get("imbalance_method") != treatment:
            raise ValueError(f"Run {run_dir} does not declare {treatment}")
        manifests[treatment] = manifest
        for collection, filename in (
            (pooled_frames, "study3b_pooled_summary.csv"),
            (yearly_frames, "study3b_by_year.csv"),
            (prediction_frames, "study3b_predictions.csv"),
            (audit_frames, "study3b_window_audit.csv"),
        ):
            frame = _load_seed_file(run_dir, filename)
            frame["imbalance_method"] = treatment
            collection.append(frame)

    contract_fields = (
        "protocol_version",
        "target_definition",
        "data_sha256",
        "feature_columns",
        "window_years",
        "pool_size",
        "ensemble_weighting",
        "validation_fpr_budget",
        "test_years",
    )
    for field in contract_fields:
        values = {json.dumps(manifest[field], sort_keys=True) for manifest in manifests.values()}
        if len(values) != 1:
            raise ValueError(f"Runs differ on controlled field {field}: {values}")

    pooled = pd.concat(pooled_frames, ignore_index=True)
    yearly = pd.concat(yearly_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    audits = pd.concat(audit_frames, ignore_index=True)

    reference = pooled.loc[pooled.imbalance_method == "none", ["method", *METRICS]].copy()
    deltas = pooled[["imbalance_method", "method", *METRICS]].merge(
        reference, on="method", suffixes=("", "_none")
    )
    for metric in METRICS:
        deltas[f"delta_{metric}_vs_none"] = deltas[metric] - deltas[f"{metric}_none"]
    deltas = deltas[[
        "imbalance_method", "method", *METRICS,
        *(f"delta_{metric}_vs_none" for metric in METRICS),
    ]]

    bmodel = predictions.loc[predictions.method == "B_model_FIFO3_equal"].copy()
    bmodel["method"] = "B_model_FIFO3_equal__" + bmodel["imbalance_method"].astype(str)
    bootstrap = paired_company_cluster_bootstrap(
        bmodel,
        reference_method="B_model_FIFO3_equal__tomek",
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )

    fitted = audits.loc[
        (audits.route != "B_model") | (audits.pool_action == "added")
    ].copy()
    fitted["n_majority_removed"] = fitted.n_negative_before - fitted.n_negative_after
    audit_summary = fitted.groupby(
        ["imbalance_method", "route"], dropna=False, as_index=False
    ).agg(
        n_fitted_models=("model_id", "nunique"),
        n_positive_before=("n_positive_before", "sum"),
        n_positive_after=("n_positive_after", "sum"),
        n_negative_before=("n_negative_before", "sum"),
        n_negative_after=("n_negative_after", "sum"),
        n_majority_removed=("n_majority_removed", "sum"),
        min_scale_pos_weight=("scale_pos_weight", "min"),
        max_scale_pos_weight=("scale_pos_weight", "max"),
    )

    output_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "sampling_ablation_pooled_summary.csv": pooled,
        "sampling_ablation_by_year.csv": yearly,
        "sampling_ablation_deltas_vs_none.csv": deltas,
        "sampling_ablation_bmodel_company_bootstrap.csv": bootstrap,
        "sampling_ablation_training_audit.csv": audit_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.12g")

    manifest = {
        "analysis": "study3b_sampling_ablation_v1",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv if argv is None else [__file__, *argv]),
        "treatments": list(TREATMENTS),
        "controlled_fields": list(contract_fields),
        "source_runs": {name: str(path) for name, path in runs.items()},
        "source_run_manifests_sha256": {
            name: _sha256(path / "run_manifest.json") for name, path in runs.items()
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
