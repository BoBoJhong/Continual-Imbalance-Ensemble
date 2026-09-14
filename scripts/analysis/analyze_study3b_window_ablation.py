"""Consolidate Study 3B training-window runs and quantify paired uncertainty."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase_flexible.study3_ab_fair_comparison import (  # noqa: E402
    paired_company_cluster_bootstrap,
)

WINDOW_WIDTHS = (1, 3, 5)
REFERENCE_WINDOW = 3
METHOD = "B_model_FIFO3_equal"
METRICS = (
    "PR_AUC", "AUC", "F1", "G_Mean", "Recall", "Precision",
    "Balanced_Accuracy", "Recall_at_5pctFPR", "Precision_at_5pctFPR",
    "Realized_FPR_at_5pctFPR",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_run(value: str) -> tuple[int, Path]:
    try:
        raw_width, raw_path = value.split("=", 1)
        width = int(raw_width)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--run must use WINDOW_YEARS=RUN_DIR") from error
    if width not in WINDOW_WIDTHS:
        raise argparse.ArgumentTypeError(f"unexpected window width {width}")
    path = Path(raw_path).resolve()
    if not (path / "run_manifest.json").is_file():
        raise argparse.ArgumentTypeError(f"missing run_manifest.json under {path}")
    return width, path


def _load_seed_file(run_dir: Path, filename: str) -> pd.DataFrame:
    seed_dirs = sorted(path for path in run_dir.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise ValueError(f"No seed directories found under {run_dir}")
    return pd.concat([pd.read_csv(path / filename) for path in seed_dirs], ignore_index=True)


def _select_bmodel(frame: pd.DataFrame, width: int) -> pd.DataFrame:
    selected = frame.loc[frame.method == METHOD].copy()
    if selected.empty:
        raise ValueError(f"Missing {METHOD} for window width {width}")
    selected["window_years"] = width
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "results" / "phase_flexible" / "study3b_window_ablation" / "analysis",
    )
    args = parser.parse_args(argv)
    runs = dict(args.run)
    if set(runs) != set(WINDOW_WIDTHS) or len(args.run) != len(WINDOW_WIDTHS):
        parser.error(f"Provide each window width exactly once: {WINDOW_WIDTHS}")
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")

    manifests = {}
    pooled_frames, yearly_frames, prediction_frames, audit_frames = [], [], [], []
    for width in WINDOW_WIDTHS:
        run_dir = runs[width]
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("window_years") != width:
            raise ValueError(f"Run {run_dir} is not a completed width-{width} run")
        manifests[width] = manifest
        pooled_frames.append(_select_bmodel(_load_seed_file(run_dir, "study3b_pooled_summary.csv"), width))
        yearly_frames.append(_select_bmodel(_load_seed_file(run_dir, "study3b_by_year.csv"), width))
        prediction_frames.append(_select_bmodel(_load_seed_file(run_dir, "study3b_predictions.csv"), width))
        audit = _load_seed_file(run_dir, "study3b_window_audit.csv")
        audit_frames.append(
            audit.loc[(audit.route == "B_model") & (audit.pool_action == "added")]
            .assign(window_years=width).copy()
        )

    controlled_fields = (
        "protocol_version", "target_definition", "data_sha256", "feature_columns",
        "sampling", "imbalance_method", "pool_size", "ensemble_weighting",
        "validation_fpr_budget", "test_years",
    )
    for field in controlled_fields:
        values = {json.dumps(manifest[field], sort_keys=True) for manifest in manifests.values()}
        if len(values) != 1:
            raise ValueError(f"Runs differ on controlled field {field}: {values}")

    pooled = pd.concat(pooled_frames, ignore_index=True).sort_values("window_years")
    yearly = pd.concat(yearly_frames, ignore_index=True).sort_values(
        ["window_years", "test_feature_year"]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    audits = pd.concat(audit_frames, ignore_index=True)

    baseline = pooled.loc[pooled.window_years == 1, list(METRICS)].iloc[0]
    deltas = pooled[["window_years", "method", *METRICS]].copy()
    for metric in METRICS:
        deltas[f"delta_{metric}_vs_window1"] = deltas[metric] - baseline[metric]

    predictions["method"] = "Window" + predictions.window_years.astype(str)
    bootstrap = paired_company_cluster_bootstrap(
        predictions,
        reference_method=f"Window{REFERENCE_WINDOW}",
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    audit_summary = audits.groupby("window_years", as_index=False).agg(
        n_unique_fitted_models=("model_id", "nunique"),
        earliest_train_start=("train_start", "min"),
        latest_train_end=("train_end", "max"),
        total_fit_rows_before_sampling=("n_train", "sum"),
        total_positive_before=("n_positive_before", "sum"),
        total_negative_before=("n_negative_before", "sum"),
    )

    output_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "window_ablation_pooled_summary.csv": pooled,
        "window_ablation_by_year.csv": yearly,
        "window_ablation_deltas_vs_window1.csv": deltas,
        "window_ablation_company_bootstrap.csv": bootstrap,
        "window_ablation_training_audit.csv": audit_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.12g")

    colors = {"blue": "#2563A6", "orange": "#D97706", "grey": "#4B5563"}
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = (
        (axes[0], (("AP", pooled.PR_AUC, colors["blue"], "o", "-"),), "Average precision"),
        (axes[1], (("ROC-AUC", pooled.AUC, colors["orange"], "s", "--"),), "ROC-AUC"),
        (
            axes[2],
            (
                ("F1", pooled.F1, colors["blue"], "o", "-"),
                ("Recall", pooled.Recall, colors["orange"], "s", "--"),
                ("Precision", pooled.Precision, colors["grey"], "^", ":"),
            ),
            "Validation-threshold metrics",
        ),
    )
    for axis, series, title in panels:
        for label, values, color, marker, linestyle in series:
            axis.plot(
                pooled.window_years, values, color=color, marker=marker,
                linestyle=linestyle, linewidth=2, markersize=6, label=label,
            )
            for x_value, y_value in zip(pooled.window_years, values, strict=True):
                label_offset = {"Recall": 18, "Precision": -14}.get(label, 7)
                axis.annotate(
                    f"{y_value:.3f}", (x_value, y_value), xytext=(0, label_offset),
                    textcoords="offset points", ha="center", fontsize=8, color=color,
                )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Training window (years)")
        axis.set_xticks(WINDOW_WIDTHS)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        if len(series) > 1:
            axis.legend(frameon=False, loc="best")
    axes[0].set_ylabel("Score")
    figure.suptitle("Study 3B training-window sensitivity", fontsize=15, y=0.985)
    figure.text(
        0.5, 0.91,
        "FIFO3 · Tomek Links · equal weights · pooled Test 2009–2018 · n=33,636, events=289",
        ha="center", fontsize=9, color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    figure.savefig(output_dir / "study3b_window_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "analysis": "study3b_window_ablation_v1", "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv if argv is None else [__file__, *argv]),
        "window_widths": list(WINDOW_WIDTHS), "reference_window": REFERENCE_WINDOW,
        "controlled_fields": list(controlled_fields),
        "source_runs": {str(width): str(path) for width, path in runs.items()},
        "source_run_manifests_sha256": {
            str(width): _sha256(path / "run_manifest.json") for width, path in runs.items()
        },
        "bootstrap": {
            "unit": "company_name", "paired": True,
            "replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
            "scope_note": "Captures company-cluster row uncertainty, not future-year variation.",
        },
        "output_hashes": {
            filename: _sha256(output_dir / filename) for filename in outputs
        },
        "figure_sha256": _sha256(output_dir / "study3b_window_ablation.png"),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
