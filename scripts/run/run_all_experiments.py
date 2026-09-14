"""Run the maintained experiment pipeline from a validated manifest.

Use ``--list`` before a long run. The default selection includes every
maintained stage and can take several hours depending on hardware.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Experiment:
    phase: str
    path: str
    timeout_seconds: int


EXPERIMENTS = (
    Experiment("phase1", "experiments/phase1_baseline/retrain.py", 600),
    Experiment(
        "phase2",
        "experiments/phase2_ensemble/static/xgb_oldnew_bankruptcy_year_splits_static.py",
        1_800,
    ),
    Experiment(
        "phase2",
        "experiments/phase2_ensemble/dynamic/des/xgb_oldnew_bankruptcy_year_splits_des.py",
        2_700,
    ),
    Experiment(
        "phase2",
        "experiments/phase2_ensemble/dynamic/dcs/xgb_oldnew_bankruptcy_year_splits_dcs.py",
        2_700,
    ),
    Experiment("phase3", "experiments/phase3_feature/fs_study.py", 900),
    Experiment("phase3", "experiments/phase3_feature/fs_sweep.py", 3_600),
    Experiment(
        "phase3",
        "experiments/phase3_feature/feature_stability_analysis.py",
        3_600,
    ),
    Experiment(
        "phase4",
        "experiments/phase4_drift/bankruptcy_drift_stream.py",
        3_600,
    ),
    Experiment(
        "phase4",
        "experiments/phase4_drift/bankruptcy_drift_auc_signal.py",
        1_800,
    ),
    Experiment(
        "phase4",
        "experiments/phase4_drift/bankruptcy_ross_validation.py",
        3_600,
    ),
    Experiment(
        "phase4",
        "experiments/phase4_drift/bankruptcy_multi_boundary_ross.py",
        3_600,
    ),
    Experiment("phase5", "experiments/phase5_weighted/awe_comparison.py", 3_600),
    Experiment(
        "rolling",
        "experiments/phase_flexible/rolling_bankruptcy_adaptive.py",
        7_200,
    ),
    Experiment(
        "study3b",
        "experiments/phase_flexible/rolling_bankruptcy_overlap_ensemble.py",
        3_600,
    ),
    Experiment("analysis", "scripts/analysis/fair_weighted_ablation.py", 1_800),
    Experiment("analysis", "scripts/analysis/weighted_split_validation.py", 1_800),
    Experiment("analysis", "scripts/analysis/current_findings_statistical_tests.py", 600),
    Experiment("analysis", "scripts/analysis/current_findings_cost_sensitivity.py", 600),
    Experiment("analysis", "scripts/analysis/export_thesis_tables.py", 600),
    Experiment("report", "scripts/plots/generate_professor_report_figures.py", 600),
)


def _selected(phases: list[str] | None) -> list[Experiment]:
    if not phases:
        return list(EXPERIMENTS)
    requested = set(phases)
    return [experiment for experiment in EXPERIMENTS if experiment.phase in requested]


def _validate_manifest(experiments: list[Experiment]) -> list[Path]:
    missing = [
        PROJECT_ROOT / item.path for item in experiments if not (PROJECT_ROOT / item.path).is_file()
    ]
    if missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Experiment manifest contains missing files:\n{rendered}")
    return [PROJECT_ROOT / item.path for item in experiments]


def main() -> int:
    phases = sorted({item.phase for item in EXPERIMENTS})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", action="append", choices=phases)
    parser.add_argument("--list", action="store_true", help="Validate and list without running")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed or timed-out experiment",
    )
    args = parser.parse_args()

    selected = _selected(args.phase)
    paths = _validate_manifest(selected)
    for item, path in zip(selected, paths, strict=True):
        print(f"[{item.phase:8s}] {path.relative_to(PROJECT_ROOT)}")
    if args.list:
        print(f"Validated {len(selected)} maintained pipeline entries.")
        return 0

    failures: list[str] = []
    for item, path in zip(selected, paths, strict=True):
        print(f"\n{'=' * 72}\n[{item.phase}] {item.path}\n{'=' * 72}")
        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=PROJECT_ROOT,
                timeout=item.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{item.path}: timeout after {item.timeout_seconds}s")
        else:
            if completed.returncode == 0:
                continue
            failures.append(f"{item.path}: exit code {completed.returncode}")
        print(f"[FAIL] {failures[-1]}")
        if not args.continue_on_error:
            break

    if failures:
        print("\nPipeline failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"\nCompleted {len(selected)} pipeline entries successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
