"""Fixed-year bankruptcy Study I/II with the reconstructed next-year event target.

Same years, pool, DESlib selectors, FS and Calibration rules as
bankruptcy_standard_dynamic.py; only the label changes. A failed company is
positive only in its final observed feature year (event in fyear + 1), so
earlier rows of the same company are negative and no row carries its future
final status. Every Old/New/DSEL/Calibration label must be available no later
than the first Test feature year.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments._shared.common_artifacts import atomic_csv, atomic_json, sha256
from experiments._shared.common_bankruptcy import US_CSV
from experiments._shared.common_modeling import SAMPLER_NAMES, SEEDS
from experiments.phase2_ensemble.dynamic.des import bankruptcy_standard_dynamic as study

PROTOCOL = "fixed_bankruptcy_standard_deslib_fit_only_fs_event_target_v1"
OUTPUT = ROOT / "results/phase2_ensemble/standard_dynamic_event/runs"
LABELS = ROOT / "data/processed/bankruptcy_event_labels_v1.csv"
REFERENCE = ROOT / "results/phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z"


def load_data(labels_path=LABELS):
    """Original rows and row ids, with the target replaced by the event ledger."""
    manifest = json.loads(labels_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["labels_sha256"] != sha256(labels_path) or manifest["raw_sha256"] != sha256(US_CSV):
        raise ValueError("Event label ledger does not match its manifest or the raw data")
    data, features = study.load_data()
    labels = pd.read_csv(labels_path)
    if labels.duplicated(["company_name", "fyear"]).any() or not set(labels.target.unique()) <= {0, 1}:
        raise ValueError("Event ledger keys must be unique and targets binary")
    merged = data.drop(columns="target").merge(
        labels[["company_name", "fyear", "target", "label_available_year"]],
        on=["company_name", "fyear"], how="left", validate="one_to_one")
    if merged.target.isna().any() or len(merged) != len(data):
        raise ValueError("Every company-year row needs one event label")
    merged["target"] = merged.target.astype(int)
    merged["label_available_year"] = merged.label_available_year.astype(int)
    merged = merged.set_index("source_row_id", drop=False).loc[data.index]
    return merged, features, manifest


def check_label_maturity(parts):
    """Labels used to fit, select or calibrate must exist before Test is scored."""
    first_test = int(parts["Test"].fyear.min())
    for name in ("Old_fit", "New_fit", "DSEL", "Calibration"):
        latest = int(parts[name].label_available_year.max())
        if latest > first_test:
            raise ValueError(f"{name} labels become available in {latest}, after Test starts {first_test}")
    if parts["Test"].groupby("company_name").target.sum().gt(1).any():
        raise ValueError("A company has more than one Test event")
    return {name: int(frame.label_available_year.max()) for name, frame in parts.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--fs-methods", nargs="*", choices=study.FS_METHODS, default=list(study.FS_METHODS))
    parser.add_argument("--fs-ratios", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.workers < 1 or args.k < 1 or any(not 0 < r < 1 for r in args.fs_ratios):
        parser.error("positive threads/k and ratios strictly between 0 and 1 required")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.fs_methods)) != len(args.fs_methods):
        parser.error("Duplicate seeds or feature-selection methods")
    fs_configs = [["none", 1.0]] + [list(v) for v in product(args.fs_methods, args.fs_ratios)]
    config = {"seeds": args.seeds, "fs_configs": fs_configs, "threads": args.threads,
              "workers": args.workers, "k": args.k}
    sources = [Path(__file__), Path(study.__file__), ROOT / "src/ensemble/standard_dynamic.py",
               ROOT / "experiments/_shared/common_artifacts.py",
               ROOT / "experiments/_shared/common_modeling.py",
               ROOT / "src/data/sampler.py", ROOT / "src/models/xgboost_wrapper.py",
               ROOT / "src/evaluation/metrics.py", ROOT / "config/model_config.yaml",
               ROOT / "config/sampling_config.yaml"]
    provenance = {"raw_sha256": sha256(US_CSV), "labels_sha256": sha256(LABELS),
                  "python": platform.python_version(),
                  "versions": {p: version(p) for p in ("deslib", "numpy", "pandas", "scikit-learn", "xgboost",
                                                       "imbalanced-learn", "lightgbm")},
                  "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources}}
    data, features, label_manifest = load_data()
    parts = study.partition_data(data)
    maturity = check_label_maturity(parts)
    if args.resume:
        run = args.resume.resolve()
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["provenance"] != provenance or manifest["config"] != config:
            raise ValueError("Resume refused: source/data/environment/settings changed")
    else:
        run = args.output_root.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run.mkdir(parents=True, exist_ok=False)
        manifest = {"protocol": PROTOCOL, "status": "created", "created_at": study.now(),
                    "config": config, "provenance": provenance, "command": sys.argv,
                    "features": features, "slots": study.SLOTS, "samplers": SAMPLER_NAMES,
                    "reference_run": str(REFERENCE.relative_to(ROOT)),
                    "target": "reconstructed next-year event: failed company's final observed feature year=1; all other rows 0",
                    "label_ledger": {"path": str(LABELS.relative_to(ROOT)), **{
                        k: label_manifest[k] for k in ("labels_sha256", "n_positive", "label_available_year_rule",
                                                       "evidence_status")}},
                    "label_available_year_max_by_partition": maturity,
                    "actual_fit": "Old 1999-2011; New 2012-2013; 2014 reserved for validation",
                    "validation": "2014, fixed seed42 stratified 50/50 disjoint DSEL/Calibration; not CV",
                    "representation": "impute, scale and supervised FS on Old_fit only, before sampling",
                    "selection": "static AP ranking and F1 thresholds on Calibration only; Test evaluation only",
                    "changed_from_reference": "target only; years, row ids, pool, selectors, FS and selection rules unchanged",
                    "limitations": "annual label availability is reconstructed, not filing-date verified; companies still repeat across years but earlier rows are negative; few events per partition (2014 validation has 33); seed SD is not independent-data uncertainty"}
        atomic_json(run / "manifest.json", manifest)
        assignments = pd.concat([frame[["source_row_id", "company_name", "fyear", "target", "label_available_year"]]
                                 .assign(partition=name) for name, frame in parts.items()])
        atomic_csv(run / "partitions.csv", assignments)
        atomic_csv(run / "partition_summary.csv", assignments.groupby("partition", as_index=False).agg(
            n_rows=("target", "size"), n_positive=("target", "sum"), first_year=("fyear", "min"),
            last_year=("fyear", "max"), last_label_available=("label_available_year", "max")))
    print(f"RUN {run}", flush=True)
    try:
        manifest.update(status="running", last_started_at=study.now(), pid=os.getpid())
        atomic_json(run / "manifest.json", manifest)
        jobs = []
        for seed in config["seeds"]:
            for method, ratio in config["fs_configs"]:
                tag = "no_fs" if method == "none" else f"{method}_r{round(ratio * 100)}"
                directory = run / f"seed_{seed}" / tag
                if (directory / "complete.json").exists():
                    completed = json.loads((directory / "complete.json").read_text())
                    for key, filename in (("metrics_sha256", "metrics.csv"), ("predictions_sha256", "predictions.npz")):
                        if completed[key] != sha256(directory / filename):
                            raise ValueError("Completed output hash changed")
                    print(f"REUSE seed={seed} fs={tag}", flush=True)
                    continue
                jobs.append((seed, method, ratio, tag, directory))
        with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
            futures = {executor.submit(study.run_configuration, parts, features, seed, method, ratio,
                                       config["threads"], config["k"], directory): (seed, tag)
                       for seed, method, ratio, tag, directory in jobs}
            for count, future in enumerate(as_completed(futures), 1):
                seed, tag = futures[future]
                metrics = future.result()
                atomic_json(run / "progress.json", {"seed": seed, "fs": tag, "at": study.now(),
                                                     "completed_new_jobs": count, "pending_jobs": len(jobs) - count})
                print(f"DONE {count}/{len(jobs)} seed={seed} fs={tag} rows={len(metrics)}", flush=True)
        raw = study.summarize(run, config)
        manifest.update(status="completed", completed_at=study.now(), n_metric_rows=len(raw))
        atomic_json(run / "manifest.json", manifest)
        atomic_json(run / "progress.json", {"stage": "completed", "at": study.now()})
        print(f"COMPLETED {run} rows={len(raw)}", flush=True)
    except Exception:
        manifest.update(status="failed", failed_at=study.now(), error=traceback.format_exc())
        atomic_json(run / "manifest.json", manifest)
        raise
    return run


if __name__ == "__main__":
    main()
