"""Exploratory 2/3/5/7-year sensitivity with paired newest/FIFO3 predictions.

Run with the project Python environment. Existing artifacts are never replaced.
All windows use the same fully initialized test years; thresholds use t-1 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_flexible import rolling_bankruptcy_overlap_ensemble as base
from src.utils import get_logger, set_seed

WIDTHS = (2, 3, 5, 7)
SEEDS = (42, 123, 456, 789, 2024, 7, 21, 84, 168, 999)
POOL_SIZE = 3
OUTPUT = ROOT / "results/phase_flexible/study3b_window_sensitivity/runs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_rows(min_year: int, max_year: int, widths=WIDTHS) -> list[dict]:
    # Oldest of K models ends at t-K-1, starts at t-K-L.
    first_test = min_year + max(widths) + POOL_SIZE
    if first_test > max_year:
        raise ValueError("Insufficient years to initialize the longest full pool")
    return [
        {"window_years": width, "test_feature_year": year,
         "validation_feature_year": year - 1, "newest_train_end": year - 2,
         "newest_train_start": year - width - 1,
         "pool_history_start": year - width - POOL_SIZE,
         "pool_history_years": width + POOL_SIZE - 1,
         "active_model_ends": ",".join(map(str, base.model_window_ends(year, POOL_SIZE))),
         "pool_size": POOL_SIZE, "test_used_for_selection": False}
        for width in widths for year in range(first_test, max_year + 1)
    ]


def run_seed(data, features, protocol, seed, output, logger):
    set_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    metrics, audits, predictions = [], [], []
    for width, rows in protocol.groupby("window_years", sort=True):
        pool = {}
        for row in rows.itertuples(index=False):
            year = row.test_feature_year
            ends = base.model_window_ends(year, POOL_SIZE)
            for retired in set(pool) - set(ends):
                del pool[retired]
            added = []
            for end in ends:
                if end not in pool:
                    start = perf_counter()
                    pool[end] = base._fit_window_model(
                        data, features, base.window_years(end, width),
                        f"W{width}_M{end}_S{seed}", logger, seed=seed,
                    )
                    added.append(end)
                    audits.append({**pool[end].audit, "window_years": width,
                                   "added_for_test_year": year,
                                   "fit_seconds": perf_counter() - start})
            assert len(pool) == POOL_SIZE
            assert len(added) == (POOL_SIZE if year == rows.test_feature_year.min() else 1)
            assert all(max(model.train_years) < year - 1 for model in pool.values())
            validation = data.loc[data.fyear == year - 1]
            test = data.loc[data.fyear == year]
            if validation.target.nunique() != 2 or test.target.nunique() != 2:
                raise ValueError("Each validation/test year must contain both classes")
            val_p = [pool[end].predict_proba(validation, features) for end in ends]
            test_p = [pool[end].predict_proba(test, features) for end in ends]
            for method, vp, tp in (
                ("Newest", val_p[-1], test_p[-1]),
                ("FIFO3", np.mean(val_p, axis=0), np.mean(test_p, axis=0)),
            ):
                result, pred = base._evaluate(method, year, validation, test, vp, tp)
                metadata = {"window_years": width, "seed": seed}
                metrics.append({**result, **metadata})
                predictions.append(pd.DataFrame(pred).assign(**metadata))
            logger.info(f"Completed seed={seed}, window={width}, test={year}")
    yearly = pd.DataFrame(metrics)
    pred = pd.concat(predictions, ignore_index=True)
    keys = ["window_years", "method", "test_feature_year", "source_row_id"]
    if pred.duplicated(keys).any():
        raise AssertionError("Duplicate prediction keys")
    reference = None
    for _, group in pred.groupby(["window_years", "method"]):
        paired = group.sort_values(["test_feature_year", "source_row_id"])[
            ["test_feature_year", "source_row_id", "company_name", "y_true"]
        ].reset_index(drop=True)
        if reference is not None and not paired.equals(reference):
            raise AssertionError("Configurations do not share identical test observations")
        reference = paired
    pooled = pd.concat([
        base._pooled_summary(group).assign(window_years=width, seed=seed)
        for width, group in pred.groupby("window_years")
    ], ignore_index=True)
    yearly.to_csv(output / "by_year.csv", index=False)
    pred.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    pooled.to_csv(output / "pooled.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "training_audit.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Seeds must be distinct uint32 values")
    data, features, annual = base.load_event_data()
    protocol = pd.DataFrame(protocol_rows(int(data.fyear.min()), int(data.fyear.max())))
    run = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    print(f"RUN_DIR={run}", flush=True)
    sources = [Path(__file__), Path(base.__file__), ROOT / "src/models/xgboost_wrapper.py",
               ROOT / "src/evaluation/metrics.py", ROOT / "config/model_config.yaml",
               ROOT / "config/sampling_config.yaml"]
    manifest = {
        "status": "running", "protocol": "window_sensitivity_2357_newest_fifo3_v1_exploratory",
        "started_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "window_years": WIDTHS, "pool_size": POOL_SIZE, "seeds": args.seeds,
        "test_years": sorted(protocol.test_feature_year.unique().tolist()),
        "target_definition": "failed company final observed fyear; event=fyear+1",
        "limitations": "Reconstructed labels; report and label availability dates unverified; post-hoc exploratory analysis",
        "ensemble": "equal probability average, FIFO, one new model per annual update",
        "thresholds": "t-1 validation F1 grid 0.01..0.99; separate validation FPR<=0.05 rule",
        "test_used_for_selection": False, "features": features, "sampling": "TomekLinks training only",
        "data_sha256": sha256(base.US_CSV),
        "source_hashes": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost")},
    }
    manifest_path = run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    protocol.to_csv(run / "protocol.csv", index=False)
    annual.to_csv(run / "label_audit.csv", index=False)
    logger = get_logger("WindowSensitivity", console=True, file=False)
    try:
        for seed in args.seeds:
            run_seed(data, features, protocol, seed, run / f"seed_{seed}", logger)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
