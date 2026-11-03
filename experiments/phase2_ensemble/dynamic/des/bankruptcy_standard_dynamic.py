"""Fixed-year bankruptcy Study I/II: official DESlib and training-only FS.

Old: 1999-2011; New: 2012-2014, reserving 2014 for disjoint DSEL/calibration;
Test: 2015-2018. No cross-validation folds or test-selected configurations.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations, product
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments._shared.common_artifacts import atomic_csv, atomic_json, sha256
from experiments._shared.common_bankruptcy import US_CSV
from experiments._shared.common_modeling import (
    SAMPLER_NAMES,
    SEEDS,
    STRATEGIES,
    feature_columns,
    make_model,
)
from src.data import ImbalanceSampler
from src.ensemble.standard_dynamic import METHODS, ProbabilityClassifier, fit_selector
from src.evaluation import compute_metrics

PROTOCOL = "fixed_bankruptcy_standard_deslib_fit_only_fs_v1"
OUTPUT = ROOT / "results/phase2_ensemble/standard_dynamic/runs"
FS_METHODS = ("mi", "cart", "shap", "rfe")
SLOTS = tuple(f"{side}_{sampling}" for side, sampling in product(
    ("Old", "New"), ("under", "over", "hybrid")))
METRICS = ("AUC", "PR_AUC", "F1", "G_Mean", "Recall", "Precision",
           "Type1_Error", "Type2_Error", "Brier")


def now():
    return datetime.now(timezone.utc).isoformat()


def load_data():
    data = pd.read_csv(US_CSV)
    data["source_row_id"] = np.arange(len(data))
    data["fyear"] = data["fyear"].astype(int)
    if not set(data.status_label.unique()) <= {"alive", "failed"}:
        raise ValueError("Unexpected status_label")
    if data.duplicated(["company_name", "fyear"]).any():
        raise ValueError("Duplicate company-year rows")
    data["target"] = data.status_label.eq("failed").astype(int)
    features = [f"X{i}" for i in range(1, 19)]
    if not set(features) <= set(data.columns):
        raise ValueError("Expected X1..X18 financial features")
    return data.sort_values(["fyear", "company_name"]).set_index("source_row_id", drop=False), features


def partition_data(data):
    """Reserve the latest operating year, with one fixed disjoint validation split."""
    old = data.loc[data.fyear.between(1999, 2011)]
    new = data.loc[data.fyear.between(2012, 2013)]
    validation = data.loc[data.fyear.eq(2014)]
    test = data.loc[data.fyear.between(2015, 2018)]
    if validation.company_name.duplicated().any():
        raise ValueError("Validation must have at most one row per company")
    dsel_ids, calibration_ids = train_test_split(
        validation.index.to_numpy(), test_size=0.5, stratify=validation.target,
        random_state=42,
    )
    parts = {"Old_fit": old, "New_fit": new,
             "DSEL": validation.loc[dsel_ids].sort_index(),
             "Calibration": validation.loc[calibration_ids].sort_index(), "Test": test}
    seen = set()
    for name, frame in parts.items():
        if frame.empty or set(frame.target.unique()) != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
        ids = set(frame.index)
        if ids & seen:
            raise ValueError("Overlapping data partitions")
        seen.update(ids)
    if max(old.fyear.max(), new.fyear.max()) >= min(parts["DSEL"].fyear.min(), parts["Calibration"].fyear.min()):
        raise ValueError("Training years must precede validation")
    if validation.fyear.max() >= test.fyear.min():
        raise ValueError("Validation must precede Test")
    return parts


def fit_representation(parts, features, method, ratio, seed, threads):
    """All learned representation steps fit only on Old_fit, before sampling."""
    old = parts["Old_fit"]
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True).fit(old[features])
    scaler = StandardScaler().fit(imputer.transform(old[features]))
    transformed = {
        name: pd.DataFrame(scaler.transform(imputer.transform(frame[features])),
                           columns=features, index=frame.index)
        for name, frame in parts.items()
    }
    selected = feature_columns(transformed["Old_fit"], old.target.to_numpy(),
                               method, ratio, seed, threads)
    return {name: frame[selected] for name, frame in transformed.items()}, {
        "selected_features": selected, "selector_fit_scope": "Old_fit_only",
        "imputer_fit_scope": "Old_fit_only", "scaler_fit_scope": "Old_fit_only",
        "imputer_statistics": imputer.statistics_.tolist(), "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(), "selector_seed": seed,
    }


def select_threshold(y, probability):
    """Same 0.01..0.99 F1 grid, computed without repeated classification passes."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    thresholds = np.arange(0.01, 1.0, 0.01)
    tp = len(p[y == 1]) - np.searchsorted(np.sort(p[y == 1]), thresholds, side="left")
    fp = len(p[y == 0]) - np.searchsorted(np.sort(p[y == 0]), thresholds, side="left")
    denominator = tp + fp + y.sum()
    scores = np.divide(2 * tp, denominator, out=np.zeros(len(thresholds)), where=denominator > 0)
    return float(thresholds[int(np.argmax(scores))])


def static_candidates():
    candidates = {slot: (i,) for i, slot in enumerate(SLOTS)}
    candidates.update(Old3=(0, 1, 2), New3=(3, 4, 5), Equal6=tuple(range(6)))
    for size in range(2, 7):
        for subset in combinations(range(6), size):
            if any(i < 3 for i in subset) and any(i >= 3 for i in subset):
                candidates["Static_" + "+".join(SLOTS[i] for i in subset)] = subset
    return candidates


def evaluate_predictions(parts, probabilities, metadata):
    """Freeze thresholds and static selection using Calibration only."""
    calibration_y = parts["Calibration"].target.to_numpy()
    thresholds = {name: select_threshold(calibration_y, p[0]) for name, p in probabilities.items()}
    ranking = sorted((
        {"method": name, "calibration_AP": float(average_precision_score(calibration_y, p[0]))}
        for name, p in probabilities.items() if name.startswith("Static_")
    ), key=lambda row: (-row["calibration_AP"], row["method"]))
    selected = ranking[0]["method"]
    probabilities = {**probabilities, "ValidationSelectedStatic": probabilities[selected]}
    thresholds["ValidationSelectedStatic"] = thresholds[selected]
    # Test labels are accessed only after all decisions above are frozen.
    test = parts["Test"]
    rows = []
    for name, (calibration_p, test_p) in probabilities.items():
        threshold = thresholds[name]
        common = {**metadata, "method": name, "threshold": threshold,
                  "calibration_AP": float(average_precision_score(calibration_y, calibration_p)),
                  "selected_static": selected if name == "ValidationSelectedStatic" else ""}
        for period, mask in [("pooled", np.ones(len(test), dtype=bool))] + [
            (str(year), test.fyear.to_numpy() == year) for year in sorted(test.fyear.unique())
        ]:
            truth, probability = test.target.to_numpy()[mask], test_p[mask]
            metrics = compute_metrics(truth, probability, threshold=threshold)
            metrics["Brier"] = float(brier_score_loss(truth, probability))
            rows.append({**common, "period": period, "n_test": int(mask.sum()),
                         "n_test_positive": int(truth.sum()), **metrics})
    return pd.DataFrame(rows), {"selected_static": selected, "ranking": ranking,
                                "thresholds": thresholds, "test_used_for_selection": False}


def run_configuration(parts, features, seed, fs_method, ratio, threads, k, directory):
    begin = perf_counter()
    fs_tag = "no_fs" if fs_method == "none" else f"{fs_method}_r{round(ratio * 100)}"
    with threadpool_limits(threads):
        transformed, representation = fit_representation(parts, features, fs_method, ratio, seed, threads)
        pool, training = [], []
        for side, sampler_name in product(("Old", "New"), ("under", "over", "hybrid")):
            train = parts[f"{side}_fit"]
            sampler = ImbalanceSampler(random_state=seed)
            sampled_X, sampled_y = sampler.apply_sampling(
                transformed[f"{side}_fit"], train.target.to_numpy(), STRATEGIES[sampler_name])
            if set(np.unique(sampled_y)) != {0, 1}:
                raise ValueError("Sampling removed a class")
            model = make_model("xgb", seed, threads)
            model.fit(sampled_X, np.asarray(sampled_y))
            pool.append(ProbabilityClassifier(model, representation["selected_features"]))
            training.append({"slot": f"{side}_{sampler_name}", "sampling": sampler_name,
                             "sampler": type(sampler.sampler).__name__, "n_before": len(train),
                             "positive_before": int(train.target.sum()), "n_after": len(sampled_y),
                             "positive_after": int(np.sum(sampled_y)), "params": model.params,
                             "train_start": int(train.fyear.min()), "train_end": int(train.fyear.max())})
        base_probability = {name: np.column_stack([
            model.predict_proba(transformed[name])[:, 1] for model in pool
        ]) for name in ("DSEL", "Calibration", "Test")}
        probabilities = {
            name: tuple(base_probability[part][:, indices].mean(axis=1) for part in ("Calibration", "Test"))
            for name, indices in static_candidates().items()
        }
        selectors = {}
        for method in METHODS:
            selector = fit_selector(method, pool, transformed["DSEL"], parts["DSEL"].target,
                                    k=k, seed=seed, threads=threads)
            probabilities[method] = tuple(
                selector.predict_proba(np.asarray(transformed[name]))[:, 1]
                for name in ("Calibration", "Test"))
            if not all(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)) for p in probabilities[method]):
                raise ValueError(f"Invalid standard selector output: {method}")
            selectors[method] = selector
        metadata = {"seed": seed, "fs": fs_tag, "n_features": len(representation["selected_features"])}
        metrics, selection = evaluate_predictions(parts, probabilities, metadata)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "selection.json", selection)
        atomic_csv(directory / "metrics.csv", metrics)
        # Raw probabilities plus feature-space coordinates permit independent replay.
        cache = {}
        for name in ("DSEL", "Calibration", "Test"):
            cache[f"{name}_row_ids"] = parts[name].index.to_numpy()
            cache[f"{name}_y"] = parts[name].target.to_numpy()
            cache[f"{name}_X"] = np.asarray(transformed[name])
            cache[f"{name}_base_probability"] = base_probability[name]
        for method in METHODS:
            for i, name in enumerate(("Calibration", "Test")):
                cache[f"{name}_{method}"] = probabilities[method][i]
        np.savez_compressed(directory / "predictions.npz", **cache)
        audit = {**metadata, **representation, "training": training, "k": k,
                 "library": f"DESlib {version('deslib')}", "DFP": False, "with_IH": False,
                 "DES_voting": "soft", "DCS_selection_method": "best",
                 "unanimous_queries": "official DESlib mean-probability shortcut",
                 "DSEL_fit_scope": f"{sorted(parts['DSEL'].fyear.unique().tolist())} DSEL only; disjoint from pool training and calibration",
                 "training_row_ids_sha256": {name: hashlib.sha256(parts[name].index.to_numpy().tobytes()).hexdigest()
                                            for name in ("Old_fit", "New_fit")},
                 "duration_seconds": perf_counter() - begin}
        atomic_json(directory / "audit.json", audit)
        atomic_json(directory / "complete.json", {"status": "completed", "at": now(),
                                                  "metrics_sha256": sha256(directory / "metrics.csv"),
                                                  "predictions_sha256": sha256(directory / "predictions.npz")})
    return metrics


def summarize(run, config):
    frames = []
    for seed in config["seeds"]:
        for method, ratio in config["fs_configs"]:
            tag = "no_fs" if method == "none" else f"{method}_r{round(ratio * 100)}"
            directory = run / f"seed_{seed}" / tag
            complete = json.loads((directory / "complete.json").read_text())
            if complete["metrics_sha256"] != sha256(directory / "metrics.csv"):
                raise ValueError("Completed metrics checksum changed")
            frames.append(pd.read_csv(directory / "metrics.csv"))
    raw = pd.concat(frames, ignore_index=True)
    if raw.duplicated(["seed", "fs", "method", "period"]).any():
        raise ValueError("Duplicate method-period rows")
    period_count = 1 + len(config.get("test_years", [2015, 2016, 2017, 2018]))
    if len(raw) != len(config["seeds"]) * len(config["fs_configs"]) * 63 * period_count:
        raise ValueError("Incomplete experiment matrix")
    atomic_csv(run / "metrics.csv", raw)
    summary = raw.groupby(["fs", "n_features", "method", "period"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    atomic_csv(run / "summary.csv", summary.reset_index())
    pooled = raw.loc[raw.period.astype(str).eq("pooled")]
    controls = pooled.loc[pooled.fs.eq("no_fs"), ["seed", "method", *METRICS]]
    paired = pooled.loc[~pooled.fs.eq("no_fs")].merge(controls, on=["seed", "method"],
                                                      suffixes=("", "_no_fs"), validate="many_to_one")
    for metric in METRICS:
        paired[f"{metric}_delta"] = paired[metric] - paired[f"{metric}_no_fs"]
    # Re-selecting the static configuration changes the policy as well as FS.
    atomic_csv(run / "fs_selected_policy_differences.csv", paired.loc[
        paired.method.eq("ValidationSelectedStatic")])
    atomic_csv(run / "fs_paired_differences.csv", paired.loc[
        ~paired.method.eq("ValidationSelectedStatic")])
    return raw


def main(argv=None, *, default_fs_methods=()):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--fs-methods", nargs="*", choices=FS_METHODS, default=list(default_fs_methods))
    parser.add_argument("--fs-ratios", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(3, max(1, (os.cpu_count() or 4) // 4)))
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.workers < 1 or args.k < 1 or any(not 0 < r < 1 for r in args.fs_ratios):
        parser.error("positive threads/k and ratios strictly between 0 and 1 required")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.fs_methods)) != len(args.fs_methods):
        parser.error("Duplicate seeds or feature-selection methods")
    fs_configs = [("none", 1.0)] + list(product(args.fs_methods, args.fs_ratios))
    tags = ["no_fs" if m == "none" else f"{m}_r{round(r * 100)}" for m, r in fs_configs]
    if len(set(tags)) != len(tags):
        parser.error("Feature-selection ratios produce duplicate run identifiers")
    config = {"seeds": args.seeds, "fs_configs": fs_configs, "threads": args.threads,
              "workers": args.workers, "k": args.k}
    sources = [Path(__file__), ROOT / "src/ensemble/standard_dynamic.py",
               ROOT / "experiments/_shared/common_artifacts.py",
               ROOT / "experiments/_shared/common_modeling.py",
               ROOT / "src/data/sampler.py", ROOT / "src/models/xgboost_wrapper.py",
               ROOT / "src/evaluation/metrics.py", ROOT / "config/model_config.yaml",
               ROOT / "config/sampling_config.yaml"]
    provenance = {"raw_sha256": sha256(US_CSV), "python": platform.python_version(),
                  "versions": {p: version(p) for p in ("deslib", "numpy", "pandas", "scikit-learn", "xgboost",
                                                       "imbalanced-learn", "lightgbm")},
                  "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources}}
    data, features = load_data()
    parts = partition_data(data)
    if args.resume:
        run = args.resume.resolve()
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["provenance"] != provenance:
            raise ValueError("Resume refused: source/data/environment changed")
        config = manifest["config"]
    else:
        run = args.output_root.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run.mkdir(parents=True, exist_ok=False)
        manifest = {"protocol": PROTOCOL, "status": "created", "created_at": now(),
                    "config": config, "provenance": provenance, "command": sys.argv,
                    "features": features, "slots": SLOTS, "samplers": SAMPLER_NAMES,
                    "target": "original company final status: failed=1; not verified next-year event",
                    "nominal_split": "Old 1999-2011; New 2012-2014; Test 2015-2018",
                    "actual_fit": "Old 1999-2011; New 2012-2013; 2014 reserved for validation",
                    "validation": "2014, fixed seed42 stratified 50/50 disjoint DSEL/Calibration; not CV",
                    "representation": "impute, scale and supervised FS on Old_fit only, before sampling",
                    "selection": "static AP ranking and F1 thresholds on Calibration only; Test evaluation only",
                    "limitations": "retrospective final-company-status target; repeated companies across years; within-2014 partition has no finer timestamps; held-out validation reduces New fitting data; seed SD is not independent-data uncertainty; no claim of superiority or statistical significance"}
        atomic_json(run / "manifest.json", manifest)
        assignments = pd.concat([frame[["source_row_id", "company_name", "fyear", "target"]].assign(partition=name)
                                 for name, frame in parts.items()])
        atomic_csv(run / "partitions.csv", assignments)
        atomic_csv(run / "partition_summary.csv", assignments.groupby("partition", as_index=False).agg(
            n_rows=("target", "size"), n_positive=("target", "sum"),
            first_year=("fyear", "min"), last_year=("fyear", "max")))
    print(f"RUN {run}", flush=True)
    try:
        manifest.update(status="running", last_started_at=now(), pid=os.getpid())
        atomic_json(run / "manifest.json", manifest)
        jobs = []
        for seed in config["seeds"]:
            for method, ratio in config["fs_configs"]:
                tag = "no_fs" if method == "none" else f"{method}_r{round(ratio * 100)}"
                directory = run / f"seed_{seed}" / tag
                if (directory / "complete.json").exists():
                    complete = json.loads((directory / "complete.json").read_text())
                    if (complete["metrics_sha256"] != sha256(directory / "metrics.csv") or
                            complete["predictions_sha256"] != sha256(directory / "predictions.npz")):
                        raise ValueError("Completed output checksum changed")
                    print(f"REUSE seed={seed} fs={tag}", flush=True)
                    continue
                jobs.append((seed, method, ratio, tag, directory))
        with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
            futures = {executor.submit(run_configuration, parts, features, seed, method, ratio,
                                       config["threads"], config["k"], directory): (seed, tag)
                       for seed, method, ratio, tag, directory in jobs}
            for count, future in enumerate(as_completed(futures), 1):
                seed, tag = futures[future]
                metrics = future.result()
                atomic_json(run / "progress.json", {"seed": seed, "fs": tag, "at": now(),
                                                      "completed_new_jobs": count, "pending_jobs": len(jobs) - count})
                print(f"DONE {count}/{len(jobs)} seed={seed} fs={tag} rows={len(metrics)}", flush=True)
        raw = summarize(run, config)
        manifest.update(status="completed", completed_at=now(), n_metric_rows=len(raw))
        atomic_json(run / "manifest.json", manifest)
        atomic_json(run / "progress.json", {"stage": "completed", "at": now()})
        print(f"COMPLETED {run} rows={len(raw)}", flush=True)
    except Exception:
        manifest.update(status="failed", failed_at=now(), error=traceback.format_exc())
        atomic_json(run / "manifest.json", manifest)
        raise
    return run


if __name__ == "__main__":
    main()
