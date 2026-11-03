"""Phase 1 baseline classifiers on the fixed-year event-target protocol.

Same rows, event labels, Old_fit-only imputation/scaling and samplers as the
event-target Study I/II run. Each classifier x training period (Old, New,
Retrain=Old+New) x sampler is tuned with the same candidate budget on the whole
2014 validation year (AUC; candidate 0 is the default used by the ensemble
factory), then refit with the ten study seeds. The F1 threshold also comes from
2014. Test 2015-2018 is evaluated only after every choice is frozen.
Sampling is the only class-imbalance intervention (no class weights).
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments._shared.baseline_val_search import (
    LGB_GRID,
    LR_GRID,
    RF_GRID,
    TABM_GRID,
    XGB_GRID,
    _random_candidates_from_grid,
)
from experiments._shared.common_artifacts import atomic_csv, atomic_json, sha256
from experiments._shared.common_bankruptcy import US_CSV
from experiments._shared.common_modeling import SAMPLER_NAMES, SEEDS, STRATEGIES, make_model
from experiments.phase2_ensemble.dynamic.des import bankruptcy_standard_dynamic as study
from experiments.phase2_ensemble.dynamic.des import bankruptcy_standard_dynamic_event as event
from src.data import ImbalanceSampler
from src.evaluation import compute_metrics

PROTOCOL = "fixed_bankruptcy_event_target_baselines_v1"
OUTPUT = ROOT / "results/phase1_baseline/fixed_event/runs"
CLASSIFIERS = ("xgb", "lgb", "rf", "lr", "tabm")
GRIDS = {"xgb": XGB_GRID, "lgb": LGB_GRID, "rf": RF_GRID, "lr": LR_GRID, "tabm": TABM_GRID}
GPU_CLASSIFIERS = ("tabm",)
PERIODS = ("Old", "New", "Retrain")
SAMPLINGS = ("none", "under", "over", "hybrid")
# Pool slots of the ensemble: the classifier choice it needs evidence for.
POOL = tuple(product(("Old", "New"), ("under", "over", "hybrid")))
TABM_FIXED = {"pos_weight": None, "batch_size": 1024, "max_epochs": 100, "patience": 10}


def candidates(classifier, budget):
    """Candidate 0 is the factory default; the rest are fixed random grid draws."""
    rng = np.random.default_rng(zlib.crc32(classifier.encode()))
    draws = _random_candidates_from_grid(GRIDS[classifier], budget - 1, rng, guard_factor=120)
    out = [{}]
    for cand in draws:
        cand = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in cand.items()}
        # rng.choice turns a mixed list such as RF max_features into strings ("0.35").
        cand = {k: float(v) if isinstance(v, str) and v.replace(".", "", 1).isdigit() else v
                for k, v in cand.items()}
        if classifier == "lgb" and cand["subsample"] < 1:
            cand["bagging_freq"] = 1  # LightGBM ignores subsample without it
        if classifier == "tabm" and not cand["use_num_embeddings"]:
            cand.pop("d_embedding")
        out.append(cand)
    if len(out) != budget:
        raise ValueError(f"{classifier}: grid too small for budget {budget}")
    return out


def build(classifier, params, seed, threads):
    if classifier == "tabm":
        from src.models import TabMWrapper
        return TabMWrapper(seed=seed, **{**TABM_FIXED, **params})
    model = make_model(classifier, seed, threads)
    if params:
        model.params.update(params)
        if classifier == "lgb" and "n_estimators" in params:
            model._num_boost_round = int(model.params.pop("n_estimators"))
        if classifier in ("rf", "lr"):
            model.model.set_params(**params)
    return model


def fit_predict(classifier, params, seed, threads, X, y, early_X, early_y, outputs):
    model = build(classifier, params, seed, threads)
    if classifier == "tabm":
        model.fit(X, y, early_X, early_y)
    else:
        model.fit(X, y)
    result = []
    for frame in outputs:
        p = np.asarray(model.predict_proba(frame), dtype=float)
        p = p[:, 1] if p.ndim == 2 else p
        if not (np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))):
            raise ValueError(f"{classifier}: invalid probabilities")
        result.append(p)
    return result


def training_data(parts, transformed, period, sampling, seed, classifier):
    names = {"Old": ["Old_fit"], "New": ["New_fit"], "Retrain": ["Old_fit", "New_fit"]}[period]
    X = pd.concat([transformed[n] for n in names])
    y = np.concatenate([parts[n].target.to_numpy() for n in names])
    early_X = early_y = None
    if classifier == "tabm":
        # Early stopping uses real training-period rows, never 2014 or synthetic rows.
        X, early_X, y, early_y = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    X, y = ImbalanceSampler(random_state=seed).apply_sampling(X, y, STRATEGIES[sampling])
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Sampling removed a class")
    return X, y, early_X, early_y, len(names)


def run_job(parts, transformed, classifier, period, sampling, budget, seeds, threads, directory):
    begin = perf_counter()
    val_y, test_y = parts["Validation"].target.to_numpy(), parts["Test"].target.to_numpy()
    with threadpool_limits(threads):
        tuning = []
        X, y, early_X, early_y, _ = training_data(parts, transformed, period, sampling, seeds[0], classifier)
        for index, params in enumerate(candidates(classifier, budget)):
            (p,) = fit_predict(classifier, params, seeds[0], threads, X, y, early_X, early_y,
                               [transformed["Validation"]])
            tuning.append({"candidate": index, "params": json.dumps(params, sort_keys=True),
                           "validation_AUC": roc_auc_score(val_y, p),
                           "validation_PR_AUC": average_precision_score(val_y, p)})
        tuning = pd.DataFrame(tuning)
        chosen = int(tuning.sort_values(["validation_AUC", "candidate"], ascending=[False, True]).candidate.iloc[0])
        params = candidates(classifier, budget)[chosen]
        rows, cache = [], {}
        for seed in seeds:
            X, y, early_X, early_y, _ = training_data(parts, transformed, period, sampling, seed, classifier)
            val_p, test_p = fit_predict(classifier, params, seed, threads, X, y, early_X, early_y,
                                        [transformed["Validation"], transformed["Test"]])
            threshold = study.select_threshold(val_y, val_p)
            common = {"classifier": classifier, "period": period, "sampling": sampling, "seed": seed,
                      "chosen_candidate": chosen, "threshold": threshold, "n_train_after_sampling": len(y),
                      "validation_AUC": roc_auc_score(val_y, val_p),
                      "validation_PR_AUC": average_precision_score(val_y, val_p)}
            test = parts["Test"]
            for year, mask in [("pooled", np.ones(len(test), dtype=bool))] + [
                    (str(t), test.fyear.to_numpy() == t) for t in sorted(test.fyear.unique())]:
                metrics = compute_metrics(test_y[mask], test_p[mask], threshold=threshold)
                metrics["Brier"] = float(brier_score_loss(test_y[mask], test_p[mask]))
                rows.append({**common, "test_period": year, "n_test": int(mask.sum()),
                             "n_test_positive": int(test_y[mask].sum()), **metrics})
            cache[f"seed_{seed}_validation"] = val_p
            cache[f"seed_{seed}_test"] = test_p
    directory.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    atomic_csv(directory / "tuning.csv", tuning)
    atomic_csv(directory / "metrics.csv", metrics)
    np.savez_compressed(directory / "predictions.npz", validation_row_ids=parts["Validation"].index.to_numpy(),
                        test_row_ids=parts["Test"].index.to_numpy(), **cache)
    atomic_json(directory / "complete.json", {
        "status": "completed", "at": study.now(), "chosen_params": params, "chosen_candidate": chosen,
        "default_validation_AUC": float(tuning.validation_AUC.iloc[0]),
        "chosen_tuning_validation_AUC": float(tuning.validation_AUC.iloc[chosen]),
        "test_used_for_selection": False, "duration_seconds": perf_counter() - begin,
        "metrics_sha256": sha256(directory / "metrics.csv"),
        "predictions_sha256": sha256(directory / "predictions.npz")})
    return metrics


def summarize(run, jobs):
    raw = pd.concat([pd.read_csv(d / "metrics.csv") for *_, d in jobs], ignore_index=True)
    atomic_csv(run / "metrics.csv", raw)
    atomic_csv(run / "tuning.csv", pd.concat([
        pd.read_csv(d / "tuning.csv").assign(classifier=c, period=p, sampling=s) for c, p, s, d in jobs]))
    pooled = raw[raw.test_period.astype(str).eq("pooled")]
    values = ["validation_AUC", "validation_PR_AUC", "AUC", "PR_AUC", "F1", "Recall", "Precision", "G_Mean"]
    summary = pooled.groupby(["classifier", "period", "sampling"])[values].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    atomic_csv(run / "summary.csv", summary.reset_index())
    # Classifier choice uses validation only; Test columns are reported beside it, not used.
    pool = pooled[[ (p, s) in POOL for p, s in zip(pooled.period, pooled.sampling)]]
    ranking = (pool.groupby("classifier")[values].mean()
               .sort_values("validation_AUC", ascending=False).reset_index())
    ranking.insert(1, "validation_rank", np.arange(1, len(ranking) + 1))
    atomic_csv(run / "classifier_ranking_pool_slots.csv", ranking)
    every = (pooled.groupby("classifier")[values].mean()
             .sort_values("validation_AUC", ascending=False).reset_index())
    atomic_csv(run / "classifier_ranking_all_settings.csv", every)
    return raw, ranking


def make_parser(description, output):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=list(CLASSIFIERS))
    parser.add_argument("--periods", nargs="+", choices=PERIODS, default=list(PERIODS))
    parser.add_argument("--samplings", nargs="+", choices=SAMPLINGS, default=list(SAMPLINGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--budget", type=int, default=24, help="tuning candidates per setting, incl. default")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=output)
    parser.add_argument("--resume", type=Path)
    return parser


def check_args(parser, args):
    if args.budget < 1 or args.threads < 1 or args.workers < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive budget/threads/workers and unique seeds required")
    return {"classifiers": args.classifiers, "periods": args.periods, "samplings": args.samplings,
            "seeds": args.seeds, "budget": args.budget, "threads": args.threads, "workers": args.workers,
            "tabm_fixed": TABM_FIXED}


def provenance(extra_sources, inputs):
    """Shared model/sampler/metric sources plus the dataset-specific inputs."""
    sources = [Path(__file__), Path(study.__file__), *extra_sources,
               ROOT / "experiments/_shared/baseline_val_search.py",
               ROOT / "experiments/_shared/common_modeling.py", ROOT / "src/data/sampler.py",
               ROOT / "src/models/xgboost_wrapper.py", ROOT / "src/models/lightgbm_wrapper.py",
               ROOT / "src/models/random_forest_wrapper.py", ROOT / "src/models/logistic_regression_wrapper.py",
               ROOT / "src/models/tabm_wrapper.py", ROOT / "src/evaluation/metrics.py",
               ROOT / "config/model_config.yaml", ROOT / "config/sampling_config.yaml"]
    return {**inputs, "python": platform.python_version(),
            "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm",
                                                 "imbalanced-learn", "torch", "tabm")},
            "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources}}


def main(argv=None):
    parser = make_parser(__doc__, OUTPUT)
    args = parser.parse_args(argv)
    config = check_args(parser, args)
    fingerprint = provenance([Path(event.__file__)], {"raw_sha256": sha256(US_CSV),
                                                       "labels_sha256": sha256(event.LABELS)})
    data, features, _ = event.load_data()
    parts = study.partition_data(data)
    event.check_label_maturity(parts)
    parts["Validation"] = pd.concat([parts.pop("DSEL"), parts.pop("Calibration")]).sort_index()
    return execute(args, config, fingerprint, parts, features, {
        "protocol": PROTOCOL, "features": features,
        "target": "reconstructed next-year event (same ledger as the event-target Study I/II run)",
        "training_periods": {"Old": "1999-2011", "New": "2012-2013", "Retrain": "1999-2013"},
        "validation": "all of 2014 (event-run DSEL+Calibration); hyperparameters by AUC, threshold by F1",
        "classifier_choice": "rank by mean 2014 validation AUC over the six ensemble pool slots (Old/New x under/over/hybrid)",
        "limitations": "2014 validation has 33 events; tuning noise is shared by all classifiers; seed SD is not independent-data uncertainty"})


def execute(args, config, fingerprint, parts, features, details):
    """Tune, refit and summarize every job on prepared Old_fit/New_fit/Validation/Test parts."""
    transformed, representation = study.fit_representation(parts, features, "none", 1.0, args.seeds[0], args.threads)
    if args.resume:
        run = args.resume.resolve()
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["provenance"] != fingerprint or manifest["config"] != config:
            raise ValueError("Resume refused: source/data/environment/settings changed")
    else:
        run = args.output_root.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run.mkdir(parents=True, exist_ok=False)
        manifest = {**details, "status": "created", "created_at": study.now(), "config": config,
                    "provenance": fingerprint, "command": sys.argv,
                    "representation": "impute and scale on Old_fit only, no FS, before sampling",
                    "samplers": SAMPLER_NAMES, "imbalance_intervention": "sampling only; no class weights",
                    "tuning": "same budget per classifier x period x sampler; candidate 0 = ensemble factory default; tuned once with the first seed",
                    "tabm_early_stopping": "10% stratified holdout of the real training-period rows, before sampling",
                    "excluded": "SVM, sklearn MLP and other deep tabular models: not part of the thesis baseline set"}
        atomic_json(run / "manifest.json", manifest)
        atomic_csv(run / "partition_summary.csv", pd.DataFrame([
            {"partition": n, "n_rows": len(f), "n_positive": int(f.target.sum()),
             "first_year": int(f.fyear.min()), "last_year": int(f.fyear.max())} for n, f in parts.items()]))
    print(f"RUN {run}", flush=True)
    jobs = [(c, p, s, run / c / f"{p}_{s}") for c in args.classifiers for p in args.periods for s in args.samplings]
    try:
        manifest.update(status="running", last_started_at=study.now(), pid=os.getpid())
        atomic_json(run / "manifest.json", manifest)
        pending = []
        for c, p, s, d in jobs:
            if (d / "complete.json").exists():
                done = json.loads((d / "complete.json").read_text())
                if done["metrics_sha256"] != sha256(d / "metrics.csv"):
                    raise ValueError("Completed output hash changed")
                print(f"REUSE {c} {p} {s}", flush=True)
            else:
                pending.append((c, p, s, d))
        # CPU classifiers share a process pool; GPU jobs run one at a time beside it.
        with ProcessPoolExecutor(max_workers=args.workers) as cpu, ProcessPoolExecutor(max_workers=1) as gpu:
            futures = {(gpu if c in GPU_CLASSIFIERS else cpu).submit(
                run_job, parts, transformed, c, p, s, args.budget, args.seeds, args.threads, d): (c, p, s)
                for c, p, s, d in pending}
            for count, future in enumerate(as_completed(futures), 1):
                c, p, s = futures[future]
                future.result()
                atomic_json(run / "progress.json", {"job": [c, p, s], "at": study.now(),
                                                     "completed_new_jobs": count, "pending_jobs": len(pending) - count})
                print(f"DONE {count}/{len(pending)} {c} {p} {s}", flush=True)
        raw, ranking = summarize(run, jobs)
        manifest.update(status="completed", completed_at=study.now(), n_metric_rows=len(raw),
                        representation={k: v for k, v in representation.items() if k.endswith("scope")})
        atomic_json(run / "manifest.json", manifest)
        print(ranking[["classifier", "validation_rank", "validation_AUC", "AUC"]].to_string(index=False), flush=True)
        print(f"COMPLETED {run} rows={len(raw)}", flush=True)
    except Exception:
        manifest.update(status="failed", failed_at=study.now(), error=traceback.format_exc())
        atomic_json(run / "manifest.json", manifest)
        raise
    return run


if __name__ == "__main__":
    main()
