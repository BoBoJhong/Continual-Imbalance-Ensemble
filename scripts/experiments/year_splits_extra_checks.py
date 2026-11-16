"""Two extra checks for the 15 year-split bankruptcy ensemble (spec section 10, open items 1 and 2).

S1 exclude 2018: Test restricted to the 2015-2017 feature years (2018 events are only inferred
   from the end of the data). Members, thresholds and the V-static choice are unchanged: they
   are all chosen on validation data, so only the Test rows that are scored change. This is
   the "S1_exclude_2018" check of config/dynamic_selection_correction_v3.json.
S3 before correction: the same members combined with the DESlib-default protocol of the
   fixed-split base run: raw member scores (no Platt), member labels by argmax (0.5), DCS
   'best' tie rule, DESlib KNORA-U weighting; static rules average raw scores. This is the
   "S3_original_reference" of the same config.

Only seed-42 predictions of the thesis XGBoost run were stored, so the six-member pool is
refitted with that run's tuning choices, seeds, partitions and the unchanged runner code of the
experiment checkout (feature selection is not repeated). Before anything is reported, the
refitted baseline and corrected-ensemble metrics on the full 2015-2018 Test are compared with
the thesis run (verification.json).

Run with the experiment checkout as the working directory:
  python year_splits_extra_checks.py --xgb-run results/phase2_ensemble/year_splits_author2022_event/runs/<run> \
      --out <folder> [--splits ...] [--seeds ...] [--workers 5 --threads 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

CHECKOUT = Path.cwd().resolve()
sys.path.insert(0, str(CHECKOUT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from experiments.phase2_ensemble.dynamic.des import bankruptcy_year_splits_event as runner  # noqa: E402
from src.ensemble.standard_dynamic import METHODS, fit_selector  # noqa: E402

study, replay = runner.study, runner.replay
LAST_KEPT_YEAR = 2017
KINDS = ("baseline_full", "baseline_1517", "ensemble_full", "ensemble_1517", "ensemble_default")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_protocol(parts, arrays, raw, seed, k, threads):
    """Fixed-split base run: raw scores, argmax member labels, DESlib defaults."""
    probabilities = {name: tuple(raw[part][:, list(indices)].mean(axis=1) for part in ("Calibration", "Test"))
                     for name, indices in study.static_candidates().items()}
    tables, _ = replay.lookup_tables(arrays, raw)
    pool = [replay.LookupMember(table) for table in tables]
    dsel_y = parts["DSEL"].target.to_numpy()
    for method in METHODS:
        selector = fit_selector(method, pool, arrays["DSEL_X"], dsel_y, k=k, seed=seed, threads=threads)
        probabilities[method] = tuple(selector.predict_proba(arrays[f"{part}_X"])[:, 1]
                                      for part in ("Calibration", "Test"))
    return probabilities


def run_job(parts, features, label, seed, params, threads, k, directory):
    runner.quiet()
    retrain = label == runner.RETRAIN
    validation_y = np.concatenate([parts["DSEL"].target.to_numpy(), parts["Calibration"].target.to_numpy()])
    keep = parts["Test"].fyear.to_numpy() <= LAST_KEPT_YEAR
    parts_1517 = {**parts, "Test": parts["Test"].loc[keep]}
    out = {kind: [] for kind in KINDS}
    with threadpool_limits(threads):
        transformed, representation = study.fit_representation(parts, features, "none", 1.0, seed, threads)
        settings = ([("Retrain", s) for s in runner.SAMPLINGS] if retrain
                    else [(p, s) for p in ("Old", "New") for s in runner.SAMPLINGS])
        scores = {}
        for period, sampling in settings:
            X, y = runner.training_frame(parts, transformed, period)
            Xs, ys, _ = runner.sample_training(X, y, sampling, seed)
            scores[(period, sampling)] = runner.fit_scores(
                "xgb", params[f"{period}_{sampling}"], seed, threads, Xs, ys,
                [transformed[part] for part in runner.SCORED])
            dsel_p, calibration_p, test_p = scores[(period, sampling)]
            common = {"classifier": "xgb", "split": label, "seed": seed, "period": period, "sampling": sampling}
            validation_p = np.concatenate([dsel_p, calibration_p])
            out["baseline_full"] += runner.baseline_rows(parts, common, validation_y, validation_p, test_p)
            out["baseline_1517"] += runner.baseline_rows(parts_1517, common, validation_y, validation_p, test_p[keep])
        if not retrain:
            raw = {part: np.column_stack([scores[(side, s)][i] for side in ("Old", "New") for s in runner.POOL_SAMPLINGS])
                   for i, part in enumerate(runner.SCORED)}
            metadata = {"split": label, "seed": seed, "fs": "no_fs",
                        "n_features": len(representation["selected_features"])}
            metrics, selection, _, probabilities, _ = runner.ensemble_metrics(
                parts, transformed, raw, metadata, seed, k, threads)
            out["ensemble_full"].append(metrics)
            restricted = {name: (c, t[keep]) for name, (c, t) in probabilities.items()}
            metrics_1517, selection_1517 = study.evaluate_predictions(parts_1517, restricted, metadata)
            if selection_1517["thresholds"] != selection["thresholds"] or \
                    selection_1517["selected_static"] != selection["selected_static"]:
                raise ValueError("Calibration-only choices changed when Test was restricted")
            out["ensemble_1517"].append(metrics_1517)
            arrays = {f"{part}_X": np.asarray(transformed[part], dtype=np.float64) for part in runner.SCORED}
            default, _ = study.evaluate_predictions(
                parts, default_protocol(parts, arrays, raw, seed, k, threads), metadata)
            out["ensemble_default"].append(default)
    directory.mkdir(parents=True, exist_ok=True)
    for kind, rows in out.items():
        if rows:
            frame = pd.concat(rows, ignore_index=True) if isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)
            frame.to_csv(directory / f"{kind}.csv", index=False)
    (directory / "complete.json").write_text(json.dumps({"at": now()}), encoding="utf-8")
    return label, seed


def verify(out, run, labels, seeds):
    """Refitted 2015-2018 numbers against the thesis run."""
    keys = ["split", "seed", "period", "sampling", "test_period"]
    mine = pd.read_csv(out / "baseline_full.csv")
    theirs = pd.read_csv(run / "baseline_metrics.csv")
    theirs = theirs.loc[theirs.split.isin(labels) & theirs.seed.isin(seeds)]
    joined = mine.merge(theirs, on=keys, suffixes=("", "_thesis"))
    result = {"baseline_rows": [len(mine), len(theirs), len(joined)]}
    for column in ("AUC", "F1", "Recall", "threshold"):
        result[f"baseline_max_abs_diff_{column}"] = float((joined[column] - joined[f"{column}_thesis"]).abs().max())
    mine = pd.read_csv(out / "ensemble_full.csv")
    mine = mine.loc[mine.period.astype(str).eq("pooled")]
    theirs = pd.read_csv(run / "ensemble_metrics_pooled.csv")
    theirs = theirs.loc[theirs.fs.eq("no_fs") & theirs.split.isin(labels) & theirs.seed.isin(seeds)]
    joined = mine.merge(theirs, on=["split", "seed", "method"], suffixes=("", "_thesis"))
    result["ensemble_rows"] = [len(mine), len(theirs), len(joined)]
    for column in ("AUC", "F1", "Recall", "threshold"):
        result[f"ensemble_max_abs_diff_{column}"] = float((joined[column] - joined[f"{column}_thesis"]).abs().max())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xgb-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()

    run = args.xgb_run.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["classifier"] != "xgb" or manifest["status"] != "completed":
        raise ValueError("Need the completed thesis XGBoost run")
    k = manifest["config"]["k"]
    seeds = args.seeds or manifest["config"]["seeds"]
    known = dict(runner.SPLITS) | {runner.RETRAIN: None}
    labels = args.splits or list(known)
    sources = [Path(runner.__file__), Path(study.__file__), Path(replay.__file__), Path(runner.event.__file__),
               CHECKOUT / "src/ensemble/standard_dynamic.py", Path(__file__)]
    current = {p.resolve().relative_to(CHECKOUT).as_posix() if p.resolve().is_relative_to(CHECKOUT) else p.name:
               sha256(p) for p in sources}
    changed = [name for name, digest in current.items() if name in manifest["provenance"]["sources"]
               and manifest["provenance"]["sources"][name] != digest]
    if changed:
        raise ValueError(f"Runner code differs from the thesis run: {changed}")

    data, features, _ = runner.event.load_data()
    dsel, calibration = runner.validation_rows(data.loc[data.fyear.between(runner.FIRST_TRAIN, runner.LAST_TRAIN)])
    parts = {label: runner.partition(data, known[label], dsel, calibration) for label in labels}
    params = {label: {p.stem: json.loads(p.read_text(encoding="utf-8"))["chosen_params"]
                      for p in sorted((run / label / "tuning").glob("*.json"))} for label in labels}

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        "purpose": "S1 Test 2015-2017 and S3 DESlib-default reference for the 15 year-split XGBoost ensemble",
        "xgb_run": run.relative_to(CHECKOUT).as_posix(), "checkout": CHECKOUT.as_posix(), "splits": labels,
        "seeds": seeds, "k": k, "threads": args.threads, "feature_selection": "none (no_fs only)",
        "sources_sha256": current, "started_at": now(), "argv": sys.argv}, indent=2), encoding="utf-8")
    print(f"OUT {out}", flush=True)

    jobs = [(label, seed) for seed in seeds for label in labels
            if not (out / label / f"seed_{seed}" / "complete.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, parts[label], features, label, seed, params[label], args.threads, k,
                                   out / label / f"seed_{seed}") for label, seed in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            print(f"DONE {count}/{len(jobs)} {future.result()}", flush=True)

    for kind in KINDS:
        frames = [pd.read_csv(path) for label in labels for seed in seeds
                  if (path := out / label / f"seed_{seed}" / f"{kind}.csv").exists()]
        pd.concat(frames, ignore_index=True).to_csv(out / f"{kind}.csv", index=False)
    result = verify(out, run, labels, seeds)
    result["completed_at"] = now()
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("VERIFY", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
