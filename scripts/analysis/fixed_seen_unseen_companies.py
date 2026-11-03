"""Split fixed-year Test results by whether a company appeared in model fitting.

The raw status label is fixed per company, so most failed Test companies also
contribute failed rows to Old_fit/New_fit. This replay measures how much of the
saved Test performance comes from such already-seen companies.

No fitting, threshold change, or test-based model choice. Thresholds and the
validation-selected static subset are read from each saved selection.json.
Company groups are defined only by partition membership, never by predictions.
Bootstrap intervals are descriptive: companies are resampled within each
(group, label) stratum, all rows of a sampled company are kept, and the
estimand is the mean of ten seed-specific metrics. No hypothesis tests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import set_num_threads
from sklearn.metrics import f1_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "fixed_company_statistics", Path(__file__).with_name("fixed_company_statistics.py"))
fcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fcs)

RUN = fcs.RUN
METRICS = fcs.METRICS
METHODS = fcs.METHODS
GROUPS = ("seen", "unseen", "seen_new_fit", "seen_old_fit_only")
CONTRASTS = (("New3", "Old3"), ("ValidationSelectedStatic", "Equal6"), ("Equal6", "New3"))


def company_groups(partitions):
    """Assign each Test company to a group from training-partition membership only."""
    test = partitions.query("partition == 'Test'")
    old = set(partitions.query("partition == 'Old_fit'").company_name)
    new = set(partitions.query("partition == 'New_fit'").company_name)
    in_2014 = set(partitions.query("partition in ('DSEL', 'Calibration')").company_name)
    fit = partitions[partitions.partition.isin(["Old_fit", "New_fit"])]
    last_fit = fit.groupby("company_name").fyear.max()
    companies = test.groupby("company_name").agg(target=("target", "max"), n_test_rows=("target", "size"),
                                                 first_test_year=("fyear", "min"))
    companies["in_old_fit"] = companies.index.isin(old)
    companies["in_new_fit"] = companies.index.isin(new)
    companies["in_2014_validation"] = companies.index.isin(in_2014)
    companies["last_fit_year"] = last_fit.reindex(companies.index)
    companies["group"] = np.select(
        [companies.in_new_fit, companies.in_old_fit], ["seen_new_fit", "seen_old_fit_only"], "unseen")
    return companies


def stratified_weights(strata, replicates, rng):
    """Resample companies with replacement inside each stratum; one row per replicate."""
    weights = np.zeros((replicates, len(strata)), dtype=np.int16)
    for code in np.unique(strata):
        members = np.flatnonzero(strata == code)
        for b in range(replicates):
            weights[b, members] += np.bincount(rng.integers(0, len(members), len(members)),
                                               minlength=len(members)).astype(np.int16)
    return weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=RUN, help="completed fixed-year run, relative to the project root")
    ap.add_argument("--replicates", type=int, default=4999)
    ap.add_argument("--bootstrap-seed", type=int, default=20261030)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    set_num_threads(args.threads)
    run = ROOT / args.run
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["protocol"] in fcs.PROTOCOLS
    seeds = manifest["config"]["seeds"]
    source_hashes = {}

    def remember(path):
        source_hashes[path.relative_to(ROOT).as_posix()] = fcs.sha(path)

    for path in (run / "manifest.json", run / "partitions.csv", run / "metrics.csv",
                 Path(__file__).resolve(), Path(fcs.__file__).resolve()):
        remember(path)
    partitions = pd.read_csv(run / "partitions.csv")
    test = partitions.query("partition == 'Test'")
    # A company is positive if any Test row is; final-status labels are constant, events occur once.
    labels = test.groupby("company_name").target
    assert test.source_row_id.is_unique and (labels.sum().max() == 1 if manifest["protocol"].endswith("_event_target_v1")
                                             else labels.nunique().max() == 1)
    companies = company_groups(partitions)
    pooled = pd.read_csv(run / "metrics.csv").query("period == 'pooled'")
    fs_tags = list(pooled.fs.unique())
    output = args.output or ROOT / "results/statistical_tests" / ("seen_unseen_companies_event" if manifest["protocol"].endswith("_event_target_v1") else "seen_unseen_companies") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output.resolve()
    assert output.is_relative_to(ROOT / "results/statistical_tests")
    output.mkdir(parents=True, exist_ok=False)
    specification = dict(run=args.run, protocol=manifest["protocol"], seeds=seeds, fs=fs_tags, methods=METHODS,
                         metrics=METRICS, groups=GROUPS, contrasts=CONTRASTS,
                         group_rule="seen = company has any Old_fit (1999-2011) or New_fit (2012-2013) row; "
                                    "seen_new_fit = has a New_fit row; seen_old_fit_only = Old_fit rows only; "
                                    "2014 DSEL/Calibration membership is recorded but does not define groups",
                         replicates=args.replicates, bootstrap_seed=args.bootstrap_seed,
                         resampling="companies with replacement within (group, label) strata; all rows retained",
                         estimand="mean of ten seed-specific metrics within a company group",
                         intervals="pointwise percentile 95%; descriptive, no hypothesis tests or multiplicity claims",
                         scope="retrospective exploratory replay of saved predictions; no refit, no test-based selection")
    (output / "specification.json").write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Analysis fixed before calculation: {output}", flush=True)

    companies.reset_index().to_csv(output / "company_groups.csv", index=False)
    row_group = test.company_name.map(companies.group).to_numpy()
    masks = {"seen": row_group != "unseen", "unseen": row_group == "unseen",
             "seen_new_fit": row_group == "seen_new_fit", "seen_old_fit_only": row_group == "seen_old_fit_only"}
    cluster, names = pd.factorize(test.company_name, sort=True)
    y = test.target.to_numpy()
    probabilities, keys, max_error = [], [], 0.0
    for fs in fs_tags:
        for seed in seeds:
            directory = run / f"seed_{seed}" / fs
            complete = json.loads((directory / "complete.json").read_text())
            assert complete["status"] == "completed"
            assert fcs.sha(directory / "predictions.npz") == complete["predictions_sha256"]
            for name in ("complete.json", "predictions.npz", "selection.json"):
                remember(directory / name)
            selected = json.loads((directory / "selection.json").read_text())
            assert selected["test_used_for_selection"] is False
            with np.load(directory / "predictions.npz", allow_pickle=False) as z:
                assert np.array_equal(z["Test_row_ids"], test.source_row_id.to_numpy())
                assert np.array_equal(z["Test_y"], y)
                for method in METHODS:
                    row = pooled[pooled.fs.eq(fs) & pooled.seed.eq(seed) & pooled.method.eq(method)]
                    assert len(row) == 1
                    threshold = selected["thresholds"][method]
                    assert abs(row.threshold.iloc[0] - threshold) < 1e-14
                    p = fcs.probability_for(z, method, selected["selected_static"])
                    scores = np.array([roc_auc_score(y, p), f1_score(y, p >= threshold), recall_score(y, p >= threshold)])
                    error = np.max(np.abs(scores - row[list(METRICS)].to_numpy(dtype=float)[0]))
                    max_error = max(max_error, error)
                    assert error < 1e-12
                    probabilities.append((p, threshold))
                    keys.append((fs, method, seed))
        print(f"Replay passed: {fs}", flush=True)
    key_frame = pd.DataFrame(keys, columns=["fs", "method", "seed"])

    # Strata codes over companies in `names` order; the same draws serve every predictor.
    company_group = companies.group.reindex(names).to_numpy()
    strata = pd.factorize(pd.Series(company_group) + "_" + companies.target.reindex(names).astype(str).to_numpy())[0]
    rng = np.random.default_rng(args.bootstrap_seed)
    weights = stratified_weights(strata, args.replicates, rng)
    point, draws, descriptive, counts = {}, {}, [], []
    for group in GROUPS:
        mask = masks[group]
        group_companies = np.unique(cluster[mask])
        group_cluster = np.searchsorted(group_companies, cluster[mask])
        counts.append(dict(group=group, n_rows=int(mask.sum()), n_positive_rows=int(y[mask].sum()),
                           n_companies=len(group_companies),
                           n_positive_companies=int(companies.target.reindex(names[group_companies]).sum())))
        inputs = [fcs.sort_inputs(y[mask], p[mask], group_cluster, t) for p, t in probabilities]
        arrays = [np.stack([item[k] for item in inputs]) for k in range(6)]
        boot = fcs.weighted_scores(weights[:, group_companies], *arrays)
        assert np.isfinite(boot).all(), "Degenerate class bootstrap; do not silently discard replicates"
        unit = fcs.weighted_scores(np.ones((1, len(group_companies)), dtype=np.int16), *arrays)[:, 0]
        direct = np.array([[roc_auc_score(y[mask], p[mask]), f1_score(y[mask], p[mask] >= t),
                            recall_score(y[mask], p[mask] >= t)] for p, t in probabilities])
        assert np.max(np.abs(unit - direct)) < 1e-12
        for (fs, method), index in key_frame.groupby(["fs", "method"], sort=False).groups.items():
            index = np.asarray(index)
            assert len(index) == len(seeds)
            point[group, fs, method] = direct[index].mean(axis=0)
            draws[group, fs, method] = boot[index].mean(axis=0)
            for k, metric in enumerate(METRICS):
                lo, hi = np.quantile(draws[group, fs, method][:, k], [0.025, 0.975])
                descriptive.append(dict(group=group, fs=fs, method=method, metric=metric,
                                        mean=point[group, fs, method][k], seed_sd=direct[index, k].std(ddof=1),
                                        ci_low=lo, ci_high=hi))
        print(f"Bootstrap done: {group}", flush=True)
    pd.DataFrame(counts).to_csv(output / "group_counts.csv", index=False)
    pd.DataFrame(descriptive).to_csv(output / "descriptive.csv", index=False)

    rows = []
    for fs in fs_tags:
        for method in METHODS:
            # Disjoint groups use independent strata, so draw-wise differences are valid.
            rows += [dict(contrast="seen_minus_unseen", fs=fs, method_a=method, method_b=method,
                          group_a="seen", group_b="unseen")]
        for method_a, method_b in CONTRASTS:
            rows += [dict(contrast=f"{method_a}_minus_{method_b}", fs=fs, method_a=method_a, method_b=method_b,
                          group_a=group, group_b=group) for group in GROUPS]
    contrasts = []
    for row in rows:
        a = (row["group_a"], row["fs"], row["method_a"])
        b = (row["group_b"], row["fs"], row["method_b"])
        delta, distribution = point[a] - point[b], draws[a] - draws[b]
        for k, metric in enumerate(METRICS):
            lo, hi = np.quantile(distribution[:, k], [0.025, 0.975])
            contrasts.append({**row, "metric": metric, "mean_a": point[a][k], "mean_b": point[b][k],
                              "difference": delta[k], "ci_low": lo, "ci_high": hi})
    pd.DataFrame(contrasts).to_csv(output / "contrasts.csv", index=False)

    seen_positive = companies[(companies.group != "unseen") & companies.target.eq(1)]
    gap = (seen_positive.first_test_year - seen_positive.last_fit_year).value_counts().sort_index()
    validation = dict(status="passed", n_test_rows=len(test), n_test_companies=len(names),
                      n_predictors_replayed=len(keys), max_metric_replay_error=max_error,
                      group_counts=counts,
                      seen_positive_gap_years={int(k): int(v) for k, v in gap.items()},
                      seen_positive_in_2014_validation=int(seen_positive.in_2014_validation.sum()),
                      weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())
    (output / "validation.json").write_text(json.dumps(validation, indent=2, default=int), encoding="utf-8")
    state = dict(status="completed", specification=specification, source_sha256=source_hashes,
                 output_sha256={p.name: fcs.sha(p) for p in output.iterdir() if p.is_file()},
                 created_at=datetime.now(timezone.utc).isoformat(), versions={})
    from importlib.metadata import version
    state["versions"] = {m: version(m) for m in ("numpy", "pandas", "scikit-learn", "numba")}
    (output / "manifest.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2, default=int), flush=True)


if __name__ == "__main__":
    main()
