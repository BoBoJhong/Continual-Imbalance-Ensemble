"""Replay fixed-year predictions and perform paired company cluster bootstrap.

No fitting, test-based model choice, or changes to the original experiment run.
Inference is conditional on the saved models, fixed split and ten listed seeds.
The analysis is explicitly retrospective/exploratory, not preregistered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from sklearn.metrics import f1_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RUN = "results/phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z"
METRICS = ("AUC", "F1", "Recall")
METHODS = ("Old3", "New3", "Equal6", "ValidationSelectedStatic", "DCS_OLA", "DCS_LCA", "KNORA_E", "KNORA_U")
FS_METHODS = METHODS[2:]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def sort_inputs(y, probability, cluster, threshold):
    """Negative score order and positive tie bounds permit exact weighted AUC."""
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    neg = neg[np.argsort(probability[neg], kind="stable")]
    left = np.searchsorted(probability[neg], probability[pos], side="left")
    right = np.searchsorted(probability[neg], probability[pos], side="right")
    return (cluster[neg], cluster[pos], left, right,
            (probability[neg] >= threshold).astype(np.int8),
            (probability[pos] >= threshold).astype(np.int8))


@njit(parallel=True)
def weighted_scores(weights, neg_cluster, pos_cluster, left, right, neg_pred, pos_pred):
    """Exact row-weighted ROC AUC, F1 and recall; each cluster keeps all rows."""
    models = neg_cluster.shape[0]
    out = np.empty((models, len(weights), 3))
    for m in prange(models):
        cumulative = np.empty(neg_cluster.shape[1] + 1)
        for b in range(len(weights)):
            cumulative[0] = 0.0
            fp = 0.0
            for j in range(neg_cluster.shape[1]):
                w = weights[b, neg_cluster[m, j]]
                cumulative[j + 1] = cumulative[j] + w
                fp += w * neg_pred[m, j]
            numerator = 0.0
            positives = 0.0
            tp = 0.0
            for j in range(pos_cluster.shape[1]):
                w = weights[b, pos_cluster[m, j]]
                positives += w
                tp += w * pos_pred[m, j]
                numerator += w * 0.5 * (cumulative[left[m, j]] + cumulative[right[m, j]])
            negatives = cumulative[-1]
            if positives == 0 or negatives == 0:
                out[m, b, :] = np.nan
            else:
                out[m, b, 0] = numerator / (positives * negatives)
                denominator = positives + tp + fp
                out[m, b, 1] = 2 * tp / denominator if denominator else 0.0
                out[m, b, 2] = tp / positives
    return out


def probability_for(z, method, selected):
    slots = ("Old_under", "Old_over", "Old_hybrid", "New_under", "New_over", "New_hybrid")
    base = z["Test_base_probability"]
    if method == "ValidationSelectedStatic":
        method = selected
    if method.startswith("Static_"):
        indices = [slots.index(s) for s in method.removeprefix("Static_").split("+")]
    elif method in ("Old3", "New3", "Equal6"):
        indices = {"Old3": [0, 1, 2], "New3": [3, 4, 5], "Equal6": list(range(6))}[method]
    else:
        return z["Test_" + method]
    return base[:, indices].mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=9999)
    ap.add_argument("--bootstrap-seed", type=int, default=20261027)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.replicates < 9999:
        ap.error("Use at least 9999 replicates for the 165-test Holm family")
    set_num_threads(args.threads)
    run = ROOT / RUN
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["protocol"] == "fixed_bankruptcy_standard_deslib_fit_only_fs_v1"
    seeds = manifest["config"]["seeds"]
    expected = [42, 123, 456, 789, 2024, 7, 21, 84, 168, 999]
    assert seeds == expected
    source_hashes = {}

    def remember(path):
        source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)

    for path in (run / "manifest.json", run / "partitions.csv", run / "metrics.csv", Path(__file__).resolve()):
        remember(path)
    test = pd.read_csv(run / "partitions.csv").query("partition == 'Test'")
    assert test.source_row_id.is_unique and test.company_name.notna().all()
    assert test.groupby("company_name").target.nunique().max() == 1
    all_metrics = pd.read_csv(run / "metrics.csv")
    pooled = all_metrics[all_metrics.period.eq("pooled")]
    fs_tags = list(pooled.fs.unique())
    assert fs_tags == ["no_fs", "mi_r50", "mi_r80", "cart_r50", "cart_r80", "shap_r50", "shap_r80", "rfe_r50", "rfe_r80"]
    output = args.output or ROOT / "results/statistical_tests/fixed_two_experiments" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output.resolve()
    assert output.is_relative_to(ROOT / "results/statistical_tests")
    output.mkdir(parents=True, exist_ok=False)
    comparisons = [("experiment1", "no_fs", method, "no_fs", "Equal6") for method in METHODS if method != "Equal6"]
    comparisons += [("experiment2", fs, method, "no_fs", method) for fs in fs_tags[1:] for method in FS_METHODS]
    specification = dict(run=RUN, protocol=manifest["protocol"], seeds=seeds, comparisons=comparisons,
                         metrics=METRICS, n_tests=3 * len(comparisons), replicates=args.replicates,
                         bootstrap_seed=args.bootstrap_seed, unit="company_name", resampling="uniform companies with replacement; all rows retained",
                         estimand="mean of ten seed-specific pooled metrics; not metrics of averaged probabilities",
                         null="two-sided zero metric difference", p_method="centered paired cluster bootstrap; plus-one Monte Carlo correction",
                         multiplicity="Holm across all 165 tests, both experiments and all three metrics",
                         scope="retrospective exploratory; fixed trained models, fixed split and fixed validation decisions",
                         test_selection=False, train_refit=False, ci="pointwise percentile 95%; not simultaneous intervals")
    assert specification["n_tests"] == 165
    (output / "specification.json").write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Analysis fixed before calculation: {output}", flush=True)
    cluster, companies = pd.factorize(test.company_name, sort=True)
    y = test.target.to_numpy()
    inputs, keys, point = [], [], []
    max_error = 0.0
    for fs in fs_tags:
        for seed in seeds:
            directory = run / f"seed_{seed}" / fs
            complete = json.loads((directory / "complete.json").read_text())
            assert complete["status"] == "completed"
            for name, key in (("predictions.npz", "predictions_sha256"), ("metrics.csv", "metrics_sha256")):
                assert sha(directory / name) == complete[key]
            for name in ("complete.json", "predictions.npz", "metrics.csv", "selection.json", "audit.json"):
                remember(directory / name)
            selected = json.loads((directory / "selection.json").read_text())
            assert selected["test_used_for_selection"] is False
            with np.load(directory / "predictions.npz", allow_pickle=False) as z:
                assert np.array_equal(z["Test_row_ids"], test.source_row_id.to_numpy())
                assert np.array_equal(z["Test_y"], y)
                for method in METHODS if fs == "no_fs" else FS_METHODS:
                    row = pooled[pooled.fs.eq(fs) & pooled.seed.eq(seed) & pooled.method.eq(method)]
                    assert len(row) == 1
                    row = row.iloc[0]
                    threshold = selected["thresholds"][method]
                    assert abs(row.threshold - threshold) < 1e-14
                    p = probability_for(z, method, selected["selected_static"])
                    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                    scores = np.array([roc_auc_score(y, p), f1_score(y, p >= threshold), recall_score(y, p >= threshold)])
                    error = np.max(np.abs(scores - row[list(METRICS)].to_numpy(dtype=float)))
                    max_error = max(max_error, error)
                    assert error < 1e-12
                    inputs.append(sort_inputs(y, p, cluster, threshold))
                    keys.append((fs, method, seed))
                    point.append(scores)
        print(f"Replay passed: {fs}", flush=True)
    arrays = [np.stack([item[k] for item in inputs]) for k in range(6)]
    # All 560 saved predictors receive the same company multiplicities.
    rng = np.random.default_rng(args.bootstrap_seed)
    weights = np.empty((args.replicates, len(companies)), dtype=np.int16)
    for b in range(args.replicates):
        weights[b] = np.bincount(rng.integers(0, len(companies), len(companies)), minlength=len(companies))
    print(f"Bootstrapping {len(companies)} companies, {len(keys)} predictors, {args.replicates} replicates", flush=True)
    boot = weighted_scores(weights, *arrays)
    assert np.isfinite(boot).all(), "Degenerate class bootstrap; do not silently discard replicates"
    replay_kernel = weighted_scores(np.ones((1, len(companies)), dtype=np.int16), *arrays)[:, 0]
    assert np.max(np.abs(replay_kernel - np.asarray(point))) < 1e-12
    key_frame = pd.DataFrame(keys, columns=["fs", "method", "seed"])
    points, draws, descriptive = {}, {}, []
    for (fs, method), group in key_frame.groupby(["fs", "method"], sort=False):
        index = group.index.to_numpy()
        assert len(index) == 10
        points[fs, method] = np.asarray(point)[index].mean(axis=0)
        draws[fs, method] = boot[index].mean(axis=0)
        for k, metric in enumerate(METRICS):
            descriptive.append(dict(fs=fs, method=method, metric=metric, mean=points[fs, method][k],
                                    seed_sd=np.asarray(point)[index, k].std(ddof=1), n_seeds=10))
    pd.DataFrame(descriptive).to_csv(output / "descriptive.csv", index=False)
    rows = []
    for experiment, fs_a, method_a, fs_b, method_b in comparisons:
        delta = points[fs_a, method_a] - points[fs_b, method_b]
        distribution = draws[fs_a, method_a] - draws[fs_b, method_b]
        for k, metric in enumerate(METRICS):
            centered = distribution[:, k] - delta[k]
            extreme = int(np.sum(np.abs(centered) >= abs(delta[k])))
            p_value = (extreme + 1) / (args.replicates + 1)
            lo, hi = np.quantile(distribution[:, k], [0.025, 0.975])
            rows.append(dict(experiment=experiment, fs_a=fs_a, method_a=method_a, fs_b=fs_b, method_b=method_b,
                             metric=metric, mean_a=points[fs_a, method_a][k], mean_b=points[fs_b, method_b][k],
                             difference=delta[k], ci_low=lo, ci_high=hi, p_raw=p_value,
                             extreme_count=extreme, p_mc_se=np.sqrt(p_value * (1 - p_value) / args.replicates)))
    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result.p_raw.to_numpy())
    result["reject_holm_005"] = result.p_holm < 0.05
    result.to_csv(output / "paired_tests.csv", index=False)
    np.savez_compressed(output / "bootstrap_seed_mean_metrics.npz", **{fs + "__" + method: v for (fs, method), v in draws.items()})
    validation = dict(status="passed", n_test_rows=len(test), n_test_positive=int(y.sum()),
                      n_test_companies=len(companies), n_positive_companies=int(test.groupby("company_name").target.first().sum()),
                      n_predictors_replayed=len(keys), n_tests=len(result), max_metric_replay_error=max_error,
                      kernel_unit_weight_error=float(np.max(np.abs(replay_kernel - np.asarray(point)))),
                      row_pairing="all 90 archives match exact source row IDs and labels", missing_configurations=0,
                      weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
                      rejected_by_experiment=result.groupby("experiment").reject_holm_005.sum().to_dict())
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    state = dict(status="completed", specification=specification, source_sha256=source_hashes,
                 output_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
                 created_at=datetime.now(timezone.utc).isoformat(), versions={})
    from importlib.metadata import version
    state["versions"] = {m: version(m) for m in ("numpy", "pandas", "scikit-learn", "numba")}
    (output / "manifest.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
