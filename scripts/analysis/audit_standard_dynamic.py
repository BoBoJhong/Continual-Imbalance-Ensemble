"""Replay saved dynamic probabilities and recompute fixed-year experiment metrics.

The replay uses literal KNORA/OLA/LCA rules, not the fitted DESlib selectors.
For KNORA-U it preserves DESlib 0.3.7's weighted-score softmax convention.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from experiments.phase2_ensemble.dynamic.des import bankruptcy_standard_dynamic as study


def replay_dynamic(method, dsel_y, dsel_probability, query_probability, neighbors):
    """Independent vectorized reference for binary DESlib with DFP/IH disabled."""
    dsel_prediction = dsel_probability > 0.5
    prediction = query_probability > 0.5
    truth = dsel_y[neighbors]
    correct = dsel_prediction[neighbors] == truth[:, :, None]
    if method == "KNORA_E":
        prefix = np.cumprod(correct, axis=1).sum(axis=1)
        weights = prefix == prefix.max(axis=1, keepdims=True)
        result = (query_probability * weights).sum(axis=1) / weights.sum(axis=1)
    elif method == "KNORA_U":
        weights = correct.sum(axis=1)
        scores = np.column_stack([
            ((1 - query_probability) * weights).mean(axis=1),
            (query_probability * weights).mean(axis=1),
        ])
        result = softmax(scores, axis=1)[:, 1]
    elif method in ("DCS_OLA", "DCS_LCA"):
        if method == "DCS_OLA":
            competence = correct.mean(axis=1)
        else:
            belongs = truth[:, :, None] == prediction[:, None, :]
            denominator = belongs.sum(axis=1)
            competence = np.divide((correct & belongs).sum(axis=1), denominator,
                                   out=np.zeros(denominator.shape), where=denominator > 0)
        selected = competence.argmax(axis=1)
        result = query_probability[np.arange(len(query_probability)), selected]
    else:
        raise ValueError(method)
    agree = np.all(prediction == prediction[:, :1], axis=1)
    result[agree] = query_probability[agree].mean(axis=1)
    return result


def independent_metrics(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "Recall": recall, "Precision": precision, "G_Mean": np.sqrt(recall * specificity),
            "Type1_Error": 1 - specificity, "Type2_Error": 1 - recall,
            "Brier": brier_score_loss(y, p)}


def audit_run(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Only a completed experiment can pass the final audit")
    if manifest["provenance"]["versions"]["deslib"] != "0.3.7":
        raise ValueError("Replay is defined for DESlib 0.3.7")
    partitions = pd.read_csv(run / "partitions.csv").set_index("source_row_id", drop=False)
    expected_n = 78_682
    if partitions.index.duplicated().any() or len(partitions) != expected_n:
        raise ValueError("Unexpected raw cohort or duplicated partition rows")
    cohorts = {name: partitions.loc[partitions.partition.eq(name)] for name in
               ("Old_fit", "New_fit", "DSEL", "Calibration", "Test")}
    if (cohorts["Old_fit"].fyear.max() != 2011 or cohorts["New_fit"].fyear.max() != 2013
            or set(cohorts["DSEL"].fyear.unique()) != {2014}
            or set(cohorts["Calibration"].fyear.unique()) != {2014}
            or set(cohorts["Test"].fyear.unique()) != {2015, 2016, 2017, 2018}):
        raise ValueError("Year boundaries changed")
    checked_probabilities, checked_metric_rows, completed, configuration_frames = 0, 0, [], []
    for seed in manifest["config"]["seeds"]:
        for method, ratio in manifest["config"]["fs_configs"]:
            tag = "no_fs" if method == "none" else f"{method}_r{round(ratio * 100)}"
            directory = run / f"seed_{seed}" / tag
            status = json.loads((directory / "complete.json").read_text())
            if (status["metrics_sha256"] != study.sha256(directory / "metrics.csv") or
                    status["predictions_sha256"] != study.sha256(directory / "predictions.npz")):
                raise ValueError("Result hash mismatch")
            rows = pd.read_csv(directory / "metrics.csv")
            configuration_frames.append(rows)
            selection = json.loads((directory / "selection.json").read_text())
            representation = json.loads((directory / "audit.json").read_text())
            if representation["selector_fit_scope"] != "Old_fit_only":
                raise ValueError("Supervised FS scope changed")
            with np.load(directory / "predictions.npz", allow_pickle=False) as cache:
                for name in ("DSEL", "Calibration", "Test"):
                    np.testing.assert_array_equal(cache[f"{name}_row_ids"], cohorts[name].index.to_numpy())
                    np.testing.assert_array_equal(cache[f"{name}_y"], cohorts[name].target)
                replay_threads = manifest["config"]["threads"]
                knn = KNeighborsClassifier(n_neighbors=manifest["config"]["k"], n_jobs=replay_threads).fit(
                    cache["DSEL_X"], cache["DSEL_y"])
                with threadpool_limits(replay_threads):
                    for name in ("Calibration", "Test"):
                        # DESlib bypasses unanimous queries BEFORE its KNN call.
                        # Reproduce that same batch: at tied distances, querying
                        # all rows can change brute-force floating-point tie order.
                        base_prediction = cache[f"{name}_base_probability"] > 0.5
                        disagreement = np.flatnonzero(np.any(base_prediction != base_prediction[:, :1], axis=1))
                        neighbors = np.zeros((len(base_prediction), manifest["config"]["k"]), dtype=int)
                        if len(disagreement):
                            neighbors[disagreement] = knn.kneighbors(cache[f"{name}_X"][disagreement, :],
                                                                    return_distance=False)
                        for dynamic_method in study.METHODS:
                            expected = replay_dynamic(dynamic_method, cache["DSEL_y"],
                                                      cache["DSEL_base_probability"],
                                                      cache[f"{name}_base_probability"], neighbors)
                            np.testing.assert_allclose(expected, cache[f"{name}_{dynamic_method}"],
                                                       rtol=1e-10, atol=1e-12)
                            checked_probabilities += len(expected)
                predictions = {name: tuple(cache[f"{part}_base_probability"][:, indices].mean(axis=1)
                                           for part in ("Calibration", "Test"))
                               for name, indices in study.static_candidates().items()}
                predictions.update({name: (cache[f"Calibration_{name}"], cache[f"Test_{name}"])
                                    for name in study.METHODS})
                ranking = sorted((
                    (average_precision_score(cache["Calibration_y"], p[0]), name)
                    for name, p in predictions.items() if name.startswith("Static_")
                ), key=lambda pair: (-pair[0], pair[1]))
                if ranking[0][1] != selection["selected_static"]:
                    raise ValueError("Static selection cannot be reproduced from calibration")
                predictions["ValidationSelectedStatic"] = predictions[ranking[0][1]]
                test_years = cohorts["Test"].fyear.to_numpy()
                for row in rows.itertuples(index=False):
                    cp, tp = predictions[row.method]
                    if row.threshold != study.select_threshold(cache["Calibration_y"], cp):
                        # CSV round trips may differ by one floating-point bit.
                        np.testing.assert_allclose(row.threshold, study.select_threshold(cache["Calibration_y"], cp),
                                                   rtol=0, atol=1e-14)
                    # Every pooled method and every dynamic annual row is independently recomputed.
                    if str(row.period) != "pooled" and row.method not in study.METHODS:
                        continue
                    mask = np.ones(len(tp), dtype=bool) if str(row.period) == "pooled" else test_years == int(row.period)
                    values = independent_metrics(cache["Test_y"][mask], tp[mask], row.threshold)
                    for metric, value in values.items():
                        np.testing.assert_allclose(getattr(row, metric), value, rtol=1e-10, atol=1e-12)
                    checked_metric_rows += 1
            completed.append({"seed": seed, "fs": tag, "status": "verified"})
            print(f"VERIFIED seed={seed} fs={tag}", flush=True)
    raw = pd.read_csv(run / "metrics.csv")
    pd.testing.assert_frame_equal(raw, pd.concat(configuration_frames, ignore_index=True),
                                  check_exact=False, rtol=1e-10, atol=1e-12)
    expected_rows = len(completed) * 63 * (1 + len(cohorts["Test"].fyear.unique()))
    if len(raw) != expected_rows or raw.duplicated(["seed", "fs", "method", "period"]).any():
        raise ValueError("Final method matrix is incomplete or duplicated")
    derived = raw.groupby(["fs", "n_features", "method", "period"])[list(study.METRICS)].agg(["mean", "std"])
    derived.columns = [f"{metric}_{stat}" for metric, stat in derived.columns]
    saved = pd.read_csv(run / "summary.csv").set_index(["fs", "n_features", "method", "period"])
    np.testing.assert_allclose(derived.to_numpy(), saved.loc[derived.index].to_numpy(),
                               rtol=1e-10, atol=1e-12, equal_nan=True)
    pooled = raw.loc[raw.period.astype(str).eq("pooled")]
    controls = pooled.loc[pooled.fs.eq("no_fs")].set_index(["seed", "method"])
    checked_fs_rows = 0
    for filename, selected_policy in (("fs_paired_differences.csv", False),
                                      ("fs_selected_policy_differences.csv", True)):
        differences = pd.read_csv(run / filename)
        expected = pooled.loc[~pooled.fs.eq("no_fs") &
                              (pooled.method.eq("ValidationSelectedStatic") == selected_policy)]
        keys = ["seed", "fs", "method"]
        pd.testing.assert_index_equal(differences.set_index(keys).sort_index().index,
                                      expected.set_index(keys).sort_index().index)
        for row in differences.itertuples(index=False):
            control = controls.loc[(row.seed, row.method)]
            for metric in study.METRICS:
                np.testing.assert_allclose(getattr(row, f"{metric}_delta"),
                                           getattr(row, metric) - control[metric],
                                           rtol=1e-10, atol=1e-12)
        checked_fs_rows += len(differences)
    audit = {"status": "verified", "at": study.now(), "protocol": manifest["protocol"],
             "configurations": len(completed), "metric_rows": len(raw),
             "independently_recomputed_metric_rows": checked_metric_rows,
             "independently_replayed_dynamic_probabilities": checked_probabilities,
             "verified_fs_difference_rows": checked_fs_rows,
             "manifest_sha256": study.sha256(run / "manifest.json"), "checks": completed,
             "scope": "all dynamic calibration/Test probabilities; all pooled methods; dynamic yearly metrics; calibration-only ranking/thresholds; checksums; partition membership; aggregate mean/SD; FS paired differences",
             "limits": "does not verify real event dates or independently retrain all models; seed SD is not independent-data uncertainty"}
    study.atomic_json(run / "audit_validation.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "checks"}, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    audit_run(parser.parse_args().run.resolve())
