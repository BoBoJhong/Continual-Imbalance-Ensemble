"""Contrast a never-updated model with the annual sliding-window protocol.

Static: the round-1 newest model (trained 2005-2007) predicts every test year
with the F1 threshold fixed once on the 2008 validation year. Sliding: each
test year t uses the model trained on t-4..t-2 and a threshold from t-1.
All probabilities come from the frozen study3b_full_pipeline cache, so no
model is refitted and no test label influences any threshold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase_flexible import rolling_bankruptcy_overlap_ensemble as base  # noqa: E402

RUN = ROOT / "results/phase_flexible/study3b_full_pipeline/runs/20260928T185026381432Z"
YEARS = list(range(2009, 2019))


def cached(seed, classifier, sampling, end):
    with np.load(RUN / f"seed_{seed}/cache/{classifier}_{sampling}_M{end}_none_r100.npz") as saved:
        return pd.Series(saved["probability"], index=saved["source_row_id"])


def scores(y, p, threshold):
    return {"AUC": roc_auc_score(y, p), "F1": f1_score(y, p >= threshold, zero_division=0),
            "Recall": recall_score(y, p >= threshold, zero_division=0)}


def run(seeds, classifier, sampling):
    data, _, _ = base.load_event_data()
    rows, pooled = [], []
    for seed in seeds:
        static = cached(seed, classifier, sampling, YEARS[0] - 2)
        validation = data.loc[data.fyear == YEARS[0] - 1]
        fixed = base._select_f1_threshold(validation.target.to_numpy(), static.loc[validation.index].to_numpy())
        parts = {"static": [], "sliding": []}
        for year in YEARS:
            test, val = data.loc[data.fyear == year], data.loc[data.fyear == year - 1]
            newest = cached(seed, classifier, sampling, year - 2)
            threshold = base._select_f1_threshold(val.target.to_numpy(), newest.loc[val.index].to_numpy())
            y = test.target.to_numpy()
            for name, p, t in (("static", static.loc[test.index].to_numpy(), fixed),
                               ("sliding", newest.loc[test.index].to_numpy(), threshold)):
                rows.append({"seed": seed, "strategy": name, "test_feature_year": year,
                             "model_age": year - (YEARS[0] - 2 if name == "static" else year - 2),
                             "threshold": t, **scores(y, p, t)})
                parts[name].append((y, p, np.full(len(y), t)))
        for name, chunks in parts.items():
            y, p, t = (np.concatenate(c) for c in zip(*chunks))
            pooled.append({"seed": seed, "strategy": name, "AUC": roc_auc_score(y, p),
                           "F1": f1_score(y, p >= t, zero_division=0), "Recall": recall_score(y, p >= t, zero_division=0)})
    yearly, pooled = pd.DataFrame(rows), pd.DataFrame(pooled)
    # The sliding branch must reproduce the saved pipeline metrics exactly.
    saved = pd.concat([pd.read_csv(RUN / f"seed_{s}/by_year.csv") for s in seeds])
    saved = saved.loc[saved.method == f"Newest_{classifier}_{sampling}"].set_index(["seed", "test_feature_year"])
    check = yearly.loc[yearly.strategy == "sliding"].set_index(["seed", "test_feature_year"])
    for metric in ("AUC", "F1", "Recall"):
        np.testing.assert_allclose(check[metric], saved.loc[check.index, metric], atol=1e-12)
    return yearly, pooled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--classifier", default="rf")
    parser.add_argument("--sampling", default="under")
    args = parser.parse_args()
    yearly, pooled = run(args.seeds, args.classifier, args.sampling)
    output = RUN / "analysis" / f"static_vs_sliding_{args.classifier}_{args.sampling}"
    output.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(output / "by_year.csv", index=False)
    pooled.to_csv(output / "pooled.csv", index=False)
    print(yearly.groupby(["strategy", "test_feature_year"])[["model_age", "threshold", "AUC", "F1", "Recall"]].mean().round(3).to_string())
    print(pooled.groupby("strategy")[["AUC", "F1", "Recall"]].mean().round(3).to_string())
    print(f"OUTPUT={output}")
