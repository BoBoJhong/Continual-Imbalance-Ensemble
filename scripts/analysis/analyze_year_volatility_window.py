"""Descriptive diagnostics for annual AP volatility and the 2/3/5/7-year window results.

For each Test feature year t and window width w (newest model trained on t-w-1..t-2):
  - training events in the newest window and in the FIFO3 history union;
  - Test prevalence and AP lift (AP / prevalence), since AP's random baseline is prevalence;
  - covariate drift: cross-validated AUC of a classifier separating newest-window rows
    from Test-year rows (0.5 = indistinguishable). Labels are never used for drift.
Post-hoc explanation only; no model, window or threshold is re-selected from these numbers.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase_flexible import rolling_bankruptcy_overlap_ensemble as base

WINDOW_RUN = ROOT / "results/phase_flexible/study3b_window_sensitivity/runs/20260916T151731817838Z"
WINDOW_ANALYSIS = WINDOW_RUN / "analysis/20260920T194618884788Z/by_year_all_seeds.csv"
WIDTHS, YEARS, POOL = (2, 3, 5, 7), range(2009, 2019), 3


def drift_auc(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> float:
    x = pd.concat([train[features], test[features]], ignore_index=True)
    domain = np.r_[np.zeros(len(train)), np.ones(len(test))]
    model = HistGradientBoostingClassifier(max_iter=100, random_state=0)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    score = cross_val_predict(model, x, domain, cv=folds, method="predict_proba")[:, 1]
    return roc_auc_score(domain, score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=WINDOW_RUN / "analysis" / ("year_volatility_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    args = parser.parse_args()
    data, features, _ = base.load_event_data()
    yearly = pd.read_csv(WINDOW_ANALYSIS)
    ap = yearly.groupby(["window_years", "method", "test_feature_year"]).PR_AUC.mean().unstack("method")
    rows = []
    for width in WIDTHS:
        for year in YEARS:
            newest = base.window_years(year - 2, width)
            history = range(year - 2 - POOL + 1 - width + 1, year - 1)
            train = data.loc[data.fyear.isin(newest)]
            test = data.loc[data.fyear == year]
            prevalence = test.target.mean()
            rows.append({"window_years": width, "test_feature_year": year,
                         "newest_train_years": f"{newest[0]}–{newest[-1]}",
                         "newest_train_events": int(train.target.sum()),
                         "fifo_history_events": int(data.loc[data.fyear.isin(history)].target.sum()),
                         "test_events": int(test.target.sum()), "test_prevalence": prevalence,
                         "AP_Newest": ap.loc[(width, year), "Newest"], "AP_FIFO3": ap.loc[(width, year), "FIFO3"],
                         "lift_Newest": ap.loc[(width, year), "Newest"] / prevalence,
                         "lift_FIFO3": ap.loc[(width, year), "FIFO3"] / prevalence,
                         "drift_AUC_newest_vs_test": drift_auc(train, test, features)})
            print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    # Within-year separability: 5-fold CV inside the Test year itself (no time gap), 5 repeats.
    # A reference for how visible that year's events are in X1-X18, not a deployable model.
    separability = []
    for year in YEARS:
        test = data.loc[data.fyear == year]
        scores = []
        for repeat in range(5):
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
            model = HistGradientBoostingClassifier(max_iter=100, random_state=repeat, class_weight="balanced")
            score = cross_val_predict(model, test[features], test.target, cv=folds, method="predict_proba")[:, 1]
            scores.append(average_precision_score(test.target, score))
        separability.append({"test_feature_year": year, "within_year_cv_AP": float(np.mean(scores))})
    year_level = (frame.groupby("test_feature_year")
                  .agg(test_events=("test_events", "first"), test_prevalence=("test_prevalence", "first"),
                       AP_mean_8configs=("AP_Newest", "mean"))
                  .reset_index())
    year_level["AP_mean_8configs"] = frame.groupby("test_feature_year")[["AP_Newest", "AP_FIFO3"]].mean().mean(axis=1).values
    year_level = year_level.merge(pd.DataFrame(separability), on="test_feature_year")
    annual_ap = frame.pivot(index="test_feature_year", columns="window_years", values=["AP_Newest", "AP_FIFO3"])
    consistency = annual_ap.corr().to_numpy()[np.triu_indices(annual_ap.shape[1], 1)].mean()
    print(f"mean pairwise corr of annual AP across configs={consistency:.3f}")
    print(f"corr(within-year AP, rolling AP)={year_level.within_year_cv_AP.corr(year_level.AP_mean_8configs):.3f}")
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "year_window_diagnostics.csv", index=False)
    year_level.to_csv(args.output / "year_level_diagnostics.csv", index=False)
    print(f"OUTPUT={args.output}")


if __name__ == "__main__":
    main()
