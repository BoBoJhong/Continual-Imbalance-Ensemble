"""Run the year-split statistical tests (spec section 11) on either dataset.

The test code is the unchanged scripts/analysis/year_splits_event_statistics.py of the
bankruptcy checkout (Wilcoxon over splits, Holm per family, Friedman across classifiers).
That script hard-codes the 15 bankruptcy splits and the retrain label "retrain_16"; this
wrapper only swaps in the dataset's split labels and retrain label, then calls its main().

Natality has 5 splits: a two-sided exact Wilcoxon on 5 non-zero differences cannot go below
p = 2/2^5 = 0.0625, so no paired comparison can be significant; Friedman tests still can.

Usage:
  python year_splits_statistics_any.py --dataset natality --stats-checkout <bankruptcy checkout> \
      --xgb-run <run> --baseline lgb=<run> ... --out <folder>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DATASETS = {"bankruptcy": (16, "retrain_16"), "natality": (6, "retrain_6")}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--stats-checkout", type=Path, required=True)
    known, rest = parser.parse_known_args()
    sys.path.insert(0, str((known.stats_checkout / "scripts/analysis").resolve()))
    import year_splits_event_statistics as tests

    n_years, retrain = DATASETS[known.dataset]
    tests.SPLITS = [f"split_{k}+{n_years - k}" for k in range(1, n_years)]

    def cell(table, period, sampling, metric):
        if period == "Retrain":
            return np.full(len(tests.SPLITS), table.loc[(retrain, "Retrain", sampling), metric])
        return np.array([table.loc[(split, period, sampling), metric] for split in tests.SPLITS])

    tests.cell = cell
    sys.argv = [sys.argv[0], *rest]
    tests.main()


if __name__ == "__main__":
    main()
