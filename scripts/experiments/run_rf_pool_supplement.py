"""Supplementary run: the same Old/New ensemble with a Random Forest pool instead of XGBoost.

The thesis ensemble uses an XGBoost pool. This wrapper runs the unchanged year-split runner of
another checkout with ``--classifier rf`` and the ensemble switched on; the only thing it
changes is the runner's guard "The ensemble pool is XGBoost; other classifiers need
--baseline-only". Nothing else in the runner is touched, so its source hashes in the
manifest stay those of the code that ran, and ``--reuse-tuning`` can copy the parameter
choices of the completed RF baseline run (same data, same code, same candidates).

Usage (cwd must be the checkout, because the runner reads data paths relative to it):
    ENSEMBLE_CHECKOUT=<checkout> ENSEMBLE_RUNNER=<bankruptcy|natality> \
        python run_rf_pool_supplement.py --classifier rf --fs-methods --reuse-tuning <rf baseline run> \
        --output-root <folder> [other runner options]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

CHECKOUT = Path(os.environ["ENSEMBLE_CHECKOUT"]).resolve()
RUNNERS = {
    "bankruptcy": "experiments.phase2_ensemble.dynamic.des.bankruptcy_year_splits_event",
    "natality": "experiments.phase2_ensemble.dynamic.des.natality_year_splits_event",
}
sys.path.insert(0, str(CHECKOUT))
runner = importlib.import_module(RUNNERS[os.environ["ENSEMBLE_RUNNER"]])

GUARD = "The ensemble pool is XGBoost; other classifiers need --baseline-only"
_error = argparse.ArgumentParser.error


def _error_without_pool_guard(self, message):
    if message == GUARD:
        return
    _error(self, message)


if __name__ == "__main__":
    argparse.ArgumentParser.error = _error_without_pool_guard
    run = runner.main()
    wrapper = Path(__file__).resolve()
    (Path(run) / "supplement_note.json").write_text(json.dumps({
        "purpose": "supplementary comparison: ensemble with a Random Forest pool (thesis pool is XGBoost)",
        "wrapper": wrapper.name,
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        "runner": runner.__file__,
        "change": f"only the guard '{GUARD}' is skipped; the runner source is unchanged",
        "feature_selection": "not run (ensemble without feature selection only)",
        "argv": sys.argv,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
