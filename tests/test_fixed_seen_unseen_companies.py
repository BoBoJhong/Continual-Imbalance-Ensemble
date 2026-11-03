"""Checks for seen/unseen company grouping and stratified company resampling."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "seen_unseen", Path(__file__).resolve().parents[1] / "scripts/analysis/fixed_seen_unseen_companies.py")
seen_unseen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seen_unseen)


def test_groups_follow_fit_partitions_not_validation():
    partitions = pd.DataFrame([
        ("A", 2005, 1, "Old_fit"), ("A", 2013, 1, "New_fit"), ("A", 2016, 1, "Test"),
        ("B", 2008, 0, "Old_fit"), ("B", 2015, 0, "Test"),
        ("C", 2014, 1, "DSEL"), ("C", 2015, 1, "Test"), ("C", 2016, 1, "Test"),
        ("D", 2017, 0, "Test"),
    ], columns=["company_name", "fyear", "target", "partition"])
    groups = seen_unseen.company_groups(partitions)
    assert groups.group.to_dict() == {"A": "seen_new_fit", "B": "seen_old_fit_only", "C": "unseen", "D": "unseen"}
    assert groups.loc["C", "in_2014_validation"] and groups.loc["C", "n_test_rows"] == 2
    assert groups.loc["A", "last_fit_year"] == 2013 and np.isnan(groups.loc["D", "last_fit_year"])


def test_stratified_weights_keep_each_stratum_size():
    strata = np.array([0, 0, 0, 1, 1, 2])
    weights = seen_unseen.stratified_weights(strata, 200, np.random.default_rng(0))
    for code, size in zip(*np.unique(strata, return_counts=True)):
        assert (weights[:, strata == code].sum(axis=1) == size).all()
    assert (weights[:, strata == 2] == 1).all()
