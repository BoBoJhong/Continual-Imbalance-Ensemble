"""Protect common origins, model reuse, and validation/test isolation."""
import logging

import numpy as np
import pandas as pd
import pytest

from experiments.phase_flexible import study3b_window_sensitivity as study
from scripts.analysis.analyze_study3b_window_sensitivity import annual_block_sensitivity, summarize


def test_full_pool_origins_and_history():
    rows = pd.DataFrame(study.protocol_rows(1999, 2018))
    assert len(rows) == 40
    for width, group in rows.groupby("window_years"):
        assert group.test_feature_year.tolist() == list(range(2009, 2019))
        assert (group.pool_history_start >= 1999).all()
        assert (group.newest_train_end < group.validation_feature_year).all()
        assert (group.validation_feature_year < group.test_feature_year).all()
        assert (group.pool_history_years == width + 2).all()
    with pytest.raises(ValueError, match="Insufficient"):
        study.protocol_rows(2010, 2018)


def test_reuses_frozen_models_and_never_fits_validation_or_test(monkeypatch, tmp_path):
    records = []

    class FakeWindow:
        def __init__(self, years, name):
            self.train_years = years
            self.audit = {"model_id": name}

        def predict_proba(self, frame, features):
            assert frame.fyear.min() > max(self.train_years)
            return frame[features[0]].to_numpy()

    def fit(data, features, years, name, logger, *, seed):
        records.append(years)
        return FakeWindow(years, name)

    monkeypatch.setattr(study.base, "_fit_window_model", fit)
    data = pd.DataFrame([
        {"fyear": year, "company_name": str(i), "target": i % 2,
         "X1": [.1, .3, .5, .9][i]}
        for year in range(1999, 2011) for i in range(4)
    ])
    protocol = pd.DataFrame(study.protocol_rows(1999, 2010))
    study.run_seed(data, ["X1"], protocol, 42, tmp_path / "first", logging.getLogger(__name__))
    assert len(records) == 4 * 4  # 3 initial + 1 new model for each of four widths
    assert len(set(records)) == len(records)
    assert max(max(years) for years in records) == 2008
    before = pd.read_csv(tmp_path / "first/predictions.csv.gz")
    # Test labels must not affect predictions or thresholds, even if metrics change.
    changed = data.copy()
    changed.loc[changed.fyear >= 2009, "target"] = 1 - changed.loc[changed.fyear >= 2009, "target"]
    study.run_seed(changed, ["X1"], protocol, 42, tmp_path / "changed", logging.getLogger(__name__))
    after = pd.read_csv(tmp_path / "changed/predictions.csv.gz")
    columns = ["window_years", "method", "y_proba", "f1_threshold", "fpr5_threshold"]
    pd.testing.assert_frame_equal(before.loc[before.test_feature_year == 2009, columns],
                                  after.loc[after.test_feature_year == 2009, columns])


def test_macro_ap_keeps_equal_year_weights():
    from scripts.analysis.analyze_study3b_window_sensitivity import METRICS

    rows = [{"window_years": 3, "method": "FIFO3", "seed": seed,
             "test_feature_year": year, **{m: value for m in METRICS}}
            for seed in (42, 123) for year, value in ((2009, .1), (2010, .3))]
    pooled = pd.DataFrame([{ "window_years": 3, "method": "FIFO3", "seed": seed,
                            **{m: .12 for m in METRICS}} for seed in (42, 123)])
    summary = summarize(pd.DataFrame(rows), pooled).iloc[0]
    assert np.isclose(summary.macro_PR_AUC, .2)
    assert np.isclose(summary.pooled_PR_AUC, .12)
    assert np.isclose(summary.macro_PR_AUC_seed_sd, 0)


def test_annual_blocks_preserve_paired_differences():
    annual = pd.DataFrame([
        {"window_years": width, "method": method, "seed": seed,
         "test_feature_year": year, "PR_AUC": .01 * width + .02 * (method == "FIFO3") + .001 * year}
        for width in study.WIDTHS for method in ("Newest", "FIFO3")
        for seed in (42, 123) for year in range(10)
    ])
    result = annual_block_sensitivity(annual, replicates=20)
    assert len(result) == 14
    same_width = result.loc[result.comparison.str.endswith("Newest")]
    np.testing.assert_allclose(same_width[["difference", "ci95_low", "ci95_high"]], .02)
