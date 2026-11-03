"""Method identity, held-out selection, train-only FS and result contracts."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from experiments.phase2_ensemble.dynamic.des import bankruptcy_standard_dynamic as study
from src.ensemble.standard_dynamic import (
    METHODS,
    ProbabilityClassifier,
    StandardKNORAE,
    StandardKNORAU,
    fit_selector,
)


def history():
    rng = np.random.default_rng(5)
    frames = []
    for year in (1999, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018):
        frame = pd.DataFrame(rng.normal(size=(24, 4)), columns=["X1", "X2", "X3", "X4"])
        frame["target"] = np.tile([0, 0, 0, 1], 6)
        frame["fyear"] = year
        frame["company_name"] = [f"C{i}" for i in range(24)]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["source_row_id"] = data.index
    return data.set_index("source_row_id", drop=False)


def test_knora_e_shrinks_neighborhood_instead_of_immediate_pool_average():
    selector = StandardKNORAE(k=3, voting="soft")
    selector.DSEL_processed_ = np.array([[True, True], [True, False], [False, True]])
    selector.n_classes_ = 2
    region = np.array([[0, 1, 2]])
    competence = selector.estimate_competence(region)
    np.testing.assert_array_equal(competence, [[2, 1]])
    np.testing.assert_array_equal(selector.select(competence), [[True, False]])
    probability = np.array([[[0.1, 0.9], [0.8, 0.2]]])
    result = selector.predict_proba_with_ds(np.array([[1, 0]]), probability, region)
    np.testing.assert_allclose(result, [[0.1, 0.9]])


def test_knora_e_zero_correct_prefix_retains_whole_pool():
    selector = StandardKNORAE(k=2)
    selector.DSEL_processed_ = np.array([[False, False], [True, False]])
    competence = selector.estimate_competence(np.array([[0, 1]]))
    np.testing.assert_array_equal(competence, [[0, 0]])
    np.testing.assert_array_equal(selector.select(competence), [[True, True]])


def test_knora_u_uses_correct_neighbor_count_not_equal_selected_weight():
    selector = StandardKNORAU(k=3, voting="soft")
    selector.DSEL_processed_ = np.array([[True, True], [True, False], [True, False]])
    selector.n_classes_ = 2
    region = np.array([[0, 1, 2]])
    np.testing.assert_array_equal(selector.estimate_competence(region), [[3, 1]])
    result = selector.predict_proba_with_ds(
        np.array([[1, 0]]), np.array([[[0.1, 0.9], [0.8, 0.2]]]), region)
    # DESlib 0.3.7's soft-voting API applies softmax to weighted mean scores.
    # Preserve the library's probability convention rather than replacing it.
    expected = softmax([(3 * 0.1 + 0.8) / 2, (3 * 0.9 + 0.2) / 2])[1]
    np.testing.assert_allclose(result[:, 1], [expected])
    assert result[0, 1] != pytest.approx((0.9 + 0.2) / 2)


@pytest.mark.parametrize("method", METHODS)
def test_official_selectors_fit_under_current_sklearn(method):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + rng.normal(size=120) > 0).astype(int)
    pool = [DecisionTreeClassifier(max_depth=i, random_state=i).fit(X[:60], y[:60])
            for i in (1, 2, 3)]
    selector = fit_selector(method, pool, X[60:90], y[60:90], threads=1)
    probability = selector.predict_proba(X[90:])
    assert probability.shape == (30, 2)
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    assert selector.DFP is False and selector.with_IH is False


def test_fixed_year_partitions_are_disjoint_and_validation_only_2014():
    parts = study.partition_data(history())
    assert parts["Old_fit"].fyear.max() == 2011
    assert set(parts["New_fit"].fyear.unique()) == {2012, 2013}
    assert set(parts["DSEL"].fyear.unique()) == {2014}
    assert set(parts["Calibration"].fyear.unique()) == {2014}
    assert parts["Test"].fyear.min() == 2015
    assert not set(parts["DSEL"].company_name) & set(parts["Calibration"].company_name)
    all_ids = [i for frame in parts.values() for i in frame.index]
    assert len(all_ids) == len(set(all_ids))


def test_fs_and_preprocessing_never_learn_from_validation_or_test(monkeypatch):
    parts = study.partition_data(history())
    seen = []

    def spy(X, y, *args):
        seen.append((X.copy(), np.asarray(y).copy()))
        return ["X1", "X2"]

    monkeypatch.setattr(study, "feature_columns", spy)
    features = ["X1", "X2", "X3", "X4"]
    _, first = study.fit_representation(parts, features, "mi", 0.5, 42, 1)
    changed = {name: frame.copy() for name, frame in parts.items()}
    for name in ("DSEL", "Calibration", "Test"):
        changed[name][features] = changed[name][features] + 10_000
        changed[name]["target"] = 1 - changed[name].target
    _, second = study.fit_representation(changed, features, "mi", 0.5, 42, 1)
    assert first == second
    assert set(seen[0][0].index) == set(parts["Old_fit"].index)
    pd.testing.assert_frame_equal(seen[0][0], seen[1][0])
    np.testing.assert_array_equal(seen[0][1], seen[1][1])


def test_all_49_mixed_static_combinations_are_present():
    candidates = study.static_candidates()
    mixed = {name: idx for name, idx in candidates.items() if name.startswith("Static_")}
    assert len(mixed) == 49
    assert [sum(len(idx) == k for idx in mixed.values()) for k in range(2, 7)] == [9, 18, 15, 6, 1]
    assert all(any(i < 3 for i in idx) and any(i >= 3 for i in idx) for idx in mixed.values())


def test_vectorized_threshold_matches_exhaustive_f1_grid():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=150)
    p = rng.uniform(size=150)
    grid = np.arange(0.01, 1.0, 0.01)
    expected = grid[np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid])]
    assert study.select_threshold(y, p) == expected


def test_test_labels_cannot_change_threshold_or_static_selection():
    parts = study.partition_data(history())
    rng = np.random.default_rng(5)
    p = {"Static_A": (rng.uniform(size=len(parts["Calibration"])), rng.uniform(size=len(parts["Test"]))),
         "Static_B": (rng.uniform(size=len(parts["Calibration"])), rng.uniform(size=len(parts["Test"])))}
    _, first = study.evaluate_predictions(parts, p, {"seed": 42, "fs": "no_fs"})
    parts["Test"] = parts["Test"].copy()
    parts["Test"]["target"] = 1 - parts["Test"].target
    _, second = study.evaluate_predictions(parts, p, {"seed": 42, "fs": "no_fs"})
    assert first == second


class _SmallModel:
    params = {"synthetic_contract_only": True}

    def __init__(self, offset):
        self.offset = offset

    def fit(self, X, y):
        self.mean = float(np.asarray(X)[:, 0].mean())
        return self

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-np.clip(np.asarray(X)[:, 0] + self.mean + self.offset, -20, 20)))


def test_real_selector_pipeline_writes_auditable_method_matrix(tmp_path, monkeypatch):
    parts = study.partition_data(history())
    calls = []

    def make(*args):
        model = _SmallModel(len(calls) * 0.4 - 1)
        calls.append(model)
        return model

    class Sampler:
        def __init__(self, **kwargs):
            self.sampler = self

        def apply_sampling(self, X, y, strategy):
            return X, y

    monkeypatch.setattr(study, "make_model", make)
    monkeypatch.setattr(study, "ImbalanceSampler", Sampler)
    rows = study.run_configuration(parts, ["X1", "X2", "X3", "X4"], 42, "none", 1,
                                   1, 7, tmp_path)
    assert len(calls) == 6
    assert len(rows) == 315
    assert set(METHODS) <= set(rows.method)
    assert not rows.duplicated(["method", "period"]).any()
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["metrics_sha256"] == study.sha256(tmp_path / "metrics.csv")
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["selector_fit_scope"] == "Old_fit_only"
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as cache:
        assert set(cache["DSEL_row_ids"]).isdisjoint(cache["Calibration_row_ids"])
        assert set(cache["DSEL_row_ids"]).isdisjoint(cache["Test_row_ids"])


def test_probability_adapter_preserves_probabilities_and_tie_rule():
    class Model:
        def predict_proba(self, X):
            return np.array([0.2, 0.5, 0.9])

    model = ProbabilityClassifier(Model(), ["X1"])
    X = np.zeros((3, 1))
    np.testing.assert_array_equal(model.predict(X), [0, 0, 1])
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], [0.2, 0.5, 0.9])
