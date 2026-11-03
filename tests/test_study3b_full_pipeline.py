import json

import numpy as np
import pandas as pd
import pytest

from experiments.phase_flexible import study3b_full_pipeline as study


def history():
    rng = np.random.default_rng(42)
    frames = []
    for year in range(2000, 2007):
        frame = pd.DataFrame(rng.normal(size=(24, 4)), columns=["X1", "X2", "X3", "X4"])
        frame["target"] = np.tile([0, 0, 0, 1], 6)
        frame["fyear"] = year
        frame["company_name"] = [f"C{i}" for i in range(24)]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def config():
    return {"classifiers": ["lr", "xgb"], "sampling": list(study.SAMPLING),
            "fs_methods": ["cart"], "fs_ratios": [0.5], "test_years": [2005, 2006],
            "window_years": 3, "pool_size": 2, "threads": 1, "seeds": [42]}


class SmallModel:
    params = {"test_model": True}

    def fit(self, X, y):
        self.mean = np.asarray(X).mean(axis=0)
        return self

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-np.clip(np.asarray(X).sum(axis=1) + self.mean.sum(), -20, 20)))


def test_candidate_matrix_separates_sampling_classifier_and_time():
    groups = study.candidates(study.CLASSIFIERS, study.SAMPLING, 2009, 3)
    assert len(groups) == 70
    assert sum(g["family"] == "single" for g in groups.values()) == 24
    assert groups["FIFO_xgb_over"]["members"] == tuple(("xgb", "over", e) for e in (2005, 2006, 2007))
    assert len(groups["CombinedFIFO"]["members"]) == 54
    assert all(s != "none" for _, s, _ in groups["CombinedFIFO"]["members"])
    assert "Newest_xgb_none" in groups


def test_validation_ranking_and_tie_break_are_independent_of_test():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    ranking = study.rank_validation(y, {"B": p, "A": p, "C": 1 - p})
    assert [row["candidate"] for row in ranking] == ["A", "B", "C"]


@pytest.mark.parametrize("sampling,expected", list(study.SAMPLER_NAMES.items()))
def test_real_sampling_training_only_and_cache_reuse(tmp_path, monkeypatch, sampling, expected):
    calls = []
    def make(*args):
        calls.append(args)
        return SmallModel()
    monkeypatch.setattr(study, "make_model", make)
    data = history()
    cache = study.FrozenCache(data, ["X1", "X2", "X3", "X4"], tmp_path, config(), 42)
    first = cache.get("lr", sampling, 2003, "cart", 0.5)
    second = cache.get("lr", sampling, 2003, "cart", 0.5)
    pd.testing.assert_series_equal(first, second)
    assert len(calls) == 1
    audit = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert audit["train_years"] == [2001, 2002, 2003]
    assert audit["n_train"] == 72
    assert audit["n_features"] == 2
    assert audit["sampler"] == expected
    assert not audit["future_labels_used"]
    assert set(data.loc[first.index, "fyear"]) == {2004, 2005, 2006}
    # Cached prediction corruption must fail rather than silently alter results.
    path = next(tmp_path.glob("*.npz"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt cache"):
        cache.get("lr", sampling, 2003, "cart", 0.5)


@pytest.mark.parametrize("method", study.FS_METHODS)
def test_fs_methods_select_only_training_columns(method):
    data = history()
    columns = ["X1", "X2", "X3", "X4"]
    selected = study.feature_columns(data[columns], data.target, method, 0.5, 42, 1)
    assert len(selected) == 2
    assert set(selected) <= set(columns)


def test_pipeline_full_stages_fifo_resume_and_test_label_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "make_model", lambda *args: SmallModel())
    data = history()
    cfg = config()
    cfg["classifiers"] = ["lr"]
    cfg["sampling"] = ["none", "under"]
    columns = ["X1", "X2", "X3", "X4"]
    study.run_seed(data, columns, cfg, 42, tmp_path / "first")
    audit = pd.read_csv(tmp_path / "first/training_audit.csv")
    assert len(audit.loc[audit.fs == "none"]) == 6  # 2 initial + 1 new x 2 sampling
    predictions = pd.read_csv(tmp_path / "first/predictions.csv.gz")
    assert {"ValidationSelectedEnsemble", "ValidationSelectedFS", "SelectedEnsemble_none_r100"} <= set(predictions.method)
    assert not predictions.duplicated(["method", "test_feature_year", "source_row_id"]).any()
    before = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "first/cache").glob("*.npz")}
    study.run_seed(data, columns, cfg, 42, tmp_path / "first")
    after = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "first/cache").glob("*.npz")}
    assert before == after
    changed = data.copy()
    changed.loc[changed.fyear == 2006, "target"] = 1 - changed.loc[changed.fyear == 2006, "target"]
    study.run_seed(changed, columns, cfg, 42, tmp_path / "changed")
    for stage in ("ensemble", "fs"):
        a = json.loads((tmp_path / f"first/year_2006/{stage}_selection.json").read_text())
        b = json.loads((tmp_path / f"changed/year_2006/{stage}_selection.json").read_text())
        assert a == b
    altered = pd.read_csv(tmp_path / "changed/predictions.csv.gz")
    compare = ["method", "test_feature_year", "source_row_id", "y_proba", "f1_threshold", "fpr5_threshold"]
    pd.testing.assert_frame_equal(predictions[compare], altered[compare])


def test_failed_sampling_does_not_silently_drop_candidate(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("deliberate sampling failure")
    monkeypatch.setattr(study.FrozenCache, "get", fail)
    with pytest.raises(RuntimeError, match="refusing incomplete"):
        study.run_seed(history(), ["X1"], config(), 42, tmp_path)
    assert (tmp_path / "failures.json").exists()
    assert not list(tmp_path.glob("year_*"))


def test_classifier_factories_disable_additional_class_weight():
    for classifier in study.CLASSIFIERS:
        model = study.make_model(classifier, 7, 1)
        params = getattr(model, "params", None)
        if params is None:
            params = model.model.get_params()
        assert params.get("class_weight") is None
        assert not params.get("is_unbalance", False)
        assert params.get("scale_pos_weight", 1) in (1, None)
    assert not study.make_model("mlp", 7, 1).model.early_stopping
