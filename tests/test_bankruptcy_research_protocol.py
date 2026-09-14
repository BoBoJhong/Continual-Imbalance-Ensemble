"""Research contracts: isolation, method identity and non-destructive run outputs."""
import json
import logging

import numpy as np
import pandas as pd
import pytest

from experiments._shared import common_bankruptcy
from experiments.phase2_ensemble import xgb_oldnew_ensemble_common as ensemble
from experiments.phase2_ensemble import xgb_year_split_shared as year_split
from experiments.phase3_feature._core import advanced_bankruptcy as fs_core
from experiments.phase4_drift import bankruptcy_ross_static_ensemble as legacy_ross
from experiments.phase_flexible import rolling_bankruptcy_adaptive as rolling
from experiments.phase_flexible import rolling_bankruptcy_overlap_ensemble as study3b
from experiments.phase_flexible import study3_ab_fair_comparison as fair_ab
from experiments.phase_flexible import study3b_continual_runtime as runtime
from experiments.phase_flexible import study3b_sliding_old_new_evaluation as sliding_old_new

LOGGER = logging.getLogger(__name__)


class _Model:
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-np.asarray(X)[:, 0]))


class _Sampler:
    def __init__(self, **kwargs):
        self.sampler = self

    def apply_sampling(self, X, y, **kwargs):
        return X, y


def test_advanced_fs_never_fits_on_validation_labels(monkeypatch):
    observed = []

    class SpySelector:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            observed.append((X.copy(), np.asarray(y).copy()))
            return self

        def transform(self, X):
            return X.iloc[:, :1].copy()

    # Fix membership to test information isolation rather than stratifier response.
    monkeypatch.setattr(fs_core, "_split_fit_val_by_year", lambda X, y, years: (
        X.iloc[:4].copy(), np.asarray(y)[:4], X.iloc[4:].copy(), np.asarray(y)[4:]
    ))
    monkeypatch.setattr(fs_core, "FeatureSelector", SpySelector)
    monkeypatch.setattr(fs_core, "XGBoostWrapper", _Model)
    monkeypatch.setattr(fs_core, "ImbalanceSampler", _Sampler)
    monkeypatch.setattr(fs_core, "_select_threshold_from_validation", lambda y, p: 0.5)
    old = pd.DataFrame({"a": [0., 1., 2., 3., np.nan, 999.], "b": [1.] * 6})
    new = old + 10
    labels = np.array([0, 1, 0, 1, 0, 1])
    for val_labels in ([0, 1], [1, 0]):
        y = labels.copy()
        y[4:] = val_labels
        rows = fs_core._run_one_split_one_fs(
            "split_4+12", old, y, np.full(6, 2000), new, y, np.full(6, 2010),
            old.iloc[:2], np.array([0, 1]), "mi_r50", "mutual_info", 0.5, LOGGER,
        )
        assert len(rows) == 11
    assert len(observed) == 2
    pd.testing.assert_frame_equal(observed[0][0], old.iloc[:4])
    pd.testing.assert_frame_equal(observed[0][0], observed[1][0])
    np.testing.assert_array_equal(observed[0][1], labels[:4])
    np.testing.assert_array_equal(observed[0][1], observed[1][1])


def test_missing_values_require_fit_partition_at_shared_loader(monkeypatch):
    monkeypatch.setattr(common_bankruptcy, "_load_us_1999_2018", lambda logger: (
        pd.DataFrame({"fyear": [2000, 2012, 2015], "a": [1., 2., np.nan]}),
        pd.Series([0, 1, 0]),
    ))
    with pytest.raises(ValueError, match="raw_features=True"):
        common_bankruptcy.get_bankruptcy_year_split(LOGGER, 2011)
    parts = common_bankruptcy.get_bankruptcy_year_split(LOGGER, 2011, raw_features=True)
    assert parts[4]["a"].isna().all()


def test_test_derived_boundary_is_rejected_before_reading_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(legacy_ross, "ROSS_SELECTED_PATH", tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="final Test"):
        legacy_ross._read_ross_boundary()


def test_ensemble_training_does_not_implicitly_load_tuning(monkeypatch):
    monkeypatch.setattr(ensemble, "XGBoostWrapper", _Model)

    def forbidden():
        raise AssertionError("Implicit tuning artifact read")

    monkeypatch.setattr(ensemble, "_load_tuned_xgb_params_map", forbidden)
    X = pd.DataFrame({"a": [0., 1.]})
    val, test = ensemble.train_one_sampling_xgb(X, np.array([0, 1]), X, X, _Sampler(), "undersampling", "test")
    np.testing.assert_array_equal(val, test)


def test_exported_baselines_identify_average_and_custom_des():
    columns = ensemble.expected_summary_wide_columns_static_only()
    assert "OldNewMean_under" in columns
    assert not any(column.startswith("Retrain") for column in columns)
    assert all(name not in {"Dynamic_KNORA_E", "Dynamic_KNORA_U", "Dynamic_DES_KNN"} for _, name in ensemble.DYNAMIC_DES_METHODS)


def test_year_split_emits_old_new_mean_not_retrain(monkeypatch):
    X = pd.DataFrame({"a": np.arange(12, dtype=float)})
    y = np.tile([0, 1], 6)
    years = np.full(12, 2000)
    monkeypatch.setattr(year_split, "train_one_sampling_xgb", lambda **kw: (
        np.full(len(kw["X_val_scaled"]), 0.5), np.full(len(kw["X_test_scaled"]), 0.5)
    ))
    monkeypatch.setattr(year_split, "DYNAMIC_DES_METHODS", ())
    # Remaining static subset calculations can use the real metric code.
    rows, _ = year_split.process_one_year_split(
        "split_4+12", 2002, LOGGER, _Sampler(),
        get_split=lambda *_: (X, y, X, y, X, y, years, years, years),
    )
    assert "OldNewMean" in {row["ensemble"] for row in rows}
    assert "Retrain" not in {row["ensemble"] for row in rows}


def test_rolling_seed_reaches_model_and_sampler(monkeypatch):
    seeds = []

    class SpySampler(_Sampler):
        def __init__(self, random_state):
            super().__init__()
            seeds.append(random_state)

    monkeypatch.setattr(rolling, "ImbalanceSampler", SpySampler)
    monkeypatch.setattr(rolling, "XGBoostWrapper", _Model)
    model = rolling._train_model(pd.DataFrame({"a": [0., 1.]}), np.array([0, 1]), "undersampling", "test", LOGGER, seed=123)
    assert seeds == [123]
    assert model.params["random_state"] == 123
    assert model.params["seed"] == 123
    assert json.loads(model.sampling_audit_["counts_after"]) == {"0": 1, "1": 1}


def test_rolling_sampling_failure_is_not_silently_relabelled(monkeypatch):
    class BrokenSampler(_Sampler):
        def apply_sampling(self, *args, **kwargs):
            raise ValueError("not enough minority samples")

    monkeypatch.setattr(rolling, "ImbalanceSampler", BrokenSampler)
    with pytest.raises(ValueError, match="no silent fallback"):
        rolling._train_model(pd.DataFrame({"a": [0., 1.]}), np.array([0, 1]), "oversampling", "test", LOGGER)


def test_rolling_real_xgboost_seed_alias_and_predictions():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=4, weights=[0.8, 0.2], random_state=9)
    model = rolling._train_model(pd.DataFrame(X), y, "undersampling", "contract_smoke", LOGGER, seed=123)
    configured = json.loads(model.model.get_booster().save_config())
    assert int(configured["learner"]["generic_param"]["seed"]) == 123
    probability = model.predict_proba(pd.DataFrame(X))
    assert probability.shape == (100,)
    assert np.isfinite(probability).all()


def _synthetic_history():
    return pd.DataFrame([
        {"company_name": f"C_{i}", "fyear": year, "a": float(i), "target": i % 2}
        for year in range(1999, 2010) for i in range(6)
    ])


def _mock_rolling_models(monkeypatch):
    monkeypatch.setattr(rolling, "ImbalanceSampler", _Sampler)
    monkeypatch.setattr(rolling, "XGBoostWrapper", _Model)
    monkeypatch.setattr(rolling, "_select_threshold", lambda y, p: 0.5)


def test_test_labels_do_not_change_rolling_selection(monkeypatch):
    _mock_rolling_models(monkeypatch)
    data = _synthetic_history()
    a = rolling._run_year(2009, data, ["a"], LOGGER, seed=7)
    data.loc[data.fyear == 2009, "target"] = 1 - data.loc[data.fyear == 2009, "target"]
    b = rolling._run_year(2009, data, ["a"], LOGGER, seed=7)
    assert a[3] == b[3]  # Includes boundary, weight and strategy (prevalence equal).
    assert a[2] == b[2]
    assert [row["threshold"] for row in a[0]] == [row["threshold"] for row in b[0]]
    assert {row["method"] for row in a[0]} == set(rolling.METHODS)
    assert all(row["company_name"].startswith("C_") for row in a[1])


def test_rolling_run_exports_separate_seeds_without_overwriting(monkeypatch, tmp_path):
    _mock_rolling_models(monkeypatch)
    data = _synthetic_history()
    monkeypatch.setattr(rolling, "load_bankruptcy_with_year", lambda *args, **kwargs: (
        data.drop(columns="target"), data.target,
    ))
    sentinel = tmp_path / "rolling_pooled_summary.csv"
    sentinel.write_text("legacy result", encoding="utf-8")
    rolling.main(["--seeds", "7", "11", "--test-years", "2009", "--output-root", str(tmp_path)])
    run_dir, = [path for path in tmp_path.iterdir() if path.is_dir()]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["seeds"] == [7, 11]
    assert manifest["label_availability"] == "unverified_no_event_dates"
    assert sentinel.read_text(encoding="utf-8") == "legacy result"
    for seed in (7, 11):
        predictions = pd.read_csv(run_dir / f"seed_{seed}" / "rolling_predictions.csv")
        assert predictions.seed.unique().tolist() == [seed]
        assert set(predictions.method) == set(rolling.METHODS)
        assert not predictions.duplicated(["test_year", "method", "source_row_id"]).any()
        metrics = pd.read_csv(run_dir / f"seed_{seed}" / "rolling_by_year.csv")
        assert {"PR_AUC", "n_positive", "n_test", "validation_PR_AUC"} <= set(metrics.columns)
    summary = pd.read_csv(run_dir / "rolling_seed_summary.csv")
    assert summary.AUC_count.eq(2).all()


def test_study3b_event_target_marks_only_failed_company_last_year():
    raw = pd.DataFrame([
        {"company_name": "alive", "fyear": 2000, "status_label": "alive", "x": 1.0},
        {"company_name": "alive", "fyear": 2001, "status_label": "alive", "x": 2.0},
        {"company_name": "failed", "fyear": 2000, "status_label": "failed", "x": 3.0},
        {"company_name": "failed", "fyear": 2001, "status_label": "failed", "x": 4.0},
    ])
    derived = study3b.build_event_target(raw)
    assert derived.target.tolist() == [0, 0, 0, 1]
    assert derived.event_year.tolist() == [2001, 2002, 2001, 2002]
    assert "target" not in raw.columns


def test_study3b_window_contract_and_fifo_shift():
    assert study3b.model_window_ends(2014) == (2010, 2011, 2012)
    assert study3b.model_window_ends(2015) == (2011, 2012, 2013)
    assert study3b.model_window_ends(2014, pool_size=1) == (2012,)
    assert study3b.model_window_ends(2014, pool_size=5) == (2008, 2009, 2010, 2011, 2012)
    assert study3b.window_years(2012) == (2010, 2011, 2012)
    assert study3b.window_years(2012, width=1) == (2012,)
    assert study3b.window_years(2012, width=5) == (2008, 2009, 2010, 2011, 2012)
    assert study3b.data_route_years(2014) == ((2010, 2011), (2012,))


def test_sliding_old_new_moves_old_new_validation_and_test_together():
    rounds = sliding_old_new.sliding_rounds(1999, 2018)
    assert len(rounds) == 16
    assert rounds[0] == {
        "round": 1,
        "model_id": "M2001",
        "old_start": 1999,
        "old_end": 2000,
        "old_years": "1999,2000",
        "new_year": 2001,
        "train_years": "1999,2000,2001",
        "validation_feature_year": 2002,
        "primary_test_feature_year": 2003,
        "primary_test_event_year": 2004,
        "test_used_for_selection": False,
    }
    assert rounds[-1]["old_years"] == "2014,2015"
    assert rounds[-1]["new_year"] == 2016
    assert rounds[-1]["validation_feature_year"] == 2017
    assert rounds[-1]["primary_test_feature_year"] == 2018


def test_sliding_old_new_fifo_warms_up_then_keeps_latest_three():
    assert sliding_old_new.active_model_years(2001, 2001) == (2001,)
    assert sliding_old_new.active_model_years(2002, 2001) == (2001, 2002)
    assert sliding_old_new.active_model_years(2003, 2001) == (2001, 2002, 2003)
    assert sliding_old_new.active_model_years(2010, 2001) == (2008, 2009, 2010)


def _synthetic_event_history():
    rows = []
    for year in range(1999, 2010):
        for i in range(8):
            rows.append({
                "company_name": f"C_{year}_{i}", "fyear": year,
                "event_year": year + 1, "status_label": "failed" if i == 0 else "alive",
                "x": float(i - 4), "target": int(i == 0),
            })
    return pd.DataFrame(rows)


def _mock_study3b_models(monkeypatch):
    monkeypatch.setattr(study3b, "ImbalanceSampler", _Sampler)
    monkeypatch.setattr(study3b, "XGBoostWrapper", _Model)


def test_study3b_test_labels_cannot_change_thresholds_or_protocol(monkeypatch):
    _mock_study3b_models(monkeypatch)
    data = _synthetic_event_history()
    first = study3b._run_year(2009, data, ["x"], LOGGER, seed=7)
    changed = data.copy()
    changed.loc[changed.fyear == 2009, "target"] = 1 - changed.loc[changed.fyear == 2009, "target"]
    second = study3b._run_year(2009, changed, ["x"], LOGGER, seed=7)
    assert first[3] == second[3]
    assert [row["f1_threshold_from_validation"] for row in first[0]] == [
        row["f1_threshold_from_validation"] for row in second[0]
    ]
    assert [row["fpr5_threshold_from_validation"] for row in first[0]] == [
        row["fpr5_threshold_from_validation"] for row in second[0]
    ]
    assert {row["method"] for row in first[0]} == set(study3b.METHODS)


def test_study3b_fifo_reuses_overlap_and_fits_one_new_window(monkeypatch):
    _mock_study3b_models(monkeypatch)
    data = _synthetic_event_history()
    pool = {}
    first = study3b._run_year(2008, data, ["x"], LOGGER, seed=7, model_pool=pool)
    first_models = dict(pool)
    second = study3b._run_year(2009, data, ["x"], LOGGER, seed=7, model_pool=pool)
    assert set(first_models) == {2004, 2005, 2006}
    assert set(pool) == {2005, 2006, 2007}
    assert pool[2005] is first_models[2005]
    assert pool[2006] is first_models[2006]
    assert first[3]["b_model_added_end_years"] == "2004,2005,2006"
    assert second[3]["b_model_added_end_years"] == "2007"
    assert second[3]["b_model_removed_end_years"] == "2004"
    assert [row["pool_action"] for row in second[2] if row["route"] == "B_model"] == [
        "retained", "retained", "added"
    ]


def test_study3b_pool_one_is_exactly_the_recent_model(monkeypatch):
    _mock_study3b_models(monkeypatch)
    _, predictions, audits, protocol = study3b._run_year(
        2009, _synthetic_event_history(), ["x"], LOGGER, seed=7, pool_size=1
    )
    frame = pd.DataFrame(predictions)
    recent = frame.loc[frame.method == "Recent3y", "y_proba"].to_numpy()
    pool_one = frame.loc[
        frame.method == "B_model_FIFO1_equal", "y_proba"
    ].to_numpy()
    np.testing.assert_array_equal(recent, pool_one)
    assert protocol["pool_size"] == 1
    assert protocol["b_model_end_years"] == "2007"
    assert len([row for row in audits if row["route"] == "B_model"]) == 1


def test_study3b_window_width_changes_training_years_and_method_name(monkeypatch):
    _mock_study3b_models(monkeypatch)
    _, predictions, audits, protocol = study3b._run_year(
        2009,
        _synthetic_event_history(),
        ["x"],
        LOGGER,
        seed=7,
        pool_size=1,
        window_width=5,
    )
    frame = pd.DataFrame(predictions)
    assert "Recent5y" in set(frame.method)
    fitted = [row for row in audits if row["route"] == "B_model"]
    assert fitted[0]["train_years"] == "2003,2004,2005,2006,2007"
    assert protocol["recent_years"] == "2003,2004,2005,2006,2007"
    assert protocol["window_years"] == 5


@pytest.mark.parametrize(
    ("imbalance_method", "expected_weight"),
    [("none", None), ("scale_pos_weight", 7.0)],
)
def test_study3b_imbalance_ablation_does_not_resample(
    monkeypatch, imbalance_method, expected_weight
):
    monkeypatch.setattr(study3b, "XGBoostWrapper", _Model)

    class ForbiddenSampler:
        def __init__(self, **kwargs):
            raise AssertionError("non-sampling ablations must not instantiate a sampler")

    monkeypatch.setattr(study3b, "ImbalanceSampler", ForbiddenSampler)
    data = _synthetic_event_history()
    fitted = study3b._fit_window_model(
        data,
        ["x"],
        (2000, 2001, 2002),
        "ablation_contract",
        LOGGER,
        seed=7,
        imbalance_method=imbalance_method,
    )
    assert fitted.audit["n_positive_before"] == fitted.audit["n_positive_after"] == 3
    assert fitted.audit["n_negative_before"] == fitted.audit["n_negative_after"] == 21
    assert fitted.audit["sampler"] == "None"
    assert fitted.audit["scale_pos_weight"] == expected_weight
    assert fitted.model.params.get("scale_pos_weight") == expected_weight


def test_fair_a_boundary_selection_never_uses_test_labels(monkeypatch):
    class FairModel:
        def __init__(self, model_id, years):
            self.model_id = model_id
            self.train_years = years
            self.audit = {"model_id": model_id, "train_years": ",".join(map(str, years))}

        def predict_proba(self, frame, feature_cols):
            values = frame[feature_cols[0]].to_numpy(dtype=float)
            return 1.0 / (1.0 + np.exp(-values))

    monkeypatch.setattr(
        fair_ab,
        "_fit_years",
        lambda data, feature_cols, years, model_id, logger, seed: FairModel(model_id, years),
    )
    data = _synthetic_event_history()
    first = fair_ab._run_a_and_baselines(2009, data, ["x"], LOGGER, seed=7)
    changed = data.copy()
    changed.loc[changed.fyear == 2009, "target"] = 1 - changed.loc[
        changed.fyear == 2009, "target"
    ]
    second = fair_ab._run_a_and_baselines(2009, changed, ["x"], LOGGER, seed=7)
    assert first[3] == second[3]
    assert first[2] == second[2]
    assert first[3]["test_used_for_selection"] is False


def test_company_cluster_bootstrap_keeps_method_predictions_paired():
    rows = []
    for method, proba in {
        fair_ab.REFERENCE_METHOD: [0.1, 0.9, 0.2, 0.8],
        "baseline": [0.4, 0.6, 0.7, 0.3],
    }.items():
        for index, (company, y_true) in enumerate(
            [("A", 0), ("A", 1), ("B", 0), ("B", 1)]
        ):
            rows.append({
                "test_feature_year": 2009 + index % 2,
                "source_row_id": index,
                "company_name": company,
                "method": method,
                "y_true": y_true,
                "y_proba": proba[index],
                "y_pred_f1": int(proba[index] >= 0.5),
            })
    result = fair_ab.paired_company_cluster_bootstrap(
        pd.DataFrame(rows), replicates=20, seed=3
    )
    assert set(result.metric) == set(fair_ab.BOOTSTRAP_METRICS)
    assert (result.n_company_clusters == 2).all()
    assert (result.reference_method == fair_ab.REFERENCE_METHOD).all()


def test_study3b_run_is_versioned_and_auditable(monkeypatch, tmp_path):
    _mock_study3b_models(monkeypatch)
    data = _synthetic_event_history()
    annual = data.groupby(["fyear", "event_year"], as_index=False).agg(
        n_rows=("target", "size"), n_events=("target", "sum")
    )
    annual["event_rate"] = annual.n_events / annual.n_rows
    monkeypatch.setattr(study3b, "load_event_data", lambda: (data, ["x"], annual))
    sentinel = tmp_path / "legacy.csv"
    sentinel.write_text("keep", encoding="utf-8")
    study3b.main(["--seeds", "7", "11", "--test-years", "2009", "--output-root", str(tmp_path)])
    run_dir, = [path for path in tmp_path.iterdir() if path.is_dir()]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["target_definition"].startswith("failed company")
    assert manifest["ensemble_weighting"] == "equal_only_no_weight_tuning"
    assert sentinel.read_text(encoding="utf-8") == "keep"
    for seed in (7, 11):
        protocol = pd.read_csv(run_dir / f"seed_{seed}" / "study3b_protocol.csv")
        assert protocol.latest_train_feature_year.tolist() == [2007]
        assert protocol.validation_feature_year.tolist() == [2008]
        predictions = pd.read_csv(run_dir / f"seed_{seed}" / "study3b_predictions.csv")
        assert set(predictions.method) == set(study3b.METHODS)
        assert not predictions.duplicated(["test_feature_year", "method", "source_row_id"]).any()


def _runtime_raw():
    rows = []
    for year in range(1999, 2011):
        for i in range(4):
            row = {
                "company_name": f"C_{year}_{i}",
                "fyear": year,
                "status_label": "failed" if i == 0 else "alive",
            }
            row.update({f"X{feature}": float(year + i + feature) for feature in range(1, 19)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_runtime_label_ledger_is_versioned_and_maturity_gated(tmp_path):
    raw_path = tmp_path / "raw.csv"
    labels_path = tmp_path / "labels.csv"
    raw = _runtime_raw()
    raw.to_csv(raw_path, index=False)
    original = raw_path.read_bytes()
    manifest = runtime.prepare_label_ledger(raw_path, labels_path)
    original_labels = labels_path.read_bytes()
    second_manifest = runtime.prepare_label_ledger(raw_path, labels_path)
    assert raw_path.read_bytes() == original
    assert labels_path.read_bytes() == original_labels
    assert second_manifest["created_at"] == manifest["created_at"]
    assert manifest["label_available_year_rule"].startswith("fyear + 1")
    assert manifest["evidence_status"] == runtime.LABEL_EVIDENCE
    data, _ = runtime._load_training_data(raw_path, labels_path, 2006)
    assert data.fyear.max() == 2005
    assert data.label_available_year.max() <= 2006
    assert (tmp_path / "labels.manifest.json").is_file()


class _RuntimeModel:
    def __init__(self, model_id, train_years):
        self.model_id = model_id
        self.train_years = train_years
        self.audit = {"model_id": model_id}

    def predict_proba(self, frame):
        return 1 / (1 + np.exp(-frame["X1"].to_numpy() / 1000))


def _mock_runtime_training(monkeypatch, raw_path, labels_path):
    raw = _runtime_raw()
    labels = raw[["company_name", "fyear"]].copy()
    labels["event_year"] = labels.fyear + 1
    labels["target"] = np.tile([1, 0, 0, 0], len(labels) // 4)
    labels["label_available_year"] = labels.event_year
    data = raw[["company_name", "fyear", *runtime.FEATURE_COLUMNS]].merge(
        labels, on=["company_name", "fyear"], validate="one_to_one"
    )
    raw.to_csv(raw_path, index=False)
    labels.to_csv(labels_path, index=False)
    label_hash = runtime._sha256(labels_path)
    manifest = {
        "labels_sha256": label_hash,
        "label_source": runtime.LABEL_SOURCE,
        "evidence_status": runtime.LABEL_EVIDENCE,
    }
    monkeypatch.setattr(runtime, "_load_training_data", lambda *args: (data, manifest))
    monkeypatch.setattr(
        runtime,
        "_fit_window_model",
        lambda data, features, years, model_id, logger, seed: _RuntimeModel(model_id, years),
    )

    def persist(model, state_dir, feature_columns):
        return {
            "model_id": model.model_id,
            "end_year": model.train_years[-1],
            "train_years": list(model.train_years),
            "artifact_dir": f"models/{model.model_id}",
            "model_sha256": "mock",
            "preprocessor_sha256": "mock",
            "created_at": runtime._now(),
        }

    monkeypatch.setattr(runtime, "_persist_model", persist)
    monkeypatch.setattr(
        runtime,
        "_load_model",
        lambda record, state_dir: _RuntimeModel(record["model_id"], tuple(record["train_years"])),
    )
    return raw, label_hash


def test_runtime_initialize_update_and_label_blind_predict(monkeypatch, tmp_path):
    raw_path = tmp_path / "raw.csv"
    labels_path = tmp_path / "labels.csv"
    state_dir = tmp_path / "state"
    output_root = tmp_path / "predictions"
    raw, _ = _mock_runtime_training(monkeypatch, raw_path, labels_path)

    initial = runtime.initialize_state(state_dir, raw_path, labels_path, 2008, 7, LOGGER)
    assert [record["end_year"] for record in initial["active_models"]] == [2004, 2005, 2006]
    updated = runtime.update_state(state_dir, raw_path, labels_path, 2009, LOGGER)
    assert [record["end_year"] for record in updated["active_models"]] == [2005, 2006, 2007]
    assert updated["updates"][-1]["added_end_years"] == [2007]
    assert updated["updates"][-1]["retained_end_years"] == [2005, 2006]
    assert updated["updates"][-1]["removed_end_years"] == [2004]
    assert [record["end_year"] for record in updated["retired_models"]] == [2004]
    assert updated["retired_models"][0]["retirement_reason"] == "fifo_capacity_3"
    with pytest.raises(ValueError, match="advance exactly one year"):
        runtime.update_state(state_dir, raw_path, labels_path, 2011, LOGGER)

    monkeypatch.setattr(
        runtime, "_load_label_ledger",
        lambda *args: (_ for _ in ()).throw(AssertionError("predict opened labels")),
    )
    run_dir = runtime.predict_current(state_dir, raw_path, 2009, output_root)
    predictions = pd.read_csv(run_dir / "predictions.csv")
    prediction_manifest = json.loads(
        (run_dir / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    assert len(predictions) == len(raw[raw.fyear == 2009])
    assert "y_true" not in predictions.columns
    assert "status_label" not in predictions.columns
    assert prediction_manifest["test_label_read"] is False
    assert prediction_manifest["labels_path_opened"] is False
    assert len(list((state_dir / "history").glob("*.json"))) == 2


def test_runtime_rejects_concurrent_state_update(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / ".update.lock").write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked"):
        with runtime._state_lock(state_dir):
            pass
