from copy import deepcopy

import numpy as np
import pandas as pd

from src.data import DataLoader, DataPreprocessor, ImbalanceSampler
from src.ensemble import DynamicClassifierSelector, DynamicEnsembleSelector, EnsembleCombiner
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.models.model_pool import ModelPool
from src.utils import get_config_loader, get_seeds_from_config


class _OneDimensionalProbabilityModel:
    def __init__(self, probability=0.25):
        self.probability = probability

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.full(len(X), self.probability)


def test_preprocess_pipeline_uses_training_imputation_values():
    train = pd.DataFrame({"x": [1.0, 3.0]})
    test = pd.DataFrame({"x": [np.nan, 100.0]})

    preprocessor = DataPreprocessor()
    _, _, test_scaled, _ = preprocessor.preprocess_pipeline(
        train,
        pd.Series([0, 1]),
        test,
        pd.Series([0, 1]),
    )

    assert preprocessor.missing_fill_values_["x"] == 2.0
    np.testing.assert_allclose(test_scaled["x"], [0.0, 98.0])


def test_dynamic_selectors_accept_one_dimensional_probabilities(monkeypatch):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    X_dsel = np.array([[0.0], [1.0]])
    y_dsel = np.array([0, 1])
    X_test = np.array([[0.25], [0.75]])
    model = _OneDimensionalProbabilityModel()

    for selector in (DynamicEnsembleSelector(k=1), DynamicClassifierSelector(k=1)):
        selector.fit([model], X_dsel, y_dsel)
        proba, prediction = selector.predict(X_test)
        np.testing.assert_allclose(proba, [0.25, 0.25])
        np.testing.assert_array_equal(prediction, [0, 0])


def test_three_model_combinations_contain_exactly_three_models():
    old = {
        key: np.array([value], dtype=float)
        for key, value in zip(EnsembleCombiner.OLD_KEYS, [1, 2, 3], strict=True)
    }
    new = {
        key: np.array([value], dtype=float)
        for key, value in zip(EnsembleCombiner.NEW_KEYS, [10, 20, 30], strict=True)
    }

    combinations = EnsembleCombiner(old, new).get_predefined_combinations()
    three_model = {k: v for k, v in combinations.items() if k.startswith("ensemble_3")}

    assert len(combinations) == 49
    assert len(three_model) == 18
    np.testing.assert_allclose(
        combinations["ensemble_3a_under_over_under"],
        [(1 + 2 + 10) / 3],
    )


def test_model_custom_parameters_do_not_mutate_cached_config():
    config = get_config_loader()
    original = deepcopy(config.get("model_config", "xgboost.base_params"))

    first = XGBoostWrapper(name="first", max_depth=99)
    second = XGBoostWrapper(name="second")

    assert first.params["max_depth"] == 99
    assert "max_depth" not in second.params
    assert config.get("model_config", "xgboost.base_params") == original


def test_model_pool_propagates_random_state(monkeypatch):
    seen = {}

    class DummyModel:
        def __init__(self, name, random_state):
            seen["name"] = name
            seen["random_state"] = random_state

        def fit(self, X, y):
            seen["fit_rows"] = len(X)

    pool = ModelPool(random_state=123)
    monkeypatch.setattr(pool.sampler, "apply_sampling", lambda X, y, strategy: (X, y))
    pool.create_model_with_sampling(
        pd.DataFrame({"x": [1.0, 2.0]}),
        np.array([0, 1]),
        "undersampling",
        "dummy",
        model_class=DummyModel,
    )

    assert seen == {"name": "dummy", "random_state": 123, "fit_rows": 2}


def test_seed_list_comes_from_nested_base_config():
    assert get_seeds_from_config(get_config_loader()) == [
        42,
        123,
        456,
        789,
        2024,
        7,
        21,
        84,
        168,
        999,
    ]


def test_sampler_uses_yaml_parameters_and_overrides_run_seed():
    sampler = ImbalanceSampler(random_state=123)

    adasyn = sampler._params("sampling_strategies.oversampling.params")
    smote = sampler._params("sampling_strategies.hybrid.params")["smote"]

    assert adasyn == {
        "sampling_strategy": "auto",
        "n_neighbors": 5,
        "random_state": 123,
    }
    assert smote["k_neighbors"] == 5


def test_data_loader_supports_repository_dataset_schemas(tmp_path):
    bankruptcy_dir = tmp_path / "bankruptcy"
    medical_dir = tmp_path / "medical" / "diabetes130"
    stock_dir = tmp_path / "stock"
    bankruptcy_dir.mkdir(parents=True)
    medical_dir.mkdir(parents=True)
    stock_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "company_name": ["A", "B"],
            "fyear": [2017, 2018],
            "status_label": ["alive", "failed"],
            "X1": [1.0, 2.0],
            "Division": [1, 1],
        }
    ).to_csv(bankruptcy_dir / "american_bankruptcy_dataset.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2007-01-01", "2008-01-01"],
            "feature": [1.0, 2.0],
            "mortality": [0, 1],
        }
    ).to_csv(medical_dir / "diabetes130_medical.csv", index=False)
    (stock_dir / "stock_spx.csv").write_text(
        "metadata\nmetadata\n"
        "2007-01-01,10,11,9,10,100,0.1,0.1,10,10,10,0.2,50,0.3,0\n"
        "2008-01-01,9,10,8,9,110,-0.1,-0.1,9,9,9,0.3,40,-0.4,1\n",
        encoding="utf-8",
    )

    loader = DataLoader(tmp_path)
    bankruptcy_X, bankruptcy_y = loader.load_bankruptcy()
    medical_X, medical_y = loader.load_medical()
    stock_X, stock_y = loader.load_stock()

    assert list(bankruptcy_y) == [0, 1]
    assert list(bankruptcy_X.columns) == ["fyear", "X1"]
    assert list(medical_y) == [0, 1]
    assert list(medical_X["Year"]) == [2007, 2008]
    assert list(stock_y) == [0, 1]
    assert "Future_Returns_20" not in stock_X
    assert list(stock_X["Year"]) == [2007, 2008]


def test_config_loader_cache_is_scoped_by_directory(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "custom.yaml").write_text("value: 1\n", encoding="utf-8")
    (second_dir / "custom.yaml").write_text("value: 2\n", encoding="utf-8")

    assert get_config_loader(first_dir).get("custom", "value") == 1
    assert get_config_loader(second_dir).get("custom", "value") == 2


def test_metrics_include_imbalance_and_calibration_safe_outputs():
    metrics = compute_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
    )

    assert metrics["AUC"] == 1.0
    assert metrics["PR_AUC"] == 1.0
    assert metrics["Balanced_Accuracy"] == 1.0
    assert metrics["Cohen_Kappa"] == 1.0


def test_metrics_return_nan_auc_for_single_class_batch():
    metrics = compute_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert np.isnan(metrics["AUC"])
    assert np.isnan(metrics["PR_AUC"])
    assert np.isnan(metrics["Balanced_Accuracy"])
    assert np.isnan(metrics["Cohen_Kappa"])
