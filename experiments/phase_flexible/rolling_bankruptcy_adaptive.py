"""
Leakage-safe annual walk-forward evaluation for ROSS + DAWCE + AdaptiveChoice.

For each test year t:
    train/search history: 1999 .. t-2
    validation:           t-1
    test:                 t

The validation batch selects the ROSS boundary, DAWCE weight, AdaptiveChoice
candidate, and classification threshold. The test batch is used exactly once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.phase4_drift.bankruptcy_drift_stream import load_bankruptcy_with_year
from src.data import ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_logger, set_seed


START_YEAR = 1999
FIRST_TEST_YEAR = 2009
LAST_TEST_YEAR = 2018
MIN_OLD_YEARS = 3
POOL_SAMPLING = ("undersampling", "oversampling", "hybrid")
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
METRICS = ("AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error")
METHODS = ("Retrain_under", "New_under", "Equal6", "DAWCE_AUC", "AdaptiveChoice_AUC")
OUTPUT_DIR = project_root / "results" / "phase_flexible" / "rolling_bankruptcy"
ALPHA = 0.05


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def _score_key(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    threshold = _select_threshold(y_true, proba)
    auc = _safe_auc(y_true, proba)
    auc_key = auc if np.isfinite(auc) else -np.inf
    f1 = f1_score(y_true, proba >= threshold, zero_division=0)
    return auc_key, float(f1)


def _select_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        score = f1_score(y_true, proba >= threshold, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


def _fit_transform(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    train_imputed = imputer.fit_transform(train[feature_cols])
    validation_imputed = imputer.transform(validation[feature_cols])
    test_imputed = imputer.transform(test[feature_cols])

    train_scaled = scaler.fit_transform(train_imputed)
    validation_scaled = scaler.transform(validation_imputed)
    test_scaled = scaler.transform(test_imputed)

    return (
        pd.DataFrame(train_scaled, columns=feature_cols, index=train.index),
        pd.DataFrame(validation_scaled, columns=feature_cols, index=validation.index),
        pd.DataFrame(test_scaled, columns=feature_cols, index=test.index),
    )


def _train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sampling: str,
    name: str,
    logger,
) -> XGBoostWrapper:
    sampler = ImbalanceSampler(random_state=42)
    try:
        X_resampled, y_resampled = sampler.apply_sampling(X_train, y_train, strategy=sampling)
    except ValueError as error:
        logger.warning(f"{sampling} failed for {name} ({error}); falling back to no sampling")
        X_resampled, y_resampled = X_train, y_train

    model = XGBoostWrapper(name=name, use_imbalance=False)
    model.fit(X_resampled, np.asarray(y_resampled))
    return model


def _evaluate_method(
    method: str,
    test_year: int,
    y_validation: np.ndarray,
    validation_proba: np.ndarray,
    y_test: np.ndarray,
    test_proba: np.ndarray,
) -> tuple[dict, list[dict]]:
    threshold = _select_threshold(y_validation, validation_proba)
    validation_metrics = compute_metrics(y_validation, validation_proba, threshold=threshold)
    test_metrics = compute_metrics(y_test, test_proba, threshold=threshold)

    row = {
        "test_year": test_year,
        "validation_year": test_year - 1,
        "method": method,
        "threshold": threshold,
        **{f"validation_{metric}": validation_metrics[metric] for metric in METRICS},
        **test_metrics,
    }
    predictions = [
        {
            "test_year": test_year,
            "method": method,
            "row_id": int(row_id),
            "y_true": int(y_true),
            "y_proba": float(proba),
            "y_pred": int(proba >= threshold),
            "threshold": threshold,
        }
        for row_id, (y_true, proba) in enumerate(zip(y_test, test_proba))
    ]
    return row, predictions


def _run_year(test_year: int, data: pd.DataFrame, feature_cols: list[str], logger):
    train_end = test_year - 2
    train = data[data["fyear"].between(START_YEAR, train_end)].copy()
    validation = data[data["fyear"] == test_year - 1].copy()
    test = data[data["fyear"] == test_year].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"Missing train/validation/test batch for test year {test_year}")

    X_train, X_validation, X_test = _fit_transform(train, validation, test, feature_cols)
    y_train = train["target"].to_numpy()
    y_validation = validation["target"].to_numpy()
    y_test = test["target"].to_numpy()

    boundary_rows: list[dict] = []
    boundary_outputs: dict[int, dict] = {}
    for boundary in range(START_YEAR + MIN_OLD_YEARS, train_end + 1):
        old_mask = train["fyear"] < boundary
        new_mask = train["fyear"] >= boundary
        pool_validation: dict[str, np.ndarray] = {}
        pool_test: dict[str, np.ndarray] = {}

        for period, mask in (("Old", old_mask), ("New", new_mask)):
            for sampling in POOL_SAMPLING:
                key = f"{period}_{sampling.replace('sampling', '')}".rstrip("_")
                model = _train_model(
                    X_train.loc[mask],
                    y_train[mask.to_numpy()],
                    sampling,
                    f"rolling_{test_year}_{boundary}_{key}",
                    logger,
                )
                pool_validation[key] = model.predict_proba(X_validation)
                pool_test[key] = model.predict_proba(X_test)

        old_keys = [key for key in pool_validation if key.startswith("Old_")]
        new_keys = [key for key in pool_validation if key.startswith("New_")]
        old3_validation = np.mean([pool_validation[key] for key in old_keys], axis=0)
        old3_test = np.mean([pool_test[key] for key in old_keys], axis=0)
        new3_validation = np.mean([pool_validation[key] for key in new_keys], axis=0)
        new3_test = np.mean([pool_test[key] for key in new_keys], axis=0)
        boundary_auc, boundary_f1 = _score_key(y_validation, new3_validation)
        boundary_rows.append(
            {
                "test_year": test_year,
                "validation_year": test_year - 1,
                "boundary_year": boundary,
                "old_start": START_YEAR,
                "old_end": boundary - 1,
                "new_start": boundary,
                "new_end": train_end,
                "validation_new3_AUC": boundary_auc,
                "validation_new3_F1": boundary_f1,
            }
        )
        boundary_outputs[boundary] = {
            "pool_validation": pool_validation,
            "pool_test": pool_test,
            "old3_validation": old3_validation,
            "old3_test": old3_test,
            "new3_validation": new3_validation,
            "new3_test": new3_test,
        }

    best_boundary_row = max(
        boundary_rows,
        key=lambda row: (row["validation_new3_AUC"], row["validation_new3_F1"], row["boundary_year"]),
    )
    best_boundary = int(best_boundary_row["boundary_year"])
    selected = boundary_outputs[best_boundary]
    pool_validation = selected["pool_validation"]
    pool_test = selected["pool_test"]

    best_single = max(pool_validation, key=lambda key: _score_key(y_validation, pool_validation[key]))
    equal6_validation = np.mean(list(pool_validation.values()), axis=0)
    equal6_test = np.mean(list(pool_test.values()), axis=0)

    weight_candidates = []
    for new_weight in WEIGHT_GRID:
        validation_proba = (
            (1.0 - new_weight) * selected["old3_validation"] + new_weight * selected["new3_validation"]
        )
        weight_candidates.append(
            (
                _score_key(y_validation, validation_proba),
                float(new_weight),
                validation_proba,
                (1.0 - new_weight) * selected["old3_test"] + new_weight * selected["new3_test"],
            )
        )
    _, best_new_weight, dawce_validation, dawce_test = max(
        weight_candidates, key=lambda item: (item[0][0], item[0][1], item[1])
    )

    adaptive_candidates = {
        best_single: (pool_validation[best_single], pool_test[best_single]),
        "DAWCE_AUC": (dawce_validation, dawce_test),
    }
    adaptive_choice = max(
        adaptive_candidates,
        key=lambda key: _score_key(y_validation, adaptive_candidates[key][0]),
    )
    adaptive_validation, adaptive_test = adaptive_candidates[adaptive_choice]

    retrain = _train_model(X_train, y_train, "undersampling", f"rolling_{test_year}_retrain_under", logger)
    method_probas = {
        "Retrain_under": (retrain.predict_proba(X_validation), retrain.predict_proba(X_test)),
        "New_under": (pool_validation["New_under"], pool_test["New_under"]),
        "Equal6": (equal6_validation, equal6_test),
        "DAWCE_AUC": (dawce_validation, dawce_test),
        "AdaptiveChoice_AUC": (adaptive_validation, adaptive_test),
    }

    method_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for method, (validation_proba, test_proba) in method_probas.items():
        row, predictions = _evaluate_method(
            method, test_year, y_validation, validation_proba, y_test, test_proba
        )
        method_rows.append(row)
        prediction_rows.extend(predictions)

    selection_row = {
        "test_year": test_year,
        "train_start": START_YEAR,
        "train_end": train_end,
        "validation_year": test_year - 1,
        "selected_boundary": best_boundary,
        "selected_old_period": f"{START_YEAR}-{best_boundary - 1}",
        "selected_new_period": f"{best_boundary}-{train_end}",
        "best_single_model": best_single,
        "dawce_new_weight": best_new_weight,
        "adaptive_choice": adaptive_choice,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "test_positive_rate": float(np.mean(y_test)),
    }
    logger.info(
        f"Year {test_year}: boundary={best_boundary}, best_single={best_single}, "
        f"DAWCE new weight={best_new_weight:.2f}, AdaptiveChoice={adaptive_choice}"
    )
    return method_rows, prediction_rows, boundary_rows, selection_row


def _pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in predictions.groupby("method", sort=False):
        metrics = compute_metrics(group["y_true"], group["y_proba"], y_pred=group["y_pred"])
        rows.append({"method": method, "n_predictions": len(group), **metrics})
    return pd.DataFrame(rows).sort_values("AUC", ascending=False)


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    running_max = 0.0
    total = len(valid)
    for rank, (index, p_value) in enumerate(valid.items()):
        running_max = max(running_max, min(1.0, (total - rank) * float(p_value)))
        adjusted.loc[index] = running_max
    return adjusted


def _paired_tests(by_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    adaptive = by_year[by_year["method"] == "AdaptiveChoice_AUC"].set_index("test_year")
    for method in METHODS:
        if method == "AdaptiveChoice_AUC":
            continue
        baseline = by_year[by_year["method"] == method].set_index("test_year")
        common_years = adaptive.index.intersection(baseline.index)
        for metric in ("AUC", "F1", "Recall", "Precision"):
            a = adaptive.loc[common_years, metric].to_numpy()
            b = baseline.loc[common_years, metric].to_numpy()
            difference = a - b
            p_value = 1.0 if np.allclose(difference, 0.0) else float(
                wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue
            )
            rows.append(
                {
                    "comparison": f"AdaptiveChoice_AUC vs {method}",
                    "baseline": method,
                    "metric": metric,
                    "n_years": len(common_years),
                    "adaptive_mean": float(np.mean(a)),
                    "baseline_mean": float(np.mean(b)),
                    "mean_difference": float(np.mean(difference)),
                    "p_two_sided": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm"] = _holm_adjust(result["p_two_sided"])
    result["significant_holm"] = result["p_holm"] < ALPHA
    return result.sort_values(["p_holm", "comparison", "metric"])


def _validate_outputs(
    by_year: pd.DataFrame,
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    expected_years = set(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    assert set(by_year["test_year"]) == expected_years
    assert set(by_year["method"]) == set(METHODS)
    assert len(by_year) == len(expected_years) * len(METHODS)
    assert set(selections["test_year"]) == expected_years
    assert (selections["train_end"] == selections["test_year"] - 2).all()
    assert (selections["validation_year"] == selections["test_year"] - 1).all()
    assert not by_year[list(METRICS)].isna().any().any()
    per_method_counts = predictions.groupby("method").size()
    assert per_method_counts.nunique() == 1


def main() -> None:
    set_seed(42)
    logger = get_logger("RollingBankruptcyAdaptive", console=True, file=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_bankruptcy_with_year(logger)
    data = X.copy()
    data["target"] = np.asarray(y)
    feature_cols = [column for column in X.columns if column != "fyear"]

    method_rows: list[dict] = []
    prediction_rows: list[dict] = []
    boundary_rows: list[dict] = []
    selection_rows: list[dict] = []
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        logger.info(f"=== Walk-forward test year {test_year} ===")
        year_methods, year_predictions, year_boundaries, year_selection = _run_year(
            test_year, data, feature_cols, logger
        )
        method_rows.extend(year_methods)
        prediction_rows.extend(year_predictions)
        boundary_rows.extend(year_boundaries)
        selection_rows.append(year_selection)

    by_year = pd.DataFrame(method_rows)
    predictions = pd.DataFrame(prediction_rows)
    boundaries = pd.DataFrame(boundary_rows)
    selections = pd.DataFrame(selection_rows)
    _validate_outputs(by_year, predictions, selections)

    pooled = _pooled_metrics(predictions)
    tests = _paired_tests(by_year)
    outputs = {
        "rolling_by_year.csv": by_year,
        "rolling_predictions.csv": predictions,
        "rolling_boundary_candidates.csv": boundaries,
        "rolling_selection_history.csv": selections,
        "rolling_pooled_summary.csv": pooled,
        "rolling_wilcoxon_holm.csv": tests,
    }
    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False, float_format="%.8f")
        logger.info(f"Saved {path}")

    print("\nPooled out-of-sample metrics:")
    print(pooled.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAdaptive selection history:")
    print(
        selections[
            ["test_year", "selected_boundary", "best_single_model", "dawce_new_weight", "adaptive_choice"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
