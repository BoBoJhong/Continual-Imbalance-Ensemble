"""
Validation-selected annual walk-forward evaluation for ROSS + DAWCE + AdaptiveChoice.

For each test year t:
    train/search history: 1999 .. t-2
    validation:           t-1
    test:                 t

The validation batch selects the ROSS boundary, DAWCE weight, AdaptiveChoice
candidate, and classification threshold. The test batch is used exactly once.
Label availability remains unverified: this is an exploratory retrospective
protocol, not a validated prospective bankruptcy forecasting experiment.
"""
from __future__ import annotations

import sys
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.phase4_drift.bankruptcy_drift_stream import load_bankruptcy_with_year, US_CSV
from src.data import ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_logger, set_seed, get_config_loader, get_seeds_from_config


START_YEAR = 1999
FIRST_TEST_YEAR = 2009
LAST_TEST_YEAR = 2018
MIN_OLD_YEARS = 3
POOL_SAMPLING = ("undersampling", "oversampling", "hybrid")
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
METRICS = ("AUC", "PR_AUC", "F1", "G_Mean", "Recall", "Precision", "Balanced_Accuracy", "Type1_Error", "Type2_Error")
RECENT_WINDOWS = (1, 3, 5)
METHODS = ("Retrain_under", "ROSS_New_under", "Equal6", "DAWCE_AUC", "AdaptiveChoice_AUC") + tuple(
    f"Recent{years}y_under" for years in RECENT_WINDOWS
)
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
    *,
    seed: int = 42,
) -> XGBoostWrapper:
    sampler = ImbalanceSampler(random_state=seed)
    try:
        X_resampled, y_resampled = sampler.apply_sampling(X_train, y_train, strategy=sampling)
    except ValueError as error:
        raise ValueError(f"Sampling failed for {name} ({sampling}); no silent fallback") from error

    # The YAML already defines XGBoost's alias `seed`; override both aliases.
    model = XGBoostWrapper(name=name, use_imbalance=False, random_state=seed, seed=seed)
    model.fit(X_resampled, np.asarray(y_resampled))
    model.sampling_audit_ = {
        "model": name, "seed": seed, "effective_sampler": type(sampler.sampler).__name__,
        "counts_before": pd.Series(y_train).value_counts().sort_index().to_json(),
        "counts_after": pd.Series(y_resampled).value_counts().sort_index().to_json(),
        "resolved_model_params": json.dumps(model.params, sort_keys=True),
    }
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
        "n_test": len(y_test),
        "n_positive": int(np.sum(y_test)),
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


def _run_year(test_year: int, data: pd.DataFrame, feature_cols: list[str], logger, *, seed: int = 42, audit_rows: list | None = None):
    if set(feature_cols) & {"company_name", "fyear", "target", "status_label"}:
        raise ValueError("Identifiers, years and target must not be model features")
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
                    seed=seed,
                )
                if audit_rows is not None:
                    audit_rows.append({"test_year": test_year, "boundary": boundary, **model.sampling_audit_})
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

    retrain = _train_model(X_train, y_train, "undersampling", f"rolling_{test_year}_retrain_under", logger, seed=seed)
    if audit_rows is not None:
        audit_rows.append({"test_year": test_year, "boundary": None, **retrain.sampling_audit_})
    method_probas = {
        "Retrain_under": (retrain.predict_proba(X_validation), retrain.predict_proba(X_test)),
        "ROSS_New_under": (pool_validation["New_under"], pool_test["New_under"]),
        "Equal6": (equal6_validation, equal6_test),
        "DAWCE_AUC": (dawce_validation, dawce_test),
        "AdaptiveChoice_AUC": (adaptive_validation, adaptive_test),
    }
    # Independent of the selected ROSS boundary. Each baseline learns its
    # imputer/scaler from its own fixed-window fit data, not all history.
    for window in RECENT_WINDOWS:
        recent = train[train["fyear"] >= train_end - window + 1]
        recent_X, recent_val, recent_test = _fit_transform(recent, validation, test, feature_cols)
        recent_model = _train_model(
            recent_X, recent["target"].to_numpy(), "undersampling",
            f"rolling_{test_year}_recent{window}y_under", logger, seed=seed,
        )
        if audit_rows is not None:
            audit_rows.append({"test_year": test_year, "boundary": None, **recent_model.sampling_audit_})
        method_probas[f"Recent{window}y_under"] = (
            recent_model.predict_proba(recent_val), recent_model.predict_proba(recent_test)
        )

    method_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for method, (validation_proba, test_proba) in method_probas.items():
        row, predictions = _evaluate_method(
            method, test_year, y_validation, validation_proba, y_test, test_proba
        )
        method_rows.append(row)
        for prediction, (source_id, sample) in zip(predictions, test.iterrows(), strict=True):
            prediction["source_row_id"] = int(source_id)
            prediction["company_name"] = sample.get("company_name", "")
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
                    "n_nonzero_pairs": int(np.count_nonzero(difference)),
                    "inference_scope": "exploratory_dependent_annual_pairs",
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
    *,
    test_years=None,
) -> None:
    expected_years = set(test_years if test_years is not None else range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    assert set(by_year["test_year"]) == expected_years
    assert set(by_year["method"]) == set(METHODS)
    assert len(by_year) == len(expected_years) * len(METHODS)
    assert set(selections["test_year"]) == expected_years
    assert (selections["train_end"] == selections["test_year"] - 2).all()
    assert (selections["validation_year"] == selections["test_year"] - 1).all()
    assert not by_year[list(METRICS)].isna().any().any()
    per_method_counts = predictions.groupby("method").size()
    assert per_method_counts.nunique() == 1
    assert not predictions.duplicated(["test_year", "method", "source_row_id"]).any()


def _run_seed(data: pd.DataFrame, feature_cols: list[str], logger, output_dir: Path, seed: int, test_years: list[int]) -> pd.DataFrame:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    audit_rows: list[dict] = []

    method_rows: list[dict] = []
    prediction_rows: list[dict] = []
    boundary_rows: list[dict] = []
    selection_rows: list[dict] = []
    for test_year in test_years:
        logger.info(f"=== Seed {seed}: Walk-forward test year {test_year} ===")
        year_methods, year_predictions, year_boundaries, year_selection = _run_year(
            test_year, data, feature_cols, logger, seed=seed, audit_rows=audit_rows,
        )
        method_rows.extend(year_methods)
        prediction_rows.extend(year_predictions)
        boundary_rows.extend(year_boundaries)
        selection_rows.append(year_selection)

    by_year = pd.DataFrame(method_rows)
    predictions = pd.DataFrame(prediction_rows)
    boundaries = pd.DataFrame(boundary_rows)
    selections = pd.DataFrame(selection_rows)
    _validate_outputs(by_year, predictions, selections, test_years=test_years)
    pooled = _pooled_metrics(predictions)
    outputs = {
        "rolling_by_year.csv": by_year,
        "rolling_predictions.csv": predictions,
        "rolling_boundary_candidates.csv": boundaries,
        "rolling_selection_history.csv": selections,
        "rolling_pooled_summary.csv": pooled,
        "rolling_sampling_audit.csv": pd.DataFrame(audit_rows),
    }
    if len(test_years) >= 2:
        outputs["rolling_wilcoxon_holm.csv"] = _paired_tests(by_year)
    for filename, frame in outputs.items():
        frame["seed"] = seed
        frame["protocol_version"] = "bankruptcy_rolling_v2_exploratory"
        frame.to_csv(output_dir / filename, index=False, float_format="%.12g")
    return pooled


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument("--seeds", type=int, nargs="+", default=None)
    seeds_group.add_argument("--configured-seeds", action="store_true", help="Run all seeds in base_config.yaml")
    parser.add_argument("--test-years", type=int, nargs="+", default=list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR / "runs")
    args = parser.parse_args(argv)
    seeds = get_seeds_from_config(get_config_loader()) if args.configured_seeds else (args.seeds or [42])
    if len(seeds) != len(set(seeds)) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        parser.error("Seeds must be unique integers in [0, 2**32).")
    years = sorted(args.test_years)
    if len(years) != len(set(years)) or any(year < FIRST_TEST_YEAR or year > LAST_TEST_YEAR for year in years):
        parser.error(f"Test years must be unique and within {FIRST_TEST_YEAR}..{LAST_TEST_YEAR}.")
    logger = get_logger("RollingBankruptcyAdaptive", console=True, file=False)
    logger.warning("Exploratory only: bankruptcy label availability is not verified; no labels are reconstructed.")
    X, y = load_bankruptcy_with_year(logger, keep_company=True)
    data = X.copy()
    data["target"] = np.asarray(y)
    feature_cols = [column for column in X.columns if column not in {"fyear", "company_name"}]
    def git_output(*arguments):
        result = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", *arguments],
            cwd=project_root, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return result.stdout.strip() if result.returncode == 0 else None

    source_paths = [
        path for folder in ("src", "experiments", "config")
        for path in (project_root / folder).rglob("*")
        if path.suffix in {".py", ".yaml"}
    ] + [project_root / "requirements.txt"]
    manifest = {
        "protocol_version": "bankruptcy_rolling_v2_exploratory",
        "label_availability": "unverified_no_event_dates",
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv if argv is None else [__file__, *argv]),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_at_start": git_output("status", "--porcelain"),
        "source_hashes": {path.relative_to(project_root).as_posix(): _sha256(path) for path in sorted(source_paths)},
        "data_sha256": _sha256(US_CSV), "seeds": seeds, "test_years": years,
        "feature_columns": feature_cols, "methods": METHODS, "weight_grid": WEIGHT_GRID.tolist(),
        "recent_windows": RECENT_WINDOWS, "min_old_years": MIN_OLD_YEARS,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost")},
        "uncertainty_note": "Seed std is algorithmic variability, not independent temporal replication.",
    }
    run_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        pooled_runs = [_run_seed(data, feature_cols, logger, run_dir / f"seed_{seed}", seed, years) for seed in seeds]
        pooled_all = pd.concat(pooled_runs, ignore_index=True)
        seed_summary = pooled_all.groupby("method")[list(METRICS)].agg(["mean", "std", "count"])
        seed_summary.columns = [f"{metric}_{stat}" for metric, stat in seed_summary.columns]
        seed_summary.to_csv(run_dir / "rolling_seed_summary.csv", float_format="%.12g")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["output_hashes"] = {path.relative_to(run_dir).as_posix(): _sha256(path) for path in sorted(run_dir.rglob("*.csv"))}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved new run without replacing legacy artifacts: {run_dir}")


if __name__ == "__main__":
    main()
