"""Study 3 route B: annual overlapping-window continual ensembles.

For a test feature year ``t`` (predicting bankruptcy in ``t + 1``):

* training data end at ``t - 2``;
* feature year ``t - 1`` is validation-only;
* feature year ``t`` is test-only.

The experiment compares a recent three-year model, B-data (two overlapping
years plus one incoming year), and B-model (three frozen three-year models in
an equal-weight FIFO pool).  Raw data and previous result files are never
modified; every run is written to a new timestamped directory.

The event target is a documented reconstruction: a failed company's final
observed fiscal year is positive.  It agrees with the source paper's annual
event counts, but exact filing/report-availability dates remain unavailable.
Consequently, this protocol is retrospective annual evidence, not a validated
real-time deployment simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments._shared.common_bankruptcy import US_CSV
from src.data import ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_config_loader, get_logger, get_seeds_from_config, set_seed


FIRST_TEST_YEAR = 2009
LAST_TEST_YEAR = 2018
WINDOW_YEARS = 3
POOL_SIZE = 3
VALIDATION_FPR_BUDGET = 0.05
SAMPLING = "undersampling"  # Configured as TomekLinks; retained for fair-A compatibility.
DEFAULT_IMBALANCE_METHOD = "tomek"
IMBALANCE_METHODS = ("none", "tomek", "scale_pos_weight")
METHODS = ("Recent3y", "B_data_equal", "B_model_FIFO3_equal")
METRICS = (
    "AUC", "PR_AUC", "F1", "G_Mean", "Recall", "Precision",
    "Balanced_Accuracy", "Type1_Error", "Type2_Error",
)
PROTOCOL_VERSION = "study3b_overlap_event_target_v1_financial18_exploratory"
OUTPUT_ROOT = project_root / "results" / "phase_flexible" / "study3b_overlap" / "runs"


def build_event_target(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the reconstructed next-year event target.

    ``status_label`` is company-level in the distributed raw CSV.  Repeating it
    on every historical row would turn pre-event history into positive labels.
    The reconstructed target therefore marks only the final observed feature
    year of a failed company.  The source columns are retained for audit only.
    """
    required = {"company_name", "fyear", "status_label"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")
    if raw[list(required)].isna().any().any():
        raise ValueError("Company, fiscal year and status label must be complete.")
    if not set(raw["status_label"].unique()) <= {"alive", "failed"}:
        raise ValueError("status_label must contain only alive/failed.")
    if raw.duplicated(["company_name", "fyear"]).any():
        raise ValueError("Duplicate company-year rows make the event target ambiguous.")
    if raw.groupby("company_name")["status_label"].nunique().gt(1).any():
        raise ValueError("A company has inconsistent status_label values across years.")

    data = raw.copy()
    data["fyear"] = data["fyear"].astype(int)
    last_year = data.groupby("company_name")["fyear"].transform("max")
    data["target"] = (
        data["status_label"].eq("failed") & data["fyear"].eq(last_year)
    ).astype(int)
    data["event_year"] = data["fyear"] + 1
    return data


def load_event_data() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    raw = pd.read_csv(US_CSV)
    data = build_event_target(raw)
    # Match the source paper's 18 financial variables. Division/MajorGroup are
    # industry categories, not continuous financial measures; treating their
    # integer codes as scaled continuous features would impose a false order.
    feature_cols = [f"X{index}" for index in range(1, 19)]
    if not set(feature_cols) <= set(data.columns) or not all(
        pd.api.types.is_numeric_dtype(data[column]) for column in feature_cols
    ):
        raise ValueError("Expected numeric financial features X1..X18.")

    annual = (
        data.groupby(["fyear", "event_year"], as_index=False)
        .agg(n_rows=("target", "size"), n_events=("target", "sum"))
    )
    annual["event_rate"] = annual["n_events"] / annual["n_rows"]
    # Full-source invariants established by docs/notebooks/bankruptcy_label_time_audit.ipynb.
    if len(data) == 78_682:
        if int(data["target"].sum()) != 609 or (int(data.fyear.min()), int(data.fyear.max())) != (1999, 2018):
            raise ValueError("Raw bankruptcy data no longer matches the audited event-target contract.")
    return data, feature_cols, annual


def model_window_ends(test_year: int, pool_size: int = POOL_SIZE) -> tuple[int, ...]:
    """FIFO model end-years available before validation year ``t - 1``."""
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    latest_end_year = test_year - 2
    return tuple(range(latest_end_year - pool_size + 1, latest_end_year + 1))


def window_years(end_year: int, width: int = WINDOW_YEARS) -> tuple[int, ...]:
    return tuple(range(end_year - width + 1, end_year + 1))


def data_route_years(test_year: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return B-data's overlap (Old) and incoming (New) training years."""
    return (test_year - 4, test_year - 3), (test_year - 2,)


@dataclass
class FittedWindowModel:
    model_id: str
    train_years: tuple[int, ...]
    imputer: SimpleImputer
    scaler: StandardScaler
    model: XGBoostWrapper
    audit: dict

    def predict_proba(self, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        transformed = self.scaler.transform(self.imputer.transform(frame[feature_cols]))
        return self.model.predict_proba(pd.DataFrame(transformed, columns=feature_cols, index=frame.index))


def _fit_window_model(
    data: pd.DataFrame,
    feature_cols: list[str],
    years: tuple[int, ...],
    model_id: str,
    logger,
    *,
    seed: int,
    imbalance_method: str = DEFAULT_IMBALANCE_METHOD,
) -> FittedWindowModel:
    train = data[data["fyear"].isin(years)].copy()
    observed_years = tuple(sorted(train["fyear"].unique().tolist()))
    if observed_years != years:
        raise ValueError(f"{model_id}: expected years {years}, observed {observed_years}")
    y_train = train["target"].to_numpy()
    if len(np.unique(y_train)) != 2:
        raise ValueError(f"{model_id}: training window must contain both classes")

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(train[feature_cols])
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed), columns=feature_cols, index=train.index
    )
    if imbalance_method not in IMBALANCE_METHODS:
        raise ValueError(
            f"Unknown imbalance method {imbalance_method!r}; expected one of {IMBALANCE_METHODS}"
        )

    sampler_name = "None"
    scale_pos_weight = None
    if imbalance_method == "tomek":
        sampler = ImbalanceSampler(random_state=seed)
        try:
            X_resampled, y_resampled = sampler.apply_sampling(
                X_scaled, y_train, strategy=SAMPLING
            )
        except ValueError as error:
            raise ValueError(f"Sampling failed for {model_id}; no silent fallback") from error
        sampler_name = type(sampler.sampler).__name__
    else:
        X_resampled, y_resampled = X_scaled, y_train
        if imbalance_method == "scale_pos_weight":
            n_positive = int(y_train.sum())
            n_negative = int(len(y_train) - n_positive)
            scale_pos_weight = n_negative / n_positive

    model_params = {"random_state": seed, "seed": seed}
    if scale_pos_weight is not None:
        model_params["scale_pos_weight"] = scale_pos_weight
    model = XGBoostWrapper(name=model_id, use_imbalance=False, **model_params)
    model.fit(X_resampled, np.asarray(y_resampled))
    audit = {
        "model_id": model_id,
        "train_start": years[0],
        "train_end": years[-1],
        "train_years": ",".join(map(str, years)),
        "n_train": len(train),
        "n_positive_before": int(y_train.sum()),
        "n_negative_before": int(len(y_train) - y_train.sum()),
        "n_positive_after": int(np.sum(y_resampled)),
        "n_negative_after": int(len(y_resampled) - np.sum(y_resampled)),
        "imbalance_method": imbalance_method,
        "sampler": sampler_name,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "resolved_model_params": json.dumps(model.params, sort_keys=True),
    }
    return FittedWindowModel(model_id, years, imputer, scaler, model, audit)


def _select_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        score = f1_score(y_true, proba >= threshold, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def _select_threshold_at_fpr(
    y_true: np.ndarray, proba: np.ndarray, budget: float = VALIDATION_FPR_BUDGET
) -> float:
    """Select on validation only: greatest recall with FPR <= budget."""
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    feasible = np.flatnonzero(fpr <= budget + 1e-12)
    if len(feasible) == 0:
        return 1.0
    best = max(feasible, key=lambda index: (tpr[index], -fpr[index], thresholds[index]))
    threshold = float(thresholds[best])
    return 1.0 if not np.isfinite(threshold) else float(np.clip(threshold, 0.0, 1.0))


def _operating_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Recall_at_5pctFPR": float(recall_score(y_true, pred, zero_division=0)),
        "Precision_at_5pctFPR": float(precision_score(y_true, pred, zero_division=0)),
        "Realized_FPR_at_5pctFPR": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "Alerts_at_5pctFPR": int(tp + fp),
    }


def _evaluate(
    method: str,
    test_year: int,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    validation_proba: np.ndarray,
    test_proba: np.ndarray,
) -> tuple[dict, list[dict]]:
    y_validation = validation["target"].to_numpy()
    y_test = test["target"].to_numpy()
    f1_threshold = _select_f1_threshold(y_validation, validation_proba)
    budget_threshold = _select_threshold_at_fpr(y_validation, validation_proba)
    validation_metrics = compute_metrics(y_validation, validation_proba, threshold=f1_threshold)
    test_metrics = compute_metrics(y_test, test_proba, threshold=f1_threshold)
    validation_budget = _operating_metrics(y_validation, validation_proba, budget_threshold)
    test_budget = _operating_metrics(y_test, test_proba, budget_threshold)
    row = {
        "test_feature_year": test_year,
        "test_event_year": test_year + 1,
        "validation_feature_year": test_year - 1,
        "validation_event_year": test_year,
        "method": method,
        "n_validation": len(validation),
        "n_validation_positive": int(y_validation.sum()),
        "n_test": len(test),
        "n_test_positive": int(y_test.sum()),
        "prevalence": float(np.mean(y_test)),
        "f1_threshold_from_validation": f1_threshold,
        "fpr5_threshold_from_validation": budget_threshold,
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **{f"validation_{key}": value for key, value in validation_budget.items()},
        **test_metrics,
        **test_budget,
    }
    predictions = []
    for source_row_id, (_, sample), proba in zip(test.index, test.iterrows(), test_proba, strict=True):
        predictions.append({
            "test_feature_year": test_year,
            "test_event_year": test_year + 1,
            "method": method,
            "source_row_id": int(source_row_id),
            "company_name": sample["company_name"],
            "y_true": int(sample["target"]),
            "y_proba": float(proba),
            "y_pred_f1": int(proba >= f1_threshold),
            "y_pred_fpr5": int(proba >= budget_threshold),
            "f1_threshold": f1_threshold,
            "fpr5_threshold": budget_threshold,
        })
    return row, predictions


def _run_year(
    test_year: int,
    data: pd.DataFrame,
    feature_cols: list[str],
    logger,
    *,
    seed: int,
    model_pool: dict[int, FittedWindowModel] | None = None,
    imbalance_method: str = DEFAULT_IMBALANCE_METHOD,
    pool_size: int = POOL_SIZE,
    window_width: int = WINDOW_YEARS,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    validation = data[data["fyear"] == test_year - 1].copy()
    test = data[data["fyear"] == test_year].copy()
    if validation.empty or test.empty:
        raise ValueError(f"Missing validation/test rows for feature year {test_year}")

    audits: list[dict] = []
    pool = model_pool if model_pool is not None else {}
    required_end_years = model_window_ends(test_year, pool_size)
    removed_end_years = tuple(sorted(set(pool) - set(required_end_years)))
    for end_year in removed_end_years:
        del pool[end_year]
    added_end_years = []
    for end_year in required_end_years:
        if end_year not in pool:
            pool[end_year] = _fit_window_model(
                data, feature_cols, window_years(end_year, window_width),
                f"seed{seed}_M{end_year}", logger, seed=seed,
                imbalance_method=imbalance_method,
            )
            added_end_years.append(end_year)
    if len(pool) != pool_size:
        raise AssertionError(f"FIFO pool must contain {pool_size} models, found {len(pool)}")
    fifo_models = [pool[end_year] for end_year in required_end_years]
    for fitted in fifo_models:
        audits.append({
            "test_feature_year": test_year,
            "route": "B_model",
            "pool_action": "added" if fitted.train_years[-1] in added_end_years else "retained",
            **fitted.audit,
        })

    # The recent-window comparator is deliberately the newest member of the FIFO pool. Reusing the
    # exact fitted model makes the comparison isolate ensemble composition.
    recent = fifo_models[-1]
    recent_validation = recent.predict_proba(validation, feature_cols)
    recent_test = recent.predict_proba(test, feature_cols)
    bmodel_validation = np.mean(
        [model.predict_proba(validation, feature_cols) for model in fifo_models], axis=0
    )
    bmodel_test = np.mean(
        [model.predict_proba(test, feature_cols) for model in fifo_models], axis=0
    )

    old_years, new_years = data_route_years(test_year)
    old_model = _fit_window_model(
        data, feature_cols, old_years, f"seed{seed}_Bdata_old_{test_year}", logger,
        seed=seed, imbalance_method=imbalance_method,
    )
    new_model = _fit_window_model(
        data, feature_cols, new_years, f"seed{seed}_Bdata_new_{test_year}", logger,
        seed=seed, imbalance_method=imbalance_method,
    )
    audits.extend([
        {"test_feature_year": test_year, "route": "B_data_old", **old_model.audit},
        {"test_feature_year": test_year, "route": "B_data_new", **new_model.audit},
    ])
    bdata_validation = 0.5 * (
        old_model.predict_proba(validation, feature_cols)
        + new_model.predict_proba(validation, feature_cols)
    )
    bdata_test = 0.5 * (
        old_model.predict_proba(test, feature_cols)
        + new_model.predict_proba(test, feature_cols)
    )

    bmodel_method = f"B_model_FIFO{pool_size}_equal"
    recent_method = f"Recent{window_width}y"
    probabilities = {
        recent_method: (recent_validation, recent_test),
        "B_data_equal": (bdata_validation, bdata_test),
        bmodel_method: (bmodel_validation, bmodel_test),
    }
    rows: list[dict] = []
    predictions: list[dict] = []
    for method, (validation_proba, test_proba) in probabilities.items():
        row, method_predictions = _evaluate(
            method, test_year, validation, test, validation_proba, test_proba
        )
        rows.append(row)
        predictions.extend(method_predictions)

    protocol = {
        "test_feature_year": test_year,
        "test_event_year": test_year + 1,
        "validation_feature_year": test_year - 1,
        "latest_train_feature_year": test_year - 2,
        "recent_years": ",".join(map(str, recent.train_years)),
        "window_years": window_width,
        "b_model_end_years": ",".join(map(str, required_end_years)),
        "b_model_added_end_years": ",".join(map(str, added_end_years)),
        "b_model_removed_end_years": ",".join(map(str, removed_end_years)),
        "b_model_union_years": (
            f"{min(year for fitted in fifo_models for year in fitted.train_years)}-"
            f"{max(year for fitted in fifo_models for year in fitted.train_years)}"
        ),
        "b_data_old_years": ",".join(map(str, old_years)),
        "b_data_new_years": str(new_years[0]),
        "pool_size": pool_size,
        "ensemble_weights": "equal",
        "imbalance_method": imbalance_method,
        "test_used_for_selection": False,
    }
    return rows, predictions, audits, protocol


def _pooled_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in predictions.groupby("method", sort=False):
        metrics = compute_metrics(group.y_true, group.y_proba, y_pred=group.y_pred_f1)
        budget = _operating_metrics(group.y_true.to_numpy(), group.y_proba.to_numpy(), 0.5)
        # Replace the fixed-threshold values with the already selected annual decisions.
        pred = group.y_pred_fpr5.to_numpy()
        tn, fp, fn, tp = confusion_matrix(group.y_true, pred, labels=[0, 1]).ravel()
        budget.update({
            "Recall_at_5pctFPR": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "Precision_at_5pctFPR": float(tp / (tp + fp)) if tp + fp else 0.0,
            "Realized_FPR_at_5pctFPR": float(fp / (fp + tn)) if fp + tn else float("nan"),
            "Alerts_at_5pctFPR": int(tp + fp),
        })
        rows.append({"method": method, "n_predictions": len(group), **metrics, **budget})
    return pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)


def _validate_outputs(
    by_year: pd.DataFrame,
    predictions: pd.DataFrame,
    protocol: pd.DataFrame,
    test_years: list[int],
    methods: tuple[str, ...] = METHODS,
) -> None:
    expected = set(test_years)
    assert set(by_year.test_feature_year) == expected
    assert set(by_year.method) == set(methods)
    assert len(by_year) == len(expected) * len(methods)
    assert set(protocol.test_feature_year) == expected
    assert (protocol.latest_train_feature_year == protocol.test_feature_year - 2).all()
    assert (protocol.validation_feature_year == protocol.test_feature_year - 1).all()
    assert not protocol.test_used_for_selection.any()
    assert not predictions.duplicated(["test_feature_year", "method", "source_row_id"]).any()
    assert set(predictions.method) == set(methods)


def _run_seed(
    data: pd.DataFrame,
    feature_cols: list[str],
    logger,
    output_dir: Path,
    seed: int,
    test_years: list[int],
    imbalance_method: str = DEFAULT_IMBALANCE_METHOD,
    pool_size: int = POOL_SIZE,
    window_width: int = WINDOW_YEARS,
) -> pd.DataFrame:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    predictions: list[dict] = []
    audits: list[dict] = []
    protocols: list[dict] = []
    model_pool: dict[int, FittedWindowModel] = {}
    for test_year in test_years:
        logger.info(f"=== Study 3B seed={seed}, test feature year={test_year} ===")
        year_rows, year_predictions, year_audits, protocol = _run_year(
            test_year, data, feature_cols, logger, seed=seed, model_pool=model_pool,
            imbalance_method=imbalance_method,
            pool_size=pool_size,
            window_width=window_width,
        )
        rows.extend(year_rows)
        predictions.extend(year_predictions)
        audits.extend(year_audits)
        protocols.append(protocol)

    by_year = pd.DataFrame(rows)
    prediction_frame = pd.DataFrame(predictions)
    protocol_frame = pd.DataFrame(protocols)
    methods = (f"Recent{window_width}y", "B_data_equal", f"B_model_FIFO{pool_size}_equal")
    _validate_outputs(by_year, prediction_frame, protocol_frame, test_years, methods)
    pooled = _pooled_summary(prediction_frame)
    outputs = {
        "study3b_by_year.csv": by_year,
        "study3b_predictions.csv": prediction_frame,
        "study3b_window_audit.csv": pd.DataFrame(audits),
        "study3b_protocol.csv": protocol_frame,
        "study3b_pooled_summary.csv": pooled,
    }
    for filename, frame in outputs.items():
        frame["seed"] = seed
        frame["protocol_version"] = PROTOCOL_VERSION
        frame["imbalance_method"] = imbalance_method
        frame.to_csv(output_dir / filename, index=False, float_format="%.12g")
    return pooled.assign(seed=seed)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={project_root.as_posix()}", *arguments],
        cwd=project_root, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument("--seeds", type=int, nargs="+", default=None)
    seeds_group.add_argument("--configured-seeds", action="store_true")
    parser.add_argument(
        "--test-years", type=int, nargs="+",
        default=list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)),
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--imbalance-method",
        choices=IMBALANCE_METHODS,
        default=DEFAULT_IMBALANCE_METHOD,
        help="Training-only imbalance treatment; validation and test remain untouched.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=POOL_SIZE,
        help="Maximum number of frozen overlapping-window models in the FIFO ensemble.",
    )
    parser.add_argument(
        "--window-years",
        type=int,
        default=WINDOW_YEARS,
        help="Number of feature years used to fit each frozen window model.",
    )
    args = parser.parse_args(argv)
    seeds = get_seeds_from_config(get_config_loader()) if args.configured_seeds else (args.seeds or [42])
    if len(seeds) != len(set(seeds)) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        parser.error("Seeds must be unique integers in [0, 2**32).")
    if args.pool_size < 1:
        parser.error("--pool-size must be positive")
    if args.window_years < 1:
        parser.error("--window-years must be positive")
    test_years = sorted(args.test_years)
    if len(test_years) != len(set(test_years)) or any(
        year < FIRST_TEST_YEAR or year > LAST_TEST_YEAR for year in test_years
    ):
        parser.error(f"Test years must be unique and within {FIRST_TEST_YEAR}..{LAST_TEST_YEAR}.")

    logger = get_logger("Study3BOverlap", console=True, file=False)
    data, feature_cols, label_audit = load_event_data()
    logger.warning(
        "Exploratory reconstructed event target: annual timing supported; exact filing/report dates unavailable."
    )
    run_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    label_audit.to_csv(run_dir / "study3b_label_audit.csv", index=False)

    source_paths = [Path(__file__), project_root / "config" / "model_config.yaml", project_root / "config" / "sampling_config.yaml"]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_status": "exploratory_reconstructed_event_target",
        "target_definition": "failed company and final observed feature year; predicts event in fyear+1",
        "label_availability": "annual_inference_supported_exact_dates_unavailable",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv if argv is None else [__file__, *argv]),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_at_start": _git_output("status", "--porcelain"),
        "source_hashes": {path.relative_to(project_root).as_posix(): _sha256(path) for path in source_paths},
        "data_sha256": _sha256(US_CSV),
        "n_rows": len(data),
        "n_events": int(data.target.sum()),
        "feature_columns": feature_cols,
        "excluded_metadata_features": ["Division", "MajorGroup"],
        "methods": (f"Recent{args.window_years}y", "B_data_equal", f"B_model_FIFO{args.pool_size}_equal"),
        "sampling": SAMPLING if args.imbalance_method == "tomek" else "none",
        "imbalance_method": args.imbalance_method,
        "window_years": args.window_years,
        "b_data_route_years": "fixed_old2_plus_new1",
        "pool_size": args.pool_size,
        "ensemble_weighting": "equal_only_no_weight_tuning",
        "validation_fpr_budget": VALIDATION_FPR_BUDGET,
        "seeds": seeds,
        "test_years": test_years,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost")
        },
        "uncertainty_note": "Seed variation is algorithmic, not independent temporal replication.",
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        pooled_runs = [
            _run_seed(
                data, feature_cols, logger, run_dir / f"seed_{seed}", seed, test_years,
                imbalance_method=args.imbalance_method,
                pool_size=args.pool_size,
                window_width=args.window_years,
            )
            for seed in seeds
        ]
        pooled_all = pd.concat(pooled_runs, ignore_index=True)
        summary_metrics = [*METRICS, "Recall_at_5pctFPR", "Precision_at_5pctFPR", "Realized_FPR_at_5pctFPR"]
        seed_summary = pooled_all.groupby("method")[summary_metrics].agg(["mean", "std", "count"])
        seed_summary.columns = [f"{metric}_{stat}" for metric, stat in seed_summary.columns]
        seed_summary.to_csv(run_dir / "study3b_seed_summary.csv", float_format="%.12g")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["output_hashes"] = {
            path.relative_to(run_dir).as_posix(): _sha256(path)
            for path in sorted(run_dir.rglob("*.csv"))
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved versioned Study 3B run: {run_dir}")


if __name__ == "__main__":
    main()
