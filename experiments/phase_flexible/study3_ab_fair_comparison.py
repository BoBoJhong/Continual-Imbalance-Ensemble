"""Fair annual comparison of Study 3 route A, route B, and temporal baselines.

For test feature year t, all methods use feature years no later than t-2 for
fitting, t-1 only for model/threshold selection, and t only for final testing.
Route A selects one Old/New boundary by validation average precision and uses
an equal-weight two-model ensemble.  It does not tune weights or inspect Test.

The bankruptcy-event target remains a retrospective annual reconstruction;
exact filing and event availability timestamps are not present in the source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.phase_flexible import rolling_bankruptcy_overlap_ensemble as route_b
from src.evaluation import compute_metrics
from src.utils import get_logger, set_seed


START_YEAR = 1999
FIRST_TEST_YEAR = 2009
LAST_TEST_YEAR = 2018
MIN_OLD_YEARS = 3
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260909
REFERENCE_METHOD = "B_model_FIFO3_equal"
ADDITIONAL_METHODS = (
    "A_validation_boundary_equal",
    "FullHistory",
    "Recent1y",
    "Recent5y",
)
METHODS = route_b.METHODS + ADDITIONAL_METHODS
PROTOCOL_VERSION = "study3_ab_fair_event_target_v1_financial18_exploratory"
OUTPUT_ROOT = project_root / "results" / "phase_flexible" / "study3_ab_fair" / "runs"
BOOTSTRAP_METRICS = ("AP", "AUC", "F1", "Recall", "Precision")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={project_root.as_posix()}", *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _fit_years(
    data: pd.DataFrame,
    feature_cols: list[str],
    years: tuple[int, ...],
    model_id: str,
    logger,
    seed: int,
):
    return route_b._fit_window_model(
        data, feature_cols, years, model_id, logger, seed=seed
    )


def _validation_score(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    ap = float(average_precision_score(y_true, proba))
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else -np.inf
    return ap, auc


def _run_a_and_baselines(
    test_year: int,
    data: pd.DataFrame,
    feature_cols: list[str],
    logger,
    *,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    train_end = test_year - 2
    validation = data[data.fyear == test_year - 1].copy()
    test = data[data.fyear == test_year].copy()
    if validation.empty or test.empty:
        raise ValueError(f"Missing validation/test rows for feature year {test_year}")
    y_validation = validation.target.to_numpy()

    candidates: list[dict] = []
    candidate_models: dict[int, tuple] = {}
    audits: list[dict] = []
    for boundary in range(START_YEAR + MIN_OLD_YEARS, train_end + 1):
        old_years = tuple(range(START_YEAR, boundary))
        new_years = tuple(range(boundary, train_end + 1))
        old_model = _fit_years(
            data, feature_cols, old_years,
            f"seed{seed}_A{test_year}_old_to_{boundary - 1}", logger, seed,
        )
        new_model = _fit_years(
            data, feature_cols, new_years,
            f"seed{seed}_A{test_year}_new_{boundary}_to_{train_end}", logger, seed,
        )
        validation_proba = 0.5 * (
            old_model.predict_proba(validation, feature_cols)
            + new_model.predict_proba(validation, feature_cols)
        )
        validation_ap, validation_auc = _validation_score(y_validation, validation_proba)
        candidates.append({
            "test_feature_year": test_year,
            "validation_feature_year": test_year - 1,
            "boundary_year": boundary,
            "old_years": f"{START_YEAR}-{boundary - 1}",
            "new_years": f"{boundary}-{train_end}",
            "validation_AP": validation_ap,
            "validation_AUC": validation_auc,
            "test_used_for_selection": False,
        })
        candidate_models[boundary] = (old_model, new_model)
        audits.extend([
            {"test_feature_year": test_year, "route": "A_old_candidate", **old_model.audit},
            {"test_feature_year": test_year, "route": "A_new_candidate", **new_model.audit},
        ])

    selected = max(
        candidates,
        key=lambda row: (
            row["validation_AP"], row["validation_AUC"], row["boundary_year"]
        ),
    )
    selected_boundary = int(selected["boundary_year"])
    old_model, new_model = candidate_models[selected_boundary]
    a_validation = 0.5 * (
        old_model.predict_proba(validation, feature_cols)
        + new_model.predict_proba(validation, feature_cols)
    )
    a_test = 0.5 * (
        old_model.predict_proba(test, feature_cols)
        + new_model.predict_proba(test, feature_cols)
    )

    probabilities = {
        "A_validation_boundary_equal": (a_validation, a_test),
    }
    for method, years in {
        "FullHistory": tuple(range(START_YEAR, train_end + 1)),
        "Recent1y": (train_end,),
        "Recent5y": tuple(range(train_end - 4, train_end + 1)),
    }.items():
        model = _fit_years(
            data, feature_cols, years, f"seed{seed}_{method}_{test_year}", logger, seed
        )
        audits.append({"test_feature_year": test_year, "route": method, **model.audit})
        probabilities[method] = (
            model.predict_proba(validation, feature_cols),
            model.predict_proba(test, feature_cols),
        )

    rows: list[dict] = []
    predictions: list[dict] = []
    for method, (validation_proba, test_proba) in probabilities.items():
        row, method_predictions = route_b._evaluate(
            method, test_year, validation, test, validation_proba, test_proba
        )
        rows.append(row)
        predictions.extend(method_predictions)

    selection = {
        "test_feature_year": test_year,
        "validation_feature_year": test_year - 1,
        "latest_train_feature_year": train_end,
        "selected_boundary": selected_boundary,
        "selected_old_years": selected["old_years"],
        "selected_new_years": selected["new_years"],
        "selection_metric": "validation_AP_then_AUC",
        "ensemble_weighting": "equal_0.5_0.5",
        "n_candidates": len(candidates),
        "test_used_for_selection": False,
    }
    return rows, predictions, candidates, selection, audits


def _metric_values(
    y_true: np.ndarray,
    proba: np.ndarray,
    pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    return {
        "AP": float(average_precision_score(y_true, proba, sample_weight=sample_weight)),
        "AUC": float(roc_auc_score(y_true, proba, sample_weight=sample_weight)),
        "F1": float(f1_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
    }


def paired_company_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    reference_method: str = REFERENCE_METHOD,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Paired percentile bootstrap with companies as the resampling unit."""
    keys = ["test_feature_year", "source_row_id", "company_name"]
    ordered: dict[str, pd.DataFrame] = {}
    for method, frame in predictions.groupby("method", sort=False):
        ordered[method] = frame.sort_values(keys).reset_index(drop=True)
    reference = ordered[reference_method]
    for method, frame in ordered.items():
        if not reference[keys].equals(frame[keys]) or not np.array_equal(
            reference.y_true.to_numpy(), frame.y_true.to_numpy()
        ):
            raise ValueError(f"Predictions are not paired for {method}")

    companies, company_codes = np.unique(
        reference.company_name.astype(str).to_numpy(), return_inverse=True
    )
    if len(companies) < 2:
        raise ValueError("At least two company clusters are required")
    arrays = {
        method: (
            frame.y_true.to_numpy(),
            frame.y_proba.to_numpy(),
            frame.y_pred_f1.to_numpy(),
        )
        for method, frame in ordered.items()
    }
    observed = {
        method: _metric_values(*values) for method, values in arrays.items()
    }
    rng = np.random.default_rng(seed)
    differences = {
        (method, metric): np.empty(replicates, dtype=float)
        for method in ordered if method != reference_method
        for metric in BOOTSTRAP_METRICS
    }
    for replicate in range(replicates):
        sampled = rng.integers(0, len(companies), size=len(companies))
        cluster_counts = np.bincount(sampled, minlength=len(companies))
        row_weights = cluster_counts[company_codes]
        reference_metrics = _metric_values(*arrays[reference_method], row_weights)
        for method in ordered:
            if method == reference_method:
                continue
            method_metrics = _metric_values(*arrays[method], row_weights)
            for metric in BOOTSTRAP_METRICS:
                differences[(method, metric)][replicate] = (
                    reference_metrics[metric] - method_metrics[metric]
                )

    rows = []
    for method in ordered:
        if method == reference_method:
            continue
        for metric in BOOTSTRAP_METRICS:
            values = differences[(method, metric)]
            rows.append({
                "reference_method": reference_method,
                "comparison_method": method,
                "metric": metric,
                "observed_reference_minus_comparison": (
                    observed[reference_method][metric] - observed[method][metric]
                ),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_probability_difference_gt_0": float(np.mean(values > 0)),
                "n_company_clusters": len(companies),
                "n_rows": len(reference),
                "replicates": replicates,
                "bootstrap_seed": seed,
                "inference_scope": "paired_company_cluster_percentile_bootstrap",
            })
    return pd.DataFrame(rows)


def _run_seed(
    data: pd.DataFrame,
    feature_cols: list[str],
    logger,
    output_dir: Path,
    seed: int,
    test_years: list[int],
    bootstrap_replicates: int,
) -> pd.DataFrame:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    predictions: list[dict] = []
    audits: list[dict] = []
    protocols: list[dict] = []
    candidates: list[dict] = []
    selections: list[dict] = []
    model_pool: dict[int, route_b.FittedWindowModel] = {}
    for test_year in test_years:
        logger.info(f"=== Fair A/B seed={seed}, test feature year={test_year} ===")
        b_rows, b_predictions, b_audits, protocol = route_b._run_year(
            test_year, data, feature_cols, logger, seed=seed, model_pool=model_pool
        )
        a_rows, a_predictions, a_candidates, selection, a_audits = _run_a_and_baselines(
            test_year, data, feature_cols, logger, seed=seed
        )
        rows.extend(b_rows + a_rows)
        predictions.extend(b_predictions + a_predictions)
        audits.extend(b_audits + a_audits)
        candidates.extend(a_candidates)
        selections.append(selection)
        protocol["a_selected_boundary"] = selection["selected_boundary"]
        protocol["a_selection_metric"] = selection["selection_metric"]
        protocols.append(protocol)

    by_year = pd.DataFrame(rows)
    prediction_frame = pd.DataFrame(predictions)
    expected = set(test_years)
    if set(by_year.test_feature_year) != expected or set(by_year.method) != set(METHODS):
        raise AssertionError("Unexpected year or method coverage")
    if len(by_year) != len(expected) * len(METHODS):
        raise AssertionError("Incomplete method-by-year result matrix")
    if prediction_frame.duplicated(
        ["test_feature_year", "method", "source_row_id"]
    ).any():
        raise AssertionError("Duplicate prediction keys")
    if any(row["test_used_for_selection"] for row in candidates + selections):
        raise AssertionError("Test must not be used for A-route selection")

    pooled = route_b._pooled_summary(prediction_frame)
    outputs = {
        "fair_by_year.csv": by_year,
        "fair_predictions.csv": prediction_frame,
        "fair_pooled_summary.csv": pooled,
        "fair_sampling_audit.csv": pd.DataFrame(audits),
        "fair_protocol.csv": pd.DataFrame(protocols),
        "a_boundary_candidates.csv": pd.DataFrame(candidates),
        "a_selection_history.csv": pd.DataFrame(selections),
    }
    if bootstrap_replicates:
        outputs["paired_company_cluster_bootstrap.csv"] = paired_company_cluster_bootstrap(
            prediction_frame, replicates=bootstrap_replicates
        )
    for filename, frame in outputs.items():
        frame["seed"] = seed
        frame["protocol_version"] = PROTOCOL_VERSION
        frame.to_csv(output_dir / filename, index=False, float_format="%.12g")
    return pooled.assign(seed=seed)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-years", type=int, nargs="+",
        default=list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    years = sorted(args.test_years)
    if len(years) != len(set(years)) or any(
        year < FIRST_TEST_YEAR or year > LAST_TEST_YEAR for year in years
    ):
        parser.error(f"Test years must be unique and within {FIRST_TEST_YEAR}..{LAST_TEST_YEAR}")
    if args.seed < 0 or args.seed >= 2**32 or args.bootstrap_replicates < 0:
        parser.error("Seed must be uint32 and bootstrap replicates must be non-negative")

    logger = get_logger("Study3ABFair", console=True, file=False)
    data, feature_cols, label_audit = route_b.load_event_data()
    run_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    label_audit.to_csv(run_dir / "label_audit.csv", index=False)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "running",
        "evidence_status": "exploratory_reconstructed_event_target",
        "target_definition": "failed company final observed feature year; event in fyear+1",
        "label_availability": "annual_inference_supported_exact_dates_unavailable",
        "a_selection": "validation_AP_then_AUC_equal_weight_no_test_selection",
        "common_sampling": route_b.SAMPLING,
        "common_features": feature_cols,
        "methods": METHODS,
        "seed": args.seed,
        "test_years": years,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "company",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv if argv is None else [__file__, *argv]),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_at_start": _git_output("status", "--porcelain"),
        "data_sha256": _sha256(route_b.US_CSV),
        "source_hashes": {
            Path(__file__).relative_to(project_root).as_posix(): _sha256(Path(__file__)),
            Path(route_b.__file__).relative_to(project_root).as_posix(): _sha256(Path(route_b.__file__)),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost")
        },
        "interpretation_limit": (
            "Company bootstrap captures clustered row-sampling uncertainty, not temporal "
            "process uncertainty or live label availability."
        ),
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        pooled = _run_seed(
            data, feature_cols, logger, run_dir / f"seed_{args.seed}", args.seed,
            years, args.bootstrap_replicates,
        )
        pooled.to_csv(run_dir / "fair_seed_summary.csv", index=False, float_format="%.12g")
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
    logger.info(f"Saved fair A/B run: {run_dir}")


if __name__ == "__main__":
    main()
