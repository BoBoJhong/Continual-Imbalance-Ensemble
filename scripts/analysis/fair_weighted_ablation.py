"""
Fair ablation of single-model and weighted Old/New ensembles.

All methods within a split share the same six trained base models, validation
set, preprocessing, feature selection, and test set. This isolates the effect
of model/weight selection from training-protocol differences.

Outputs:
    results/phase5_weighted/bk_fair_ablation_by_split.csv
    results/phase5_weighted/bk_fair_ablation_summary.csv
    results/phase5_weighted/bk_fair_ablation_vs_new_under_wilcoxon.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments._shared.common_bankruptcy import YEAR_SPLITS
from experiments.phase4_drift.bankruptcy_drift_stream import (
    POOL_SAMPLING,
    TRAIN_END_YEAR,
    _preprocess,
    _select_threshold,
    load_bankruptcy_with_year,
)
from experiments.phase4_drift.bankruptcy_ross_fs_static_ensemble import _apply_old_fit_fs
from experiments.phase4_drift.bankruptcy_ross_static_ensemble import _old_new_split, _test_split
from experiments.phase5_weighted.bankruptcy_ross_weight_sweep import WEIGHT_GRID
from src.data import ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_logger, set_seed


OUTPUT_DIR = project_root / "results" / "phase5_weighted"
METRICS = ("AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error")
MODEL_LABELS = tuple(
    [f"Old_{sampling.replace('sampling', '')}" for sampling in POOL_SAMPLING]
    + [f"New_{sampling.replace('sampling', '')}" for sampling in POOL_SAMPLING]
)


def _train_pool_probas(
    X_old_raw: pd.DataFrame,
    y_old: np.ndarray,
    X_new_raw: pd.DataFrame,
    y_new: np.ndarray,
    X_test_raw: pd.DataFrame,
    tag: str,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Train the shared six-model pool and return validation/test probabilities."""
    n_old_val = max(1, int(len(X_old_raw) * 0.2))
    n_new_val = max(1, int(len(X_new_raw) * 0.2))
    X_old_fit_raw, X_old_val_raw = X_old_raw.iloc[:-n_old_val], X_old_raw.iloc[-n_old_val:]
    X_new_fit_raw, X_new_val_raw = X_new_raw.iloc[:-n_new_val], X_new_raw.iloc[-n_new_val:]
    X_old_fit, X_new_fit, X_old_val, X_new_val, X_test_s, _ = _preprocess(
        X_old_fit_raw,
        X_new_fit_raw,
        X_old_val_raw,
        X_new_val_raw,
        X_test_raw,
    )
    y_old_fit = y_old[:-n_old_val]
    y_new_fit = y_new[:-n_new_val]
    X_val = pd.concat(
        [X_old_val, X_new_val],
        ignore_index=True,
    )
    y_val = np.concatenate([y_old[-n_old_val:], y_new[-n_new_val:]])

    sampler = ImbalanceSampler()
    val_probas: list[np.ndarray] = []
    test_probas: list[np.ndarray] = []
    for period, X_fit, y_fit in (
        ("old", X_old_fit, y_old_fit),
        ("new", X_new_fit, y_new_fit),
    ):
        for sampling in POOL_SAMPLING:
            X_resampled, y_resampled = sampler.apply_sampling(
                X_fit,
                y_fit,
                strategy=sampling,
            )
            model = XGBoostWrapper(
                name=f"{tag}_{period}_{sampling}",
                use_imbalance=False,
            )
            model.fit(X_resampled, y_resampled)
            val_probas.append(model.predict_proba(X_val))
            test_probas.append(model.predict_proba(X_test_s))

    return y_val, val_probas, test_probas


def _evaluate_candidate(
    *,
    split_label: str,
    drift_start_year: int,
    fs_variant: str,
    method: str,
    selected_component: str,
    w_new: float | None,
    y_val: np.ndarray,
    val_proba: np.ndarray,
    y_test: np.ndarray,
    test_proba: np.ndarray,
) -> dict:
    threshold = _select_threshold(y_val, val_proba)
    val_metrics = compute_metrics(y_val, val_proba, threshold=threshold)
    test_metrics = compute_metrics(y_test, test_proba, threshold=threshold)
    row = {
        "split": split_label,
        "drift_start_year": drift_start_year,
        "old_window": f"1999-{drift_start_year - 1}",
        "new_window": f"{drift_start_year}-{TRAIN_END_YEAR}",
        "fs_variant": fs_variant,
        "method": method,
        "selected_component": selected_component,
        "w_new": w_new,
        "threshold": threshold,
    }
    for metric in METRICS:
        row[f"val_{metric}"] = val_metrics[metric]
        row[f"test_{metric}"] = test_metrics[metric]
    return row


def _run_split(
    *,
    split_label: str,
    drift_start_year: int,
    fs_variant: str,
    X_train_all: pd.DataFrame,
    y_train_all: pd.Series,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    logger,
) -> list[dict]:
    X_old, y_old, X_new, y_new = _old_new_split(X_train_all, y_train_all, drift_start_year)
    X_test_used = X_test
    if fs_variant == "fs":
        n_old_val = max(1, int(len(X_old) * 0.2))
        X_old, X_new, X_test_used, _, _ = _apply_old_fit_fs(
            X_old,
            y_old,
            X_new,
            X_test,
            logger,
            n_old_val=n_old_val,
        )

    y_val, val_pool, test_pool = _train_pool_probas(
        X_old,
        y_old,
        X_new,
        y_new,
        X_test_used,
        tag=f"fair_{split_label}_{fs_variant}",
    )
    old_val = np.mean(val_pool[:3], axis=0)
    new_val = np.mean(val_pool[3:], axis=0)
    old_test = np.mean(test_pool[:3], axis=0)
    new_test = np.mean(test_pool[3:], axis=0)

    candidates: list[tuple[str, str, float | None, np.ndarray, np.ndarray]] = [
        ("New_under", "New_under", 1.0, val_pool[3], test_pool[3]),
        ("New3_mean", "New_under+New_over+New_hybrid", 1.0, new_val, new_test),
        (
            "Equal6",
            "Old3/New3 equal weight",
            0.5,
            0.5 * old_val + 0.5 * new_val,
            0.5 * old_test + 0.5 * new_test,
        ),
    ]

    single_rows: list[tuple[int, dict]] = []
    for idx, (label, val_proba, test_proba) in enumerate(
        zip(MODEL_LABELS, val_pool, test_pool)
    ):
        threshold = _select_threshold(y_val, val_proba)
        single_rows.append(
            (
                idx,
                {
                    "label": label,
                    "val_proba": val_proba,
                    "test_proba": test_proba,
                    **compute_metrics(y_val, val_proba, threshold=threshold),
                },
            )
        )

    best_single_f1 = max(single_rows, key=lambda item: (item[1]["F1"], item[1]["AUC"]))[1]
    best_single_auc = max(single_rows, key=lambda item: (item[1]["AUC"], item[1]["F1"]))[1]
    candidates.extend(
        [
            (
                "ValBestSingle_F1",
                best_single_f1["label"],
                None,
                best_single_f1["val_proba"],
                best_single_f1["test_proba"],
            ),
            (
                "ValBestSingle_AUC",
                best_single_auc["label"],
                None,
                best_single_auc["val_proba"],
                best_single_auc["test_proba"],
            ),
        ]
    )

    weight_rows: list[dict] = []
    for w_new in WEIGHT_GRID:
        val_proba = (1.0 - w_new) * old_val + w_new * new_val
        test_proba = (1.0 - w_new) * old_test + w_new * new_test
        threshold = _select_threshold(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold=threshold)
        weight_rows.append(
            {
                "w_new": float(w_new),
                "val_proba": val_proba,
                "test_proba": test_proba,
                **val_metrics,
            }
        )

    best_weight_f1 = max(weight_rows, key=lambda row: (row["F1"], row["AUC"]))
    best_weight_auc = max(weight_rows, key=lambda row: (row["AUC"], row["F1"]))
    candidates.extend(
        [
            (
                "DAWCE_F1",
                f"Old3/New3 w_new={best_weight_f1['w_new']:.2f}",
                best_weight_f1["w_new"],
                best_weight_f1["val_proba"],
                best_weight_f1["test_proba"],
            ),
            (
                "DAWCE_AUC",
                f"Old3/New3 w_new={best_weight_auc['w_new']:.2f}",
                best_weight_auc["w_new"],
                best_weight_auc["val_proba"],
                best_weight_auc["test_proba"],
            ),
        ]
    )

    return [
        _evaluate_candidate(
            split_label=split_label,
            drift_start_year=drift_start_year,
            fs_variant=fs_variant,
            method=method,
            selected_component=component,
            w_new=w_new,
            y_val=y_val,
            val_proba=val_proba,
            y_test=y_test,
            test_proba=test_proba,
        )
        for method, component, w_new, val_proba, test_proba in candidates
    ]


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    aggregation = {
        **{f"val_{metric}": "mean" for metric in METRICS},
        **{f"test_{metric}": "mean" for metric in METRICS},
        "w_new": "mean",
    }
    return (
        results.groupby(["fs_variant", "method"], as_index=False)
        .agg(aggregation)
        .sort_values(["fs_variant", "test_AUC"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _add_adaptive_choices(results: pd.DataFrame) -> pd.DataFrame:
    """Let validation choose between the best single model and DAWCE."""
    adaptive_rows: list[pd.Series] = []
    choice_specs = (
        ("AdaptiveChoice_F1", "ValBestSingle_F1", "DAWCE_F1", "val_F1", "val_AUC"),
        ("AdaptiveChoice_AUC", "ValBestSingle_AUC", "DAWCE_AUC", "val_AUC", "val_F1"),
    )
    for (_, _), group in results.groupby(["split", "fs_variant"], sort=False):
        for method, single_method, dawce_method, primary, secondary in choice_specs:
            choices = group[group["method"].isin([single_method, dawce_method])]
            selected = choices.sort_values(
                [primary, secondary],
                ascending=[False, False],
            ).iloc[0].copy()
            source_method = selected["method"]
            selected["method"] = method
            selected["selected_component"] = (
                f"{source_method}: {selected['selected_component']}"
            )
            adaptive_rows.append(selected)
    return pd.concat([results, pd.DataFrame(adaptive_rows)], ignore_index=True)


def _build_wilcoxon(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for fs_variant, fs_group in results.groupby("fs_variant"):
        baseline = fs_group[fs_group["method"] == "New_under"].sort_values("split")
        for method, method_group in fs_group.groupby("method"):
            if method == "New_under":
                continue
            method_group = method_group.sort_values("split")
            for metric in ("AUC", "F1", "Recall", "Precision"):
                candidate = method_group[f"test_{metric}"].to_numpy(dtype=float)
                reference = baseline[f"test_{metric}"].to_numpy(dtype=float)
                diff = candidate - reference
                if np.allclose(diff, 0.0):
                    p_two_sided = 1.0
                    p_greater = 1.0
                else:
                    p_two_sided = float(
                        wilcoxon(candidate, reference, alternative="two-sided").pvalue
                    )
                    p_greater = float(
                        wilcoxon(candidate, reference, alternative="greater").pvalue
                    )
                rows.append(
                    {
                        "fs_variant": fs_variant,
                        "method": method,
                        "baseline": "New_under",
                        "metric": metric,
                        "n_pairs": len(diff),
                        "method_mean": float(np.mean(candidate)),
                        "baseline_mean": float(np.mean(reference)),
                        "mean_diff": float(np.mean(diff)),
                        "n_method_better": int(np.sum(diff > 0)),
                        "n_baseline_better": int(np.sum(diff < 0)),
                        "p_greater": p_greater,
                        "p_two_sided": p_two_sided,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    logger = get_logger("FairWeightedAblation", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_all, y_all = load_bankruptcy_with_year(logger)
    train_mask = (X_all["fyear"] >= 1999) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_train_all = X_all.loc[train_mask].reset_index(drop=True)
    y_train_all = y_all.loc[train_mask].reset_index(drop=True)
    X_test, y_test = _test_split(X_all, y_all)

    rows: list[dict] = []
    for idx, (split_label, old_end_year) in enumerate(YEAR_SPLITS, 1):
        logger.info(f"[{idx}/{len(YEAR_SPLITS)}] Running fair ablation for {split_label}")
        for fs_variant in ("no_fs", "fs"):
            rows.extend(
                _run_split(
                    split_label=split_label,
                    drift_start_year=old_end_year + 1,
                    fs_variant=fs_variant,
                    X_train_all=X_train_all,
                    y_train_all=y_train_all,
                    X_test=X_test,
                    y_test=y_test,
                    logger=logger,
                )
            )

    results = _add_adaptive_choices(pd.DataFrame(rows))
    summary = _build_summary(results)
    wilcoxon_df = _build_wilcoxon(results)

    results_path = OUTPUT_DIR / "bk_fair_ablation_by_split.csv"
    summary_path = OUTPUT_DIR / "bk_fair_ablation_summary.csv"
    wilcoxon_path = OUTPUT_DIR / "bk_fair_ablation_vs_new_under_wilcoxon.csv"
    results.to_csv(results_path, index=False, float_format="%.8f")
    summary.to_csv(summary_path, index=False, float_format="%.8f")
    wilcoxon_df.to_csv(wilcoxon_path, index=False, float_format="%.8f")

    logger.info(f"Saved split results -> {results_path}")
    logger.info(f"Saved summary -> {summary_path}")
    logger.info(f"Saved Wilcoxon -> {wilcoxon_path}")
    logger.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
