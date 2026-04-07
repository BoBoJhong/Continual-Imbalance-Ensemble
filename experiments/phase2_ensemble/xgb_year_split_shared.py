"""
XGB Old/New 年份切割 — 靜態與動態 DES 共用訓練（Bankruptcy / Stock / Medical）。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import DataPreprocessor, ImbalanceSampler

from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split
from experiments._shared.common_dataset import MEDICAL_YEAR_SPLITS, STOCK_YEAR_SPLITS, get_medical_year_split, get_stock_year_split

from experiments.phase2_ensemble.xgb_oldnew_ensemble_common import (
    SAMPLING_STRATEGIES,
    SAMPLING_DYNAMIC_ALL6,
    DYNAMIC_DES_METHODS,
    train_one_sampling_xgb,
    ensemble_metrics_with_threshold,
    dynamic_ensemble_metrics_with_threshold,
)

SplitFn = Callable[[logging.Logger, int], tuple]


def _split_fit_val(X_raw, y_raw, test_size: float = 0.2, random_state: int = 42):
    y_arr = np.asarray(y_raw)
    try:
        X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
            X_raw,
            y_arr,
            test_size=test_size,
            random_state=random_state,
            stratify=y_arr,
        )
    except ValueError:
        X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
            X_raw,
            y_arr,
            test_size=test_size,
            random_state=random_state,
        )
    return X_fit_raw, np.asarray(y_fit), X_val_raw, np.asarray(y_val)


def _split_fit_val_by_year(X_raw, y_raw, year_arr, val_ratio: float = 0.2, random_state: int = 42):
    years = np.asarray(year_arr)
    if years.size == 0 or len(np.unique(years)) <= 1:
        return _split_fit_val(X_raw, y_raw, test_size=val_ratio, random_state=random_state)

    fit_idx_all = []
    val_idx_all = []
    y_arr = np.asarray(y_raw)

    for yr in sorted(np.unique(years)):
        idx = np.where(years == yr)[0]
        n = len(idx)
        if n <= 1:
            fit_idx_all.extend(idx.tolist())
            continue

        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)

        stratify = None
        y_sub = y_arr[idx]
        if len(np.unique(y_sub)) >= 2 and n_val >= len(np.unique(y_sub)):
            stratify = y_sub

        try:
            idx_fit, idx_val = train_test_split(
                idx,
                test_size=n_val,
                random_state=random_state + int(yr),
                stratify=stratify,
            )
        except ValueError:
            idx_fit, idx_val = train_test_split(
                idx,
                test_size=n_val,
                random_state=random_state + int(yr),
            )

        fit_idx_all.extend(idx_fit.tolist())
        val_idx_all.extend(idx_val.tolist())

    if len(val_idx_all) == 0:
        return _split_fit_val(X_raw, y_raw, test_size=val_ratio, random_state=random_state)

    fit_idx = np.array(sorted(fit_idx_all))
    val_idx = np.array(sorted(val_idx_all))
    X_fit_raw = X_raw.iloc[fit_idx].reset_index(drop=True)
    y_fit = y_arr[fit_idx]
    X_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
    y_val = y_arr[val_idx]
    return X_fit_raw, np.asarray(y_fit), X_val_raw, np.asarray(y_val)


def process_one_year_split(
    label: str,
    old_end_year: int,
    logger: logging.Logger,
    sampler: ImbalanceSampler,
    *,
    get_split: SplitFn,
    model_infix: str = "",
) -> Tuple[List[dict], List[dict]]:
    static_rows: List[dict] = []
    des_rows: List[dict] = []

    X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, year_test = get_split(logger, old_end_year)

    y_old = np.asarray(y_old)
    y_new = np.asarray(y_new)
    y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)

    X_old_fit_raw, y_old_fit, X_old_val_raw, y_old_val = _split_fit_val_by_year(X_old, y_old, year_old)
    X_new_fit_raw, y_new_fit, X_new_val_raw, y_new_val = _split_fit_val_by_year(X_new, y_new, year_new)

    pre_old = DataPreprocessor()
    X_old_fit_scaled = pre_old.scale_features(X_old_fit_raw, fit=True)[0]
    X_old_val_scaled_old = pre_old.scale_features(X_old_val_raw, fit=False)[0]
    X_new_val_scaled_old = pre_old.scale_features(X_new_val_raw, fit=False)[0]
    X_test_scaled_old = pre_old.scale_features(X_test, fit=False)[0]

    pre_new = DataPreprocessor()
    X_new_fit_scaled = pre_new.scale_features(X_new_fit_raw, fit=True)[0]
    X_old_val_scaled_new = pre_new.scale_features(X_old_val_raw, fit=False)[0]
    X_new_val_scaled_new = pre_new.scale_features(X_new_val_raw, fit=False)[0]
    X_test_scaled_new = pre_new.scale_features(X_test, fit=False)[0]

    X_val_scaled_old_all = np.concatenate(
        [np.asarray(X_old_val_scaled_old), np.asarray(X_new_val_scaled_old)], axis=0
    )
    X_val_scaled_new_all = np.concatenate(
        [np.asarray(X_old_val_scaled_new), np.asarray(X_new_val_scaled_new)], axis=0
    )
    y_val_arr = np.concatenate([np.asarray(y_old_val), np.asarray(y_new_val)], axis=0)

    old_val_probas_all: list[np.ndarray] = []
    old_test_probas_all: list[np.ndarray] = []
    new_val_probas_all: list[np.ndarray] = []
    new_test_probas_all: list[np.ndarray] = []

    mi = model_infix

    for s in SAMPLING_STRATEGIES:
        val_p, test_p = train_one_sampling_xgb(
            X_train_scaled=X_old_fit_scaled,
            y_train=np.asarray(y_old_fit),
            X_val_scaled=X_val_scaled_old_all,
            X_test_scaled=X_test_scaled_old,
            sampler=sampler,
            sampling_strategy=s,
            model_name=f"old_{label}{mi}_{s}",
            split_label=label,
            method_label="Old",
        )
        old_val_probas_all.append(val_p)
        old_test_probas_all.append(test_p)

    for s in SAMPLING_STRATEGIES:
        val_p, test_p = train_one_sampling_xgb(
            X_train_scaled=X_new_fit_scaled,
            y_train=np.asarray(y_new_fit),
            X_val_scaled=X_val_scaled_new_all,
            X_test_scaled=X_test_scaled_new,
            sampler=sampler,
            sampling_strategy=s,
            model_name=f"new_{label}{mi}_{s}",
            split_label=label,
            method_label="New",
        )
        new_val_probas_all.append(val_p)
        new_test_probas_all.append(test_p)

    X_val_np = np.asarray(X_val_scaled_new_all, dtype=np.float64)
    X_test_np = np.asarray(X_test_scaled_new, dtype=np.float64)

    year_combo = label.split("split_", 1)[-1] if str(label).startswith("split_") else str(label)
    sampling_to_col = {
        "undersampling": "under",
        "oversampling": "over",
        "hybrid": "hybrid",
    }
    ensemble_map = {
        "old_only": "Old",
        "new_only": "New",
        "old_new_all": "Retrain",
    }

    for i, s in enumerate(SAMPLING_STRATEGIES):
        sampling_col = sampling_to_col[s]

        old_only_metrics = ensemble_metrics_with_threshold(
            y_val=y_val_arr,
            val_probas=[old_val_probas_all[i]],
            y_test=y_test_arr,
            test_probas=[old_test_probas_all[i]],
        )
        static_rows.append(
            {
                "dataset": year_combo,
                "split": label,
                "ensemble": ensemble_map["old_only"],
                "sampling_col": sampling_col,
                "subset": "",
                "subset_indices": "",
                **{k: old_only_metrics[k] for k in ["AUC", "F1", "Recall"]},
            }
        )

        new_only_metrics = ensemble_metrics_with_threshold(
            y_val=y_val_arr,
            val_probas=[new_val_probas_all[i]],
            y_test=y_test_arr,
            test_probas=[new_test_probas_all[i]],
        )
        static_rows.append(
            {
                "dataset": year_combo,
                "split": label,
                "ensemble": ensemble_map["new_only"],
                "sampling_col": sampling_col,
                "subset": "",
                "subset_indices": "",
                **{k: new_only_metrics[k] for k in ["AUC", "F1", "Recall"]},
            }
        )

        old_new_all_metrics = ensemble_metrics_with_threshold(
            y_val=y_val_arr,
            val_probas=[old_val_probas_all[i], new_val_probas_all[i]],
            y_test=y_test_arr,
            test_probas=[old_test_probas_all[i], new_test_probas_all[i]],
        )
        static_rows.append(
            {
                "dataset": year_combo,
                "split": label,
                "ensemble": ensemble_map["old_new_all"],
                "sampling_col": sampling_col,
                "subset": "",
                "subset_indices": "",
                **{k: old_new_all_metrics[k] for k in ["AUC", "F1", "Recall"]},
            }
        )

        for method_key, ens_name in DYNAMIC_DES_METHODS:
            dyn_m = dynamic_ensemble_metrics_with_threshold(
                y_val_arr,
                X_val_np,
                [old_val_probas_all[i], new_val_probas_all[i]],
                y_test_arr,
                X_test_np,
                [old_test_probas_all[i], new_test_probas_all[i]],
                method=method_key,
                k_neighbors=7,
            )
            des_rows.append(
                {
                    "dataset": year_combo,
                    "split": label,
                    "ensemble": ens_name,
                    "sampling_col": sampling_col,
                    "subset": "",
                    "subset_indices": "",
                    **{k: dyn_m[k] for k in ["AUC", "F1", "Recall"]},
                }
            )

    val_probas_6 = old_val_probas_all + new_val_probas_all
    test_probas_6 = old_test_probas_all + new_test_probas_all
    for method_key, ens_name in DYNAMIC_DES_METHODS:
        dyn6 = dynamic_ensemble_metrics_with_threshold(
            y_val_arr,
            X_val_np,
            val_probas_6,
            y_test_arr,
            X_test_np,
            test_probas_6,
            method=method_key,
            k_neighbors=7,
        )
        des_rows.append(
            {
                "dataset": year_combo,
                "split": label,
                "ensemble": ens_name,
                "sampling_col": SAMPLING_DYNAMIC_ALL6,
                "subset": "",
                "subset_indices": "",
                **{k: dyn6[k] for k in ["AUC", "F1", "Recall"]},
            }
        )

    return static_rows, des_rows


def iter_bankruptcy_year_splits(logger: logging.Logger) -> Tuple[List[dict], List[dict]]:
    sampler = ImbalanceSampler()
    all_static: List[dict] = []
    all_des: List[dict] = []
    for label, old_end_year in YEAR_SPLITS:
        sr, dr = process_one_year_split(
            label,
            old_end_year,
            logger,
            sampler,
            get_split=lambda lg, oy: get_bankruptcy_year_split(lg, old_end_year=oy, return_years=True),
            model_infix="",
        )
        all_static.extend(sr)
        all_des.extend(dr)
    return all_static, all_des


def iter_stock_year_splits(logger: logging.Logger, ticker: str = "spx") -> Tuple[List[dict], List[dict]]:
    sampler = ImbalanceSampler()
    all_static: List[dict] = []
    all_des: List[dict] = []
    mi = f"_{ticker}"
    for label, old_end_year in STOCK_YEAR_SPLITS:
        sr, dr = process_one_year_split(
            label,
            old_end_year,
            logger,
            sampler,
            get_split=lambda lg, oy, t=ticker: get_stock_year_split(
                lg,
                old_end_year=oy,
                ticker=t,
                return_years=True,
            ),
            model_infix=mi,
        )
        all_static.extend(sr)
        all_des.extend(dr)
    return all_static, all_des


def iter_medical_year_splits(logger: logging.Logger) -> Tuple[List[dict], List[dict]]:
    sampler = ImbalanceSampler()
    all_static: List[dict] = []
    all_des: List[dict] = []
    for label, old_end_year in MEDICAL_YEAR_SPLITS:
        sr, dr = process_one_year_split(
            label,
            old_end_year,
            logger,
            sampler,
            get_split=lambda lg, oy: get_medical_year_split(lg, old_end_year=oy, return_years=True),
            model_infix="",
        )
        all_static.extend(sr)
        all_des.extend(dr)
    return all_static, all_des
