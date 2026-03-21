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
    TYPE_K_SUBSET_DETAIL,
    TYPE_K_SUBSETS_MEAN,
    SAMPLING_DYNAMIC_ALL6,
    DYNAMIC_DES_METHODS,
    train_one_sampling_xgb,
    ensemble_metrics_with_threshold,
    dynamic_ensemble_metrics_with_threshold,
    combination_metrics_6_models_details,
)

SplitFn = Callable[[logging.Logger, int], tuple]


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

    X_old, y_old, X_new, y_new, X_test, y_test = get_split(logger, old_end_year)

    y_old = np.asarray(y_old)
    y_new = np.asarray(y_new)
    y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)

    try:
        X_new_fit_raw, X_new_val_raw, y_new_fit, y_new_val = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
        )
    except ValueError:
        X_new_fit_raw, X_new_val_raw, y_new_fit, y_new_val = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42
        )

    pre_old = DataPreprocessor()
    X_old_scaled = pre_old.scale_features(X_old, fit=True)[0]
    X_new_val_scaled_old = pre_old.scale_features(X_new_val_raw, fit=False)[0]
    X_test_scaled_old = pre_old.scale_features(X_test, fit=False)[0]

    pre_new = DataPreprocessor()
    X_new_fit_scaled = pre_new.scale_features(X_new_fit_raw, fit=True)[0]
    X_new_val_scaled_new = pre_new.scale_features(X_new_val_raw, fit=False)[0]
    X_test_scaled_new = pre_new.scale_features(X_test, fit=False)[0]

    old_val_probas_all: list[np.ndarray] = []
    old_test_probas_all: list[np.ndarray] = []
    new_val_probas_all: list[np.ndarray] = []
    new_test_probas_all: list[np.ndarray] = []

    mi = model_infix

    for s in SAMPLING_STRATEGIES:
        val_p, test_p = train_one_sampling_xgb(
            X_train_scaled=X_old_scaled,
            y_train=y_old,
            X_val_scaled=X_new_val_scaled_old,
            X_test_scaled=X_test_scaled_old,
            sampler=sampler,
            sampling_strategy=s,
            model_name=f"old_{label}{mi}_{s}",
        )
        old_val_probas_all.append(val_p)
        old_test_probas_all.append(test_p)

    for s in SAMPLING_STRATEGIES:
        val_p, test_p = train_one_sampling_xgb(
            X_train_scaled=X_new_fit_scaled,
            y_train=np.asarray(y_new_fit),
            X_val_scaled=X_new_val_scaled_new,
            X_test_scaled=X_test_scaled_new,
            sampler=sampler,
            sampling_strategy=s,
            model_name=f"new_{label}{mi}_{s}",
        )
        new_val_probas_all.append(val_p)
        new_test_probas_all.append(test_p)

    y_val_arr = np.asarray(y_new_val)
    X_val_np = np.asarray(X_new_val_scaled_new, dtype=np.float64)
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
        "old_new_all": "OldNew",
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

    for k in [2, 3, 4, 5, 6]:
        details = combination_metrics_6_models_details(
            y_val=y_val_arr,
            val_probas_6=val_probas_6,
            y_test=y_test_arr,
            test_probas_6=test_probas_6,
            k=k,
        )
        for d in details:
            static_rows.append(
                {
                    "dataset": year_combo,
                    "split": label,
                    "ensemble": f"{k}models",
                    "sampling_col": TYPE_K_SUBSET_DETAIL,
                    "subset": d["subset_label"],
                    "subset_indices": d["subset_indices"],
                    "AUC": d["AUC"],
                    "F1": d["F1"],
                    "Recall": d["Recall"],
                }
            )
        if details:
            static_rows.append(
                {
                    "dataset": year_combo,
                    "split": label,
                    "ensemble": f"{k}models",
                    "sampling_col": TYPE_K_SUBSETS_MEAN,
                    "subset": "",
                    "subset_indices": "",
                    "AUC": float(np.mean([d["AUC"] for d in details])),
                    "F1": float(np.mean([d["F1"] for d in details])),
                    "Recall": float(np.mean([d["Recall"] for d in details])),
                }
            )
        else:
            static_rows.append(
                {
                    "dataset": year_combo,
                    "split": label,
                    "ensemble": f"{k}models",
                    "sampling_col": TYPE_K_SUBSETS_MEAN,
                    "subset": "",
                    "subset_indices": "",
                    "AUC": np.nan,
                    "F1": np.nan,
                    "Recall": np.nan,
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
            get_split=lambda lg, oy: get_bankruptcy_year_split(lg, old_end_year=oy),
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
            get_split=lambda lg, oy, t=ticker: get_stock_year_split(lg, old_end_year=oy, ticker=t),
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
            get_split=lambda lg, oy: get_medical_year_split(lg, old_end_year=oy),
            model_infix="",
        )
        all_static.extend(sr)
        all_des.extend(dr)
    return all_static, all_des
