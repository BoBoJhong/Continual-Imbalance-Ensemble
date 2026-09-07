"""
Phase 3 follow-up: feature selection stability analysis for Bankruptcy.

This script reuses the same year splits and FeatureSelector implementation used
by Study II, then measures whether selected feature sets remain stable as the
Old/New boundary changes.

Outputs:
    results/phase3_feature/stability/bankruptcy_feature_stability_selected_features.csv
    results/phase3_feature/stability/bankruptcy_feature_stability_pairwise_jaccard.csv
    results/phase3_feature/stability/bankruptcy_feature_stability_summary.csv
    results/phase3_feature/stability/bankruptcy_feature_stability_frequency.csv
    results/phase3_feature/stability/bankruptcy_feature_stability_errors.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments._shared.common_bankruptcy import (  # noqa: E402
    YEAR_SPLITS,
    get_bankruptcy_year_split,
)
from src.features import FeatureSelector  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402

OUTPUT_DIR = project_root / "results" / "phase3_feature" / "stability"
FS_METHODS = ("mutual_info", "shap", "rfe")
FS_RATIOS = (0.5, 0.8)
TRAIN_SCOPES = ("old", "new", "old_new")
ERROR_COLUMNS = [
    "split",
    "old_end_year",
    "new_start_year",
    "train_scope",
    "fs_method",
    "fs_ratio",
    "error_type",
    "error_message",
]


def _fit_frame_for_scope(
    scope: str,
    X_old: pd.DataFrame,
    y_old: np.ndarray,
    X_new: pd.DataFrame,
    y_new: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    if scope == "old":
        return X_old, y_old
    if scope == "new":
        return X_new, y_new
    if scope == "old_new":
        X_fit = pd.concat([X_old, X_new], ignore_index=True)
        y_fit = np.concatenate([y_old, y_new])
        return X_fit, y_fit
    raise ValueError(f"Unknown train scope: {scope}")


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _build_pairwise_jaccard(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["train_scope", "fs_method", "fs_ratio"]

    for group_key, group in selected_df.groupby(group_cols):
        train_scope, fs_method, fs_ratio = group_key
        records = group.to_dict("records")
        for left, right in combinations(records, 2):
            left_features = str(left["selected_features"]).split(";")
            right_features = str(right["selected_features"]).split(";")
            rows.append(
                {
                    "train_scope": train_scope,
                    "fs_method": fs_method,
                    "fs_ratio": fs_ratio,
                    "split_a": left["split"],
                    "split_b": right["split"],
                    "old_end_year_a": left["old_end_year"],
                    "old_end_year_b": right["old_end_year"],
                    "jaccard": _jaccard(left_features, right_features),
                }
            )

    return pd.DataFrame(rows)


def _build_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["train_scope", "fs_method", "fs_ratio"]

    for group_key, group in pairwise_df.groupby(group_cols):
        train_scope, fs_method, fs_ratio = group_key
        rows.append(
            {
                "train_scope": train_scope,
                "fs_method": fs_method,
                "fs_ratio": fs_ratio,
                "n_pairs": len(group),
                "jaccard_mean": group["jaccard"].mean(),
                "jaccard_std": group["jaccard"].std(),
                "jaccard_min": group["jaccard"].min(),
                "jaccard_max": group["jaccard"].max(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["train_scope", "jaccard_mean", "fs_method", "fs_ratio"],
        ascending=[True, False, True, True],
    )


def _build_frequency(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["train_scope", "fs_method", "fs_ratio"]

    for group_key, group in selected_df.groupby(group_cols):
        train_scope, fs_method, fs_ratio = group_key
        n_splits = group["split"].nunique()
        counts: dict[str, int] = {}
        for features in group["selected_features"]:
            for feature in str(features).split(";"):
                counts[feature] = counts.get(feature, 0) + 1
        for feature, count in counts.items():
            rows.append(
                {
                    "train_scope": train_scope,
                    "fs_method": fs_method,
                    "fs_ratio": fs_ratio,
                    "feature": feature,
                    "selected_count": count,
                    "n_splits": n_splits,
                    "selected_rate": count / n_splits if n_splits else 0.0,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["train_scope", "fs_method", "fs_ratio", "selected_count", "feature"],
        ascending=[True, True, True, False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair-empty-errors",
        action="store_true",
        help="Normalize an existing blank error artifact without rerunning selectors",
    )
    args = parser.parse_args()
    errors_path = OUTPUT_DIR / "bankruptcy_feature_stability_errors.csv"
    if args.repair_empty_errors:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if errors_path.exists() and errors_path.read_text(encoding="utf-8").strip():
            pd.read_csv(errors_path)
            print(f"Existing error artifact is already readable: {errors_path}")
        else:
            pd.DataFrame(columns=ERROR_COLUMNS).to_csv(errors_path, index=False)
            print(f"Normalized empty error artifact: {errors_path}")
        return

    logger = get_logger("Phase3_Feature_Stability", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict] = []
    error_rows: list[dict] = []

    for split_idx, (label, old_end_year) in enumerate(YEAR_SPLITS, 1):
        logger.info(
            f"\n[{split_idx}/{len(YEAR_SPLITS)}] Feature stability for {label} "
            f"(Old<= {old_end_year}, New={old_end_year + 1}-2014)"
        )
        X_old, y_old, X_new, y_new, _, _, _, _, _ = get_bankruptcy_year_split(
            logger,
            old_end_year=old_end_year,
            return_years=True,
        )
        y_old_arr = np.asarray(y_old)
        y_new_arr = np.asarray(y_new)
        n_features_before = X_old.shape[1]

        for train_scope in TRAIN_SCOPES:
            X_fit, y_fit = _fit_frame_for_scope(
                train_scope,
                X_old,
                y_old_arr,
                X_new,
                y_new_arr,
            )
            for fs_method in FS_METHODS:
                for fs_ratio in FS_RATIOS:
                    k = max(1, int(n_features_before * fs_ratio))
                    logger.info(
                        f"  scope={train_scope:7s} method={fs_method:11s} "
                        f"ratio={fs_ratio:.1f} k={k}"
                    )
                    try:
                        selector = FeatureSelector(method=fs_method, k=k)
                        selector.fit(X_fit, y_fit)
                        selected_cols = list(selector.selected_cols_ or [])
                    except Exception as exc:
                        logger.error(
                            f"    skipped scope={train_scope} method={fs_method} "
                            f"ratio={fs_ratio}: {exc}"
                        )
                        error_rows.append(
                            {
                                "split": label,
                                "old_end_year": old_end_year,
                                "new_start_year": old_end_year + 1,
                                "train_scope": train_scope,
                                "fs_method": fs_method,
                                "fs_ratio": fs_ratio,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                        )
                        continue
                    selected_rows.append(
                        {
                            "split": label,
                            "old_end_year": old_end_year,
                            "new_start_year": old_end_year + 1,
                            "train_scope": train_scope,
                            "fs_method": fs_method,
                            "fs_ratio": fs_ratio,
                            "n_features_before": n_features_before,
                            "n_features_after": len(selected_cols),
                            "selected_features": ";".join(selected_cols),
                        }
                    )

    selected_df = pd.DataFrame(selected_rows)
    pairwise_df = _build_pairwise_jaccard(selected_df)
    summary_df = _build_summary(pairwise_df)
    frequency_df = _build_frequency(selected_df)

    selected_path = OUTPUT_DIR / "bankruptcy_feature_stability_selected_features.csv"
    pairwise_path = OUTPUT_DIR / "bankruptcy_feature_stability_pairwise_jaccard.csv"
    summary_path = OUTPUT_DIR / "bankruptcy_feature_stability_summary.csv"
    frequency_path = OUTPUT_DIR / "bankruptcy_feature_stability_frequency.csv"

    selected_df.to_csv(selected_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False, float_format="%.6f")
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    frequency_df.to_csv(frequency_path, index=False, float_format="%.6f")
    pd.DataFrame(error_rows, columns=ERROR_COLUMNS).to_csv(errors_path, index=False)

    logger.info(f"\nSaved selected features -> {selected_path}")
    logger.info(f"Saved pairwise Jaccard -> {pairwise_path}")
    logger.info(f"Saved stability summary -> {summary_path}")
    logger.info(f"Saved feature frequency -> {frequency_path}")
    logger.info(f"Saved skipped configs/errors -> {errors_path}")
    logger.info("\nTop stability summary:")
    logger.info("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
