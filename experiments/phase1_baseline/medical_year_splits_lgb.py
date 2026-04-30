"""
Phase 1 - Medical 年份切割基準線實驗（LightGBM + scale_pos_weight）
=======================================================
資料集：UCI Diabetes 130-US Hospitals (1999-2008，偽時序)

固定 Test = 2007-2008；訓練窗 1999–2006（8 年）。依 `experiments._shared.common_dataset.MEDICAL_YEAR_SPLITS`
逐年滑動 old_end，共 7 組 Old/New 切割（split_1+7 … split_7+1）。

訓練策略（與 XGB year-split baseline 對齊）：
  - Old      : 只用 Old 資料訓練
  - New      : 只用 New 資料訓練
  - Retrain  : Old+New 全量合併訓練（整個實驗只跑一次）

採樣策略：none / undersampling / oversampling / hybrid

不平衡處理：LightGBM 使用 `scale_pos_weight = n_negative / n_positive`（由採樣後的訓練資料計算）。

調參（tuned）：固定 validation set 上做 Random search（n_iter 組），目標為 validation ROC-AUC。

輸出：會自動建立子資料夾避免覆蓋
  - results/phase1_baseline/lgbm/<tag>/...
  - results/phase1_baseline/lgbm/<tag>/tuned/...

用法：
  python experiments/phase1_baseline/medical_year_splits_lgb.py --tuning tuned --tune-n-iter 200 --output-tag med_lgb_tuned200_20260421
  python experiments/phase1_baseline/medical_year_splits_lgb.py --tuning both  --tune-n-iter 200 --output-tag med_lgb_both_20260421
"""

from __future__ import annotations

import argparse
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataPreprocessor, ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import LightGBMWrapper
from src.utils import get_config_loader, get_logger, set_seed

from experiments._shared.baseline_val_search import export_tuning_log, search_lgb_on_val, tuning_meta
from experiments._shared.common_dataset import MEDICAL_YEAR_SPLITS, get_medical_year_split


SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]

METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]

OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "lgbm"

BASE_YEAR = 1999
TRAIN_END_YEAR = 2006
TOTAL_TRAIN_YEARS = TRAIN_END_YEAR - BASE_YEAR + 1


def _select_threshold_from_validation(y_val: np.ndarray, y_proba_val: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def _split_fit_val(
    X_raw: pd.DataFrame,
    y_raw: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
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


def _split_fit_val_by_year(
    X_raw: pd.DataFrame,
    y_raw: np.ndarray,
    year_arr: pd.Series | np.ndarray,
    *,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    years = np.asarray(year_arr)
    if years.size == 0 or len(np.unique(years)) <= 1:
        return _split_fit_val(X_raw, y_raw, test_size=val_ratio, random_state=random_state)

    fit_idx_all: list[int] = []
    val_idx_all: list[int] = []
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


def _build_retrain_fit_val(
    X_old: pd.DataFrame,
    y_old: pd.Series | np.ndarray,
    year_old: pd.Series | np.ndarray,
    X_new: pd.DataFrame,
    y_new: pd.Series | np.ndarray,
    year_new: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    X_old_fit, y_old_fit, X_old_val, y_old_val = _split_fit_val_by_year(X_old, y_old, year_old)
    X_new_fit, y_new_fit, X_new_val, y_new_val = _split_fit_val_by_year(X_new, y_new, year_new)

    X_fit = pd.concat([X_old_fit, X_new_fit], ignore_index=True)
    y_fit = np.concatenate([y_old_fit, y_new_fit])
    X_val = pd.concat([X_old_val, X_new_val], ignore_index=True)
    y_val = np.concatenate([y_old_val, y_new_val])
    return X_fit, y_fit, X_val, y_val


def _scale_pos_weight(y: np.ndarray) -> float:
    yv = np.asarray(y).ravel()
    unique, counts = np.unique(yv, return_counts=True)
    if len(unique) != 2:
        return 1.0
    neg_count = counts[0] if unique[0] == 0 else counts[1]
    pos_count = counts[1] if unique[1] == 1 else counts[0]
    return float(neg_count / max(pos_count, 1))


def _train_eval(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    sampler: ImbalanceSampler,
    strategy: str,
    tag: str,
    logger,
    X_val_raw: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    year_train: pd.Series | np.ndarray | None = None,
    *,
    split_label: str = "",
    use_tuning: bool = False,
    n_tune_iter: int = 48,
) -> dict:
    y_train_arr = np.asarray(y_train)

    if X_val_raw is None or y_val is None:
        if year_train is not None:
            X_fit_raw, y_fit, X_val_raw, y_val = _split_fit_val_by_year(
                X_train_raw,
                y_train_arr,
                year_train,
            )
        else:
            X_fit_raw, y_fit, X_val_raw, y_val = _split_fit_val(X_train_raw, y_train_arr)
    else:
        X_fit_raw = X_train_raw
        y_fit = y_train_arr

    pre = DataPreprocessor()
    X_fit, X_val = pre.scale_features(X_fit_raw, X_val_raw, fit=True)
    _, X_test = pre.scale_features(X_fit_raw, X_test_raw, fit=False)

    X_r, y_r = sampler.apply_sampling(X_fit, np.asarray(y_fit), strategy=strategy)
    spw = _scale_pos_weight(y_r)

    tune_seed = 42 + (zlib.adler32(f"{split_label}|{tag}|{strategy}".encode()) % 100000)

    if use_tuning:
        cfg = get_config_loader()
        base = dict(cfg.get("model_config", "lightgbm.base_params", {}))
        best, auc_s = search_lgb_on_val(
            X_r,
            y_r,
            X_val,
            np.asarray(y_val),
            base,
            spw,
            n_iter=int(n_tune_iter),
            seed=int(tune_seed),
        )
        if best:
            best_params = dict(best)
            n_estimators = int(best_params.get("n_estimators", 100))
            model = LightGBMWrapper(
                name=f"{tag}_{strategy}",
                use_imbalance=False,
                **{**base, **best_params, "scale_pos_weight": float(spw)},
            )
            model.fit(X_r, y_r, num_boost_round=n_estimators)
        else:
            model = LightGBMWrapper(name=f"{tag}_{strategy}", use_imbalance=False, **{**base, "scale_pos_weight": float(spw)})
            model.fit(X_r, y_r)
            best, auc_s = {}, float("nan")
        tune_ex = tuning_meta(best, auc_s)
    else:
        cfg = get_config_loader()
        base = dict(cfg.get("model_config", "lightgbm.base_params", {}))
        model = LightGBMWrapper(
            name=f"{tag}_{strategy}",
            use_imbalance=False,
            **{**base, "scale_pos_weight": float(spw)},
        )
        model.fit(X_r, y_r)
        tune_ex = {}

    y_proba_val = model.predict_proba(X_val)
    threshold, val_f1 = _select_threshold_from_validation(np.asarray(y_val), y_proba_val)

    y_t = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    metrics = compute_metrics(y_t, model.predict_proba(X_test), threshold=threshold)
    metrics.update(tune_ex)

    logger.info(
        f"    {tag:12s} {strategy:12s} [spw={spw:.3f}, thr={threshold:.3f}, valF1={val_f1:.4f}]: "
        f"AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}"
    )
    return metrics


def run_split(
    label: str,
    X_old: pd.DataFrame,
    y_old: pd.Series,
    year_old: pd.Series,
    X_new: pd.DataFrame,
    y_new: pd.Series,
    year_new: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logger,
    include_retrain: bool = True,
    *,
    use_tuning: bool = False,
    n_tune_iter: int = 48,
) -> list[dict]:
    sampler = ImbalanceSampler()
    rows: list[dict] = []

    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(
            X_old,
            y_old,
            X_test,
            y_test,
            sampler,
            strat,
            "Old",
            logger,
            year_train=year_old,
            split_label=label,
            use_tuning=use_tuning,
            n_tune_iter=n_tune_iter,
        )
        rows.append({"split": label, "method": "Old", "sampling": strat, **m})

    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(
            X_new,
            y_new,
            X_test,
            y_test,
            sampler,
            strat,
            "New",
            logger,
            year_train=year_new,
            split_label=label,
            use_tuning=use_tuning,
            n_tune_iter=n_tune_iter,
        )
        rows.append({"split": label, "method": "New", "sampling": strat, **m})

    if include_retrain:
        X_fit_re, y_fit_re, X_val_re, y_val_re = _build_retrain_fit_val(X_old, y_old, year_old, X_new, y_new, year_new)
        logger.info(f"  Retrain: fit={len(X_fit_re)} val={len(X_val_re)}")
        for strat in SAMPLING_STRATEGIES:
            m = _train_eval(
                X_fit_re,
                y_fit_re,
                X_test,
                y_test,
                sampler,
                strat,
                "Retrain",
                logger,
                X_val_raw=X_val_re,
                y_val=y_val_re,
                split_label=label,
                use_tuning=use_tuning,
                n_tune_iter=n_tune_iter,
            )
            rows.append({"split": label, "method": "Retrain", "sampling": strat, **m})

    return rows


def format_tables(df_raw: pd.DataFrame, logger, output_dir: Path) -> None:
    split_yr = {label: (old_end - BASE_YEAR + 1, TRAIN_END_YEAR - old_end) for label, old_end in MEDICAL_YEAR_SPLITS}

    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]

    split_labels = [label for label, _ in MEDICAL_YEAR_SPLITS]
    sampling_cols = SAMPLING_STRATEGIES

    for metric in METRICS:
        df_old = df_raw[df_raw["method"] == "Old"]
        pivot_old = df_old.pivot(index="split", columns="col", values=metric).reindex(
            index=split_labels, columns=sampling_cols
        )
        pivot_old.index = [f"{split_yr[s][0]}yr" for s in pivot_old.index]
        pivot_old["avg"] = pivot_old.mean(axis=1)
        pivot_old.loc["avg"] = pivot_old.mean()
        pivot_old.index.name = "old_years"
        out = output_dir / f"med_lgb_table_{metric}_old.csv"
        pivot_old.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        df_rt = df_raw[df_raw["method"] == "Retrain"]
        pivot_rt = df_rt.pivot_table(index="method", columns="col", values=metric, aggfunc="mean").reindex(
            columns=sampling_cols
        )
        pivot_rt.index = [f"full_{TOTAL_TRAIN_YEARS}yr"]
        pivot_rt["avg"] = pivot_rt.mean(axis=1)
        pivot_rt.index.name = "retrain_scope"
        out = output_dir / f"med_lgb_table_{metric}_retrain.csv"
        pivot_rt.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        df_new = df_raw[df_raw["method"] == "New"]
        pivot_new = df_new.pivot(index="split", columns="col", values=metric).reindex(
            index=split_labels, columns=sampling_cols
        )
        pivot_new.index = [f"{split_yr[s][1]}yr" for s in pivot_new.index]
        pivot_new["avg"] = pivot_new.mean(axis=1)
        pivot_new.loc["avg"] = pivot_new.mean()
        pivot_new.index.name = "new_years"
        out = output_dir / f"med_lgb_table_{metric}_new.csv"
        pivot_new.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")


def _mean_pivot_by_method_sampling(df_raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    t = (
        df_raw.groupby(["method", "sampling"])[metric]
        .mean()
        .unstack("sampling")
        .reindex(index=["Old", "New", "Retrain"], columns=SAMPLING_REPORT_ORDER)
    )
    return t.round(4)


def export_compact_report(df_raw: pd.DataFrame, logger, output_dir: Path) -> None:
    rows = []
    for m in COMPACT_SUMMARY_METRICS:
        if m not in df_raw.columns:
            continue
        pv = _mean_pivot_by_method_sampling(df_raw, m).reset_index()
        pv.insert(0, "metric", m)
        rows.append(pv)

    if rows:
        out = output_dir / "med_lgb_compact_summary.csv"
        pd.concat(rows, ignore_index=True).to_csv(out, index=False, float_format="%.4f")
        logger.info(f"Saved -> {out.name}")

    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pv = _mean_pivot_by_method_sampling(df_raw, m)
        out = output_dir / f"med_lgb_compact_{m}_only.csv"
        pv.to_csv(out, float_format="%.4f")
        logger.info(f"Saved -> {out.name}")


def _run_one_output_dir(
    output_dir: Path,
    *,
    use_tuning: bool,
    n_tune_iter: int,
    logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for split_idx, (label, old_end_year) in enumerate(MEDICAL_YEAR_SPLITS):
        include_retrain = split_idx == 0
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Split: {label}  (Old<={old_end_year}, New={old_end_year+1}-{TRAIN_END_YEAR}, Test=2007-2008)"
        )
        logger.info("=" * 60)
        try:
            (
                X_old,
                y_old,
                X_new,
                y_new,
                X_test,
                y_test,
                year_old,
                year_new,
                _year_test,
            ) = get_medical_year_split(logger, old_end_year=old_end_year, return_years=True)

            rows = run_split(
                label,
                X_old,
                y_old,
                year_old,
                X_new,
                y_new,
                year_new,
                X_test,
                y_test,
                logger,
                include_retrain=include_retrain,
                use_tuning=use_tuning,
                n_tune_iter=int(n_tune_iter),
            )
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            import traceback

            logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果，請確認醫療資料檔存在。")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = output_dir / "medical_year_splits_lgb_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")

    logger.info("\n產出指標 pivot 表格...")
    format_tables(df_raw, logger, output_dir)

    logger.info("\n輸出 compact summary...")
    export_compact_report(df_raw, logger, output_dir)

    if use_tuning:
        export_tuning_log(df_raw, output_dir / "med_lgb_tuning_log.csv", logger)

    logger.info("\n=== 完成 ===")
    summary = _mean_pivot_by_method_sampling(df_raw, "AUC")
    logger.info("\nAUC 摘要（method × sampling 平均，跨切割）:\n" + summary.to_string())


def main() -> None:
    p = argparse.ArgumentParser(description="Phase1 medical year splits — LightGBM baseline (scale_pos_weight)")
    p.add_argument(
        "--tuning",
        type=str,
        default="default",
        choices=["default", "tuned", "both"],
        help="default: base params; tuned: val-AUC tuning; both: run both",
    )
    p.add_argument("--tune-n-iter", type=int, default=48, help="random candidates for tuning")
    p.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="結果輸出子資料夾標籤（避免覆蓋）。會寫到 results/phase1_baseline/lgbm/<tag>/ 或 <tag>/tuned/",
    )
    args = p.parse_args()

    logger = get_logger("Medical_YearSplits_LGB", console=True, file=True)
    set_seed(42)

    tag = args.output_tag.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_output_dir = OUTPUT_DIR / tag

    if args.tuning == "default":
        runs = [(False, base_output_dir)]
    elif args.tuning == "tuned":
        runs = [(True, base_output_dir / "tuned")]
    else:
        runs = [(False, base_output_dir), (True, base_output_dir / "tuned")]

    for use_tune, out_dir in runs:
        mode = "validation AUC 調參" if use_tune else "預設超參數"
        logger.info(f"\n{'#'*60}\n模式: {mode}\n輸出: {out_dir}\n{'#'*60}")
        _run_one_output_dir(out_dir, use_tuning=use_tune, n_tune_iter=int(args.tune_n_iter), logger=logger)


if __name__ == "__main__":
    main()
