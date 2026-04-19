"""
Phase 1 - Medical 年份切割基準線實驗（XGBoost）
=======================================================
資料集：UCI Diabetes 130-US Hospitals (1999-2008，偽時序)

固定 Test = 2007-2008；訓練窗 1999–2006（8 年）。依 `experiments._shared.common_dataset.MEDICAL_YEAR_SPLITS`
逐年滑動 old_end，共 7 組 Old/New 切割（split_1+7 … split_7+1）。

訓練策略（與 bankruptcy baseline 流程對齊）：
  - Old      : 只用 Old 資料訓練
  - New      : 只用 New 資料訓練
  - Retrain  : Old+New「全量」合併訓練（Re-training baseline；整個實驗只跑一次）

採樣策略：none / undersampling / oversampling / hybrid

最終輸出（default 與 tuned 各一套；tuned 會寫到 results/phase1_baseline/xgb/tuned/）：
  - medical_year_splits_xgb_raw.csv
  - med_xgb_compact_summary.csv
  - med_xgb_compact_{AUC|F1|Recall}_only.csv
  - med_xgb_table_{metric}_{old|retrain|new}.csv
  - med_xgb_tuning_log.csv（若有 tuning）

用法：
  python experiments/phase1_baseline/medical_year_splits_xgb.py
  python experiments/phase1_baseline/medical_year_splits_xgb.py --tuning tuned --tune-n-iter 48
  python experiments/phase1_baseline/medical_year_splits_xgb.py --tuning both --tune-n-iter 48
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataPreprocessor, ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_config_loader, get_logger, set_seed

from experiments._shared.baseline_val_search import (
    export_tuning_log,
    search_xgb_on_val,
    tuning_meta,
)
from experiments._shared.common_dataset import MEDICAL_YEAR_SPLITS, get_medical_year_split

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
# 精簡報告欄位順序（與常見論文表格一致：hybrid / none / over / under）
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]

METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]

OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "xgb"

BASE_YEAR = 1999
TRAIN_END_YEAR = 2006
TOTAL_TRAIN_YEARS = TRAIN_END_YEAR - BASE_YEAR + 1

# 精簡摘要檔包含的指標（全指標仍寫在 raw 與各 med_xgb_table_*）
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
# 各寫一份 method×採樣 小表（方便貼簡報）
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]


def _select_threshold_from_validation(y_val: np.ndarray, y_proba_val: np.ndarray) -> tuple[float, float]:
    """用 validation set 搜尋最佳 F1 閾值，避免固定規則。"""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def _split_fit_val(
    X_raw: pd.DataFrame, y_raw: np.ndarray, *, test_size: float = 0.2, random_state: int = 42
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
    """依 year 分層切 validation：每個年分抽約 20% 當 val。"""
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
    """先切 train/val，再只用 train fold fit scaler，避免 validation leakage。"""
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
    tune_seed = 42 + (zlib.adler32(f"{split_label}|{tag}|{strategy}".encode()) % 100000)

    if use_tuning:
        cfg = get_config_loader()
        base = dict(cfg.get("model_config", "xgboost.base_params", {}))
        spw = _scale_pos_weight(y_r)
        best, auc_s = search_xgb_on_val(
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
            model = XGBoostWrapper(
                name=f"{tag}_{strategy}",
                use_imbalance=False,
                **{**base, **best, "scale_pos_weight": spw},
            )
        else:
            model = XGBoostWrapper(name=f"{tag}_{strategy}")
            best, auc_s = {}, float("nan")
        model.fit(X_r, y_r)
        tune_ex = tuning_meta(best, auc_s)
    else:
        model = XGBoostWrapper(name=f"{tag}_{strategy}")
        model.fit(X_r, y_r)
        tune_ex = {}

    y_proba_val = model.predict_proba(X_val)
    threshold, val_f1 = _select_threshold_from_validation(np.asarray(y_val), y_proba_val)

    y_t = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    metrics = compute_metrics(y_t, model.predict_proba(X_test), threshold=threshold)
    metrics.update(tune_ex)
    logger.info(
        f"    {tag:12s} {strategy:12s} [thr={threshold:.3f}, valF1={val_f1:.4f}]: "
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
    """對一組切割跑 Old/New（與可選 Retrain）；含 Retrain 時 12 列，否則 8 列。"""
    sampler = ImbalanceSampler()
    rows: list[dict] = []

    # Old
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

    # New
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
        X_fit_re, y_fit_re, X_val_re, y_val_re = _build_retrain_fit_val(
            X_old,
            y_old,
            year_old,
            X_new,
            y_new,
            year_new,
        )
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
    """
    Old / New 表：列 = 各 split；Retrain 表為單列（跨唯一 Retrain 列聚合平均）。
    """
    split_yr = {label: (old_end - BASE_YEAR + 1, TRAIN_END_YEAR - old_end) for label, old_end in MEDICAL_YEAR_SPLITS}

    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]

    split_labels = [label for label, _ in MEDICAL_YEAR_SPLITS]
    sampling_cols = SAMPLING_STRATEGIES

    for metric in METRICS:
        # Old
        df_old = df_raw[df_raw["method"] == "Old"]
        pivot_old = df_old.pivot(index="split", columns="col", values=metric).reindex(
            index=split_labels, columns=sampling_cols
        )
        pivot_old.index = [f"{split_yr[s][0]}yr" for s in pivot_old.index]
        pivot_old["avg"] = pivot_old.mean(axis=1)
        pivot_old.loc["avg"] = pivot_old.mean()
        pivot_old.index.name = "old_years"
        out = output_dir / f"med_xgb_table_{metric}_old.csv"
        pivot_old.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        # Retrain（full 8yr 單列）
        df_rt = df_raw[df_raw["method"] == "Retrain"]
        pivot_rt = df_rt.pivot_table(index="method", columns="col", values=metric, aggfunc="mean").reindex(
            columns=sampling_cols
        )
        pivot_rt.index = [f"full_{TOTAL_TRAIN_YEARS}yr"]
        pivot_rt["avg"] = pivot_rt.mean(axis=1)
        pivot_rt.index.name = "retrain_scope"
        out = output_dir / f"med_xgb_table_{metric}_retrain.csv"
        pivot_rt.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        # New
        df_new = df_raw[df_raw["method"] == "New"]
        pivot_new = df_new.pivot(index="split", columns="col", values=metric).reindex(
            index=split_labels, columns=sampling_cols
        )
        pivot_new.index = [f"{split_yr[s][1]}yr" for s in pivot_new.index]
        pivot_new["avg"] = pivot_new.mean(axis=1)
        pivot_new.loc["avg"] = pivot_new.mean()
        pivot_new.index.name = "new_years"
        out = output_dir / f"med_xgb_table_{metric}_new.csv"
        pivot_new.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")


def _mean_pivot_by_method_sampling(df_raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """method × sampling，數值為跨各 split 之平均；欄位順序：hybrid, none, oversampling, undersampling。"""
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
        out = output_dir / "med_xgb_compact_summary.csv"
        pd.concat(rows, ignore_index=True).to_csv(out, index=False, float_format="%.4f")
        logger.info(f"Saved -> {out.name}")

    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pv = _mean_pivot_by_method_sampling(df_raw, m)
        out = output_dir / f"med_xgb_compact_{m}_only.csv"
        pv.to_csv(out, float_format="%.4f")
        logger.info(f"Saved -> {out.name}")

    if "tune_best_params" in df_raw.columns or "tune_val_auc" in df_raw.columns:
        try:
            out = output_dir / "med_xgb_tuning_log.csv"
            export_tuning_log(df_raw, out)
            logger.info(f"Saved -> {out.name}")
        except Exception as e:
            logger.warning(f"export_tuning_log failed: {e}")


def _run(tuning_mode: str, n_tune_iter: int, *, results_subdir: str | None = None) -> None:
    logger = get_logger("Medical_YearSplits_XGB", console=True, file=True)
    set_seed(42)

    # 預設：tuned → OUTPUT_DIR/tuned；default → OUTPUT_DIR
    # 若指定 results_subdir：一律輸出到 OUTPUT_DIR/results_subdir（避免覆蓋既有結果）
    results_subdir = (results_subdir or "").strip()
    if results_subdir:
        out_dir = OUTPUT_DIR / results_subdir
    else:
        out_dir = OUTPUT_DIR / "tuned" if tuning_mode == "tuned" else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    use_tuning = tuning_mode == "tuned"
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Tuning: {tuning_mode} (n_iter={n_tune_iter})")

    all_rows: list[dict] = []

    for split_idx, (label, old_end_year) in enumerate(MEDICAL_YEAR_SPLITS):
        include_retrain = split_idx == 0  # Retrain(ALL) 只跑一次
        logger.info(f"\n{'='*60}")
        logger.info(f"Split: {label}  (Old<={old_end_year}, New={old_end_year+1}-{TRAIN_END_YEAR}, Test=2007-2008)")
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
    raw_path = out_dir / "medical_year_splits_xgb_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")

    logger.info("\n產出指標 pivot 表格...")
    format_tables(df_raw, logger, out_dir)

    logger.info("\n輸出 compact summary...")
    export_compact_report(df_raw, logger, out_dir)

    logger.info("\n=== 完成 ===")
    summary = (
        df_raw.groupby(["method", "sampling"])["AUC"]
        .mean()
        .unstack("sampling")
        .reindex(["Old", "New", "Retrain"])
    )
    logger.info("\nAUC 摘要（method × sampling 平均，跨切割）:\n" + summary.to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tuning",
        type=str,
        default="default",
        choices=["default", "tuned", "both"],
        help="default: base params; tuned: val-AUC tuning; both: run both",
    )
    p.add_argument("--tune-n-iter", type=int, default=48, help="random candidates for tuning")
    p.add_argument(
        "--results-subdir",
        type=str,
        default="",
        help="Write outputs into results/phase1_baseline/xgb/<results-subdir>/ (avoid overwriting).",
    )
    args = p.parse_args()

    if args.tuning == "both":
        _run("default", int(args.tune_n_iter), results_subdir=args.results_subdir or None)
        _run("tuned", int(args.tune_n_iter), results_subdir=args.results_subdir or None)
    else:
        _run(str(args.tuning), int(args.tune_n_iter), results_subdir=args.results_subdir or None)


if __name__ == "__main__":
    main()

