"""
Phase 1 - Bankruptcy 年份切割基準線實驗（RandomForest）
==========================================================
固定 Test = 2015-2018；訓練窗 1999–2014。依 `YEAR_SPLITS` 共 **15 組** Old/New 切割；validation 為
**逐年**各抽約 20% 後合併。Retrain 僅第一次迭代跑一次，跑滿全部分割時 raw **124** 列。

訓練策略（對齊 XGB 規格）：
  - Old
  - New
  - Retrain（Old+New 全量，僅跑一次）

採樣策略：none / undersampling / oversampling / hybrid
輸出：raw + pivot tables + compact summaries（與 xgb 腳本同規格）

超參數調整：python bankruptcy_year_splits_random_forest.py --tuning both --tune-n-iter 48
  （--tuning default|tuned|both；RF 於多維網格隨機搜尋 validation AUC，結果在 random_forest/tuned/）
"""
import argparse
import gc
import sys
import zlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler, DataPreprocessor
from src.models import RandomForestWrapper
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split
from experiments._shared.baseline_val_search import (
    export_tuning_log,
    rf_wrapper_kwargs_from_best,
    search_rf_on_val,
    tuning_meta,
)

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]
METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]
OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "random_forest"


def _select_threshold_from_validation(y_val, y_proba_val):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def _split_fit_val(X_raw, y_raw, test_size=0.2, random_state=42):
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


def _split_fit_val_by_year(X_raw, y_raw, year_arr, val_ratio=0.2, random_state=42):
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


def _build_retrain_fit_val(X_old, y_old, year_old, X_new, y_new, year_new):
    X_old_fit, y_old_fit, X_old_val, y_old_val = _split_fit_val_by_year(X_old, y_old, year_old)
    X_new_fit, y_new_fit, X_new_val, y_new_val = _split_fit_val_by_year(X_new, y_new, year_new)

    X_fit = pd.concat([X_old_fit, X_new_fit], ignore_index=True)
    y_fit = np.concatenate([y_old_fit, y_new_fit])
    X_val = pd.concat([X_old_val, X_new_val], ignore_index=True)
    y_val = np.concatenate([y_old_val, y_new_val])
    return X_fit, y_fit, X_val, y_val


def _train_eval(
    X_train_raw,
    y_train,
    X_test_raw,
    y_test,
    sampler,
    strategy,
    tag,
    logger,
    X_val_raw=None,
    y_val=None,
    year_train=None,
    *,
    split_label: str = "",
    use_tuning: bool = False,
    n_tune_iter: int = 0,
    rf_n_jobs: int = -1,
):
    y_train_arr = np.asarray(y_train)

    if X_val_raw is None or y_val is None:
        if year_train is not None:
            X_fit_raw, y_fit, X_val_raw, y_val = _split_fit_val_by_year(
                X_train_raw,
                y_train_arr,
                year_train,
            )
        else:
            X_fit_raw, X_val_raw, y_fit, y_val = _split_fit_val(X_train_raw, y_train_arr)
    else:
        X_fit_raw = X_train_raw
        y_fit = y_train_arr

    pre = DataPreprocessor()
    X_fit, X_val = pre.scale_features(X_fit_raw, X_val_raw, fit=True)
    _, X_test = pre.scale_features(X_fit_raw, X_test_raw, fit=False)

    X_r, y_r = sampler.apply_sampling(X_fit, np.asarray(y_fit), strategy=strategy)
    tune_seed = 42 + (zlib.adler32(f"{split_label}|{tag}|{strategy}".encode()) % 100000)

    if use_tuning:
        n_iter = int(n_tune_iter) if int(n_tune_iter) > 0 else 48
        best, auc_s = search_rf_on_val(
            X_r,
            y_r,
            X_val,
            np.asarray(y_val),
            n_iter=n_iter,
            seed=tune_seed,
            n_jobs=rf_n_jobs,
        )
        if best:
            wkw = rf_wrapper_kwargs_from_best(best)
            n_est = int(wkw.pop("n_estimators", 200))
            model = RandomForestWrapper(
                name=f"{tag}_{strategy}", n_estimators=n_est, n_jobs=rf_n_jobs, **wkw
            )
        else:
            model = RandomForestWrapper(name=f"{tag}_{strategy}", n_jobs=rf_n_jobs)
            best, auc_s = {}, float("nan")
        model.fit(X_r, y_r)
        tune_ex = tuning_meta(best, auc_s)
    else:
        model = RandomForestWrapper(name=f"{tag}_{strategy}", n_jobs=rf_n_jobs)
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
    include_retrain=True,
    *,
    use_tuning: bool = False,
    n_tune_iter: int = 0,
    rf_n_jobs: int = -1,
):
    sampler = ImbalanceSampler()
    rows = []

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
            rf_n_jobs=rf_n_jobs,
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
            rf_n_jobs=rf_n_jobs,
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
                rf_n_jobs=rf_n_jobs,
            )
            rows.append({"split": label, "method": "Retrain", "sampling": strat, **m})

    return rows


def format_tables(df_raw, logger, output_dir: Path):
    split_yr = {label: (old_end - 1998, 2014 - old_end) for label, old_end in YEAR_SPLITS}

    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]

    split_labels = [label for label, _ in YEAR_SPLITS]
    sampling_cols = SAMPLING_STRATEGIES

    for metric in METRICS:
        df_old = df_raw[df_raw["method"] == "Old"]
        pivot_old = (
            df_old.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_old.index = [f"{split_yr[s][0]}yr" for s in pivot_old.index]
        pivot_old["avg"] = pivot_old.mean(axis=1)
        pivot_old.loc["avg"] = pivot_old.mean()
        pivot_old.index.name = "old_years"
        out = output_dir / f"bk_rf_table_{metric}_old.csv"
        pivot_old.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        df_rt = df_raw[df_raw["method"] == "Retrain"]
        pivot_rt = (
            df_rt.pivot_table(index="method", columns="col", values=metric, aggfunc="mean")
            .reindex(columns=sampling_cols)
        )
        pivot_rt.index = ["full_16yr"]
        pivot_rt["avg"] = pivot_rt.mean(axis=1)
        pivot_rt.index.name = "retrain_scope"
        out = output_dir / f"bk_rf_table_{metric}_retrain.csv"
        pivot_rt.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        df_new = df_raw[df_raw["method"] == "New"]
        pivot_new = (
            df_new.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_new.index = [f"{split_yr[s][1]}yr" for s in pivot_new.index]
        pivot_new["avg"] = pivot_new.mean(axis=1)
        pivot_new.loc["avg"] = pivot_new.mean()
        pivot_new.index.name = "new_years"
        out = output_dir / f"bk_rf_table_{metric}_new.csv"
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
    if not rows:
        return

    long_df = pd.concat(rows, ignore_index=True)
    path_all = output_dir / "bk_rf_compact_summary.csv"
    long_df.to_csv(path_all, index=False, float_format="%.4f")
    logger.info(f"  Saved -> {path_all.name}")

    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pivot = _mean_pivot_by_method_sampling(df_raw, m)
        path_m = output_dir / f"bk_rf_compact_{m}_only.csv"
        pivot.to_csv(path_m, float_format="%.4f")
        logger.info(f"  Saved -> {path_m.name}")


def _run_one_output_dir(
    output_dir: Path,
    *,
    use_tuning: bool,
    n_tune_iter: int,
    logger,
    resume: bool = False,
    rf_n_jobs: int = -1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "bankruptcy_year_splits_rf_raw.csv"
    all_rows: list = []
    retrain_done = False
    completed_splits: set[str] = set()

    if resume and raw_path.exists():
        try:
            df_prev = pd.read_csv(raw_path)
            if "split" in df_prev.columns and len(df_prev) > 0:
                completed_splits = set(df_prev["split"].astype(str).unique())
                all_rows = df_prev.to_dict("records")
                if "method" in df_prev.columns:
                    retrain_done = bool((df_prev["method"] == "Retrain").any())
                logger.info(
                    f"Resume：已載入 {len(all_rows)} 列，略過 split: {sorted(completed_splits)}"
                )
        except Exception as e:
            logger.warning(f"Resume 讀取失敗（{e}），改為從頭跑。")
            all_rows = []
            completed_splits = set()
            retrain_done = False

    def _write_raw_checkpoint() -> None:
        """Persist current progress so --resume can continue safely after interruptions."""
        if not all_rows:
            return
        pd.DataFrame(all_rows).to_csv(raw_path, index=False, float_format="%.6f")

    for label, old_end_year in YEAR_SPLITS:
        if label in completed_splits:
            logger.info(f"\n{'='*60}\nSplit: {label}  [Resume 略過]\n{'='*60}")
            continue
        logger.info(f"\n{'='*60}")
        logger.info(f"Split: {label}  (Old<={old_end_year}, New={old_end_year + 1}-2014, Test=2015-2018)")
        logger.info("="*60)
        try:
            X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, year_test = get_bankruptcy_year_split(
                logger,
                old_end_year=old_end_year,
                return_years=True,
            )
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
                include_retrain=(not retrain_done),
                use_tuning=use_tuning,
                n_tune_iter=n_tune_iter if use_tuning else 0,
                rf_n_jobs=rf_n_jobs,
            )
            all_rows.extend(rows)
            retrain_done = True
            # Save after each finished split; if process is interrupted, --resume loses at most current split.
            _write_raw_checkpoint()
            gc.collect()
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果，請確認資料檔存在。")
        return

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")

    logger.info("\n產出指標 pivot 表格...")
    format_tables(df_raw, logger, output_dir)

    logger.info("\n產出精簡摘要（跨各 split 平均）...")
    export_compact_report(df_raw, logger, output_dir)

    if use_tuning:
        export_tuning_log(df_raw, output_dir / "bk_rf_tuning_log.csv", logger)

    logger.info("\n=== 完成 ===")


def main():
    parser = argparse.ArgumentParser(description="Phase1 bankruptcy year splits — RandomForest baseline")
    parser.add_argument("--tuning", choices=["default", "tuned", "both"], default="default")
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=48,
        help="RF validation AUC 隨機搜尋候選數（僅 --tuning tuned/both 時使用；<=0 時改用 48）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若輸出目錄已有 bankruptcy_year_splits_rf_raw.csv，載入已完成的 split 並只跑剩餘分割",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=1,
        help="sklearn RandomForest 的 n_jobs（預設 1 以降低記憶體峰值；記憶體充足可用 -1 全核心）",
    )
    args = parser.parse_args()

    logger = get_logger("BK_YearSplits_RF", console=True, file=True)
    set_seed(42)

    if args.tuning == "default":
        runs = [(False, OUTPUT_DIR)]
    elif args.tuning == "tuned":
        runs = [(True, OUTPUT_DIR / "tuned")]
    else:
        runs = [(False, OUTPUT_DIR), (True, OUTPUT_DIR / "tuned")]

    for use_tune, out_dir in runs:
        tag = "validation AUC 調參" if use_tune else "預設超參數"
        logger.info(f"\n{'#'*60}\n模式: {tag}\n輸出: {out_dir}\n{'#'*60}")
        _run_one_output_dir(
            out_dir,
            use_tuning=use_tune,
            n_tune_iter=args.tune_n_iter,
            logger=logger,
            resume=args.resume,
            rf_n_jobs=args.rf_n_jobs,
        )


if __name__ == "__main__":
    main()
