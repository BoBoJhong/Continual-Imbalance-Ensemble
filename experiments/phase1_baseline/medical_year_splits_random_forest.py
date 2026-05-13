"""
Phase 1 - Medical 年份切割基準線實驗（RandomForest）
==========================================================
資料集：UCI Diabetes 130-US Hospitals (1999-2008，偽時序)

固定 Test = 2007-2008；訓練窗 1999–2006（8 年）。依 `experiments._shared.common_dataset.MEDICAL_YEAR_SPLITS`
逐年滑動 old_end，共 7 組 Old/New 切割（split_1+7 … split_7+1）。

訓練策略（與 XGB 規格對齊）：
  - Old      : 只用 Old 資料訓練
  - New      : 只用 New 資料訓練
  - Retrain  : Old+New 全量合併訓練（Re-training baseline；整個實驗只跑一次）

採樣策略：none / undersampling / oversampling / hybrid
Validation：訓練段內 **依 year 逐年**各抽約 20% 作 validation（_split_fit_val_by_year）。

輸出（default 與 tuned 各一套）：
  - medical_year_splits_rf_raw.csv
  - med_rf_compact_summary.csv
  - med_rf_compact_{AUC|F1|Recall}_only.csv
  - med_rf_table_{metric}_{old|retrain|new}.csv
  - med_rf_tuning_log.csv（若有 tuning）

用法（預設只跑調參 tuned；未調參請加 --tuning default）：
  python experiments/phase1_baseline/medical_year_splits_random_forest.py
  python experiments/phase1_baseline/medical_year_splits_random_forest.py --tune-n-iter 48
  python experiments/phase1_baseline/medical_year_splits_random_forest.py --tuning default
  python experiments/phase1_baseline/medical_year_splits_random_forest.py --tuning both --tune-n-iter 48
  python experiments/phase1_baseline/medical_year_splits_random_forest.py --results-subdir med_rf_run_20260201

  （`--output-tag` 在未指定 `--results-subdir` 時等同子資料夾名稱，與舊版相容。）
"""

from __future__ import annotations

import argparse
import gc
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

from experiments._shared.baseline_val_search import (  # noqa: E402
    export_tuning_log,
    rf_wrapper_kwargs_from_best,
    search_rf_on_val,
    tuning_meta,
)
from experiments._shared.common_dataset import MEDICAL_YEAR_SPLITS, get_medical_year_split  # noqa: E402
from src.data import DataPreprocessor, ImbalanceSampler  # noqa: E402
from src.evaluation import compute_metrics  # noqa: E402
from src.models import RandomForestWrapper  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402


SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]
METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]

OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "random_forest"


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
    n_tune_iter: int = 0,
    rf_n_jobs: int = -1,
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
            model = RandomForestWrapper(name=f"{tag}_{strategy}", n_estimators=n_est, n_jobs=rf_n_jobs, **wkw)
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
    rf_n_jobs: int = -1,
) -> list[dict]:
    """對一組切割跑 Old/New（與可選 Retrain）；含 Retrain 時 12 列，否則 8 列。"""
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
                rf_n_jobs=rf_n_jobs,
            )
            rows.append({"split": label, "method": "Retrain", "sampling": strat, **m})

    return rows


def format_tables(df_raw: pd.DataFrame, logger, output_dir: Path) -> None:
    """Old/New 表：列 = 各 split；Retrain 表為單列（跨唯一 Retrain 列聚合平均）。"""
    split_labels = [label for label, _ in MEDICAL_YEAR_SPLITS]

    # --- pivot tables ---
    for metric in METRICS:
        df_old = df_raw[df_raw["method"] == "Old"]
        pivot_old = (
            df_old.pivot_table(index="split", columns="sampling", values=metric, aggfunc="mean")
            .reindex(index=split_labels, columns=SAMPLING_REPORT_ORDER)
            .round(6)
        )
        out = output_dir / f"med_rf_table_{metric}_old.csv"
        pivot_old.to_csv(out)
        logger.info(f"[SAVE] {out}")

        df_new = df_raw[df_raw["method"] == "New"]
        pivot_new = (
            df_new.pivot_table(index="split", columns="sampling", values=metric, aggfunc="mean")
            .reindex(index=split_labels, columns=SAMPLING_REPORT_ORDER)
            .round(6)
        )
        out = output_dir / f"med_rf_table_{metric}_new.csv"
        pivot_new.to_csv(out)
        logger.info(f"[SAVE] {out}")

        df_rt = df_raw[df_raw["method"] == "Retrain"]
        if len(df_rt) > 0:
            pivot_rt = (
                df_rt.pivot_table(index="method", columns="sampling", values=metric, aggfunc="mean")
                .reindex(index=["Retrain"], columns=SAMPLING_REPORT_ORDER)
                .round(6)
            )
            pivot_rt.index.name = "retrain_scope"
            out = output_dir / f"med_rf_table_{metric}_retrain.csv"
            pivot_rt.to_csv(out)
            logger.info(f"[SAVE] {out}")

    # --- compact summary ---
    summary = (
        df_raw[df_raw["method"].isin(["Old", "New", "Retrain"])]
        .pivot_table(index="method", columns="sampling", values=COMPACT_SUMMARY_METRICS, aggfunc="mean")
        .reindex(index=["Old", "New", "Retrain"])
    )
    # flatten MultiIndex columns: (metric, sampling) -> metric__sampling
    summary.columns = [f"{m}__{s}" for m, s in summary.columns]
    out = output_dir / "med_rf_compact_summary.csv"
    summary.round(6).to_csv(out)
    logger.info(f"[SAVE] {out}")

    for m in COMPACT_ONLY_METRICS:
        one = (
            df_raw[df_raw["method"].isin(["Old", "New", "Retrain"])]
            .pivot_table(index="method", columns="sampling", values=m, aggfunc="mean")
            .reindex(index=["Old", "New", "Retrain"], columns=SAMPLING_REPORT_ORDER)
            .round(6)
        )
        out = output_dir / f"med_rf_compact_{m}_only.csv"
        one.to_csv(out)
        logger.info(f"[SAVE] {out}")


def _run(
    tuning_mode: str,
    n_tune_iter: int,
    *,
    results_subdir: str | None = None,
    resume: bool = False,
    splits_filter: list[str] | None = None,
    rf_n_jobs: int = 1,
) -> None:
    logger = get_logger("MED_YearSplits_RF", console=True, file=True)
    set_seed(42)

    results_subdir = (results_subdir or "").strip()
    base = OUTPUT_DIR / results_subdir if results_subdir else OUTPUT_DIR
    out_dir = base / "tuned" if tuning_mode == "tuned" else base
    out_dir.mkdir(parents=True, exist_ok=True)

    use_tuning = tuning_mode == "tuned"
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Tuning: {tuning_mode} (n_iter={n_tune_iter})")

    raw_path = out_dir / "medical_year_splits_rf_raw.csv"
    completed: set[tuple] = set()
    all_rows: list[dict] = []
    if resume and raw_path.exists():
        df0 = pd.read_csv(raw_path)
        all_rows = df0.to_dict("records")
        completed = {(r.get("split"), r.get("method"), r.get("sampling")) for r in all_rows}
        logger.info(f"[RESUME] loaded {len(all_rows)} rows from {raw_path}")

    split_set = set(splits_filter) if splits_filter else None
    retrain_done = False

    for label, old_end_year in MEDICAL_YEAR_SPLITS:
        if split_set is not None and label not in split_set:
            continue

        logger.info("\n" + "=" * 60)
        logger.info(f"Split: {label}  (Old<= {old_end_year}, New= {old_end_year + 1}-2006, Test=2007-2008)")
        logger.info("=" * 60)

        X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, _ = get_medical_year_split(
            logger, old_end_year=old_end_year, return_years=True
        )

        include_retrain = not retrain_done
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
            rf_n_jobs=int(rf_n_jobs),
        )
        if include_retrain:
            retrain_done = True

        for r in rows:
            k = (r.get("split"), r.get("method"), r.get("sampling"))
            if k in completed:
                continue
            all_rows.append(r)
            completed.add(k)

        df_raw = pd.DataFrame(all_rows)
        df_raw.to_csv(raw_path, index=False)
        logger.info(f"[SAVE] snapshot -> {raw_path}  ({len(df_raw)} rows)")

        gc.collect()

    if not all_rows:
        logger.warning("No rows produced (check --splits filter).")
        return

    df_raw = pd.DataFrame(all_rows)
    format_tables(df_raw, logger, out_dir)
    if use_tuning:
        export_tuning_log(df_raw, out_dir / "med_rf_tuning_log.csv", logger)

    logger.info("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 medical year splits — RandomForest baseline")
    parser.add_argument(
        "--tuning",
        choices=["default", "tuned", "both"],
        default="tuned",
        help="預設 tuned（validation AUC 調參）；default=未調參；both=兩者都跑（較耗時）",
    )
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=48,
        help="RF validation AUC 隨機搜尋候選數（<=0 時改用 48）",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="",
        help="寫入 results/phase1_baseline/random_forest/<subdir>/（與醫療 XGB 的 --results-subdir 相同概念）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若輸出目錄已有 medical_year_splits_rf_raw.csv，載入已完成的 split 並只跑剩餘分割",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="只跑指定 split labels，例如：--splits split_1+7 split_2+6",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=1,
        help="sklearn RandomForest 的 n_jobs（預設 1 以降低記憶體峰值；記憶體充足可用 -1 全核心）",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="未指定 --results-subdir 時，作為子資料夾名稱（舊版相容）",
    )
    args = parser.parse_args()

    sub = (args.results_subdir or "").strip() or (args.output_tag or "").strip() or None
    n_iter = int(args.tune_n_iter) if int(args.tune_n_iter) > 0 else 48
    splits = list(args.splits) if args.splits else None

    if args.tuning == "both":
        _run("default", n_iter, results_subdir=sub, resume=args.resume, splits_filter=splits, rf_n_jobs=args.rf_n_jobs)
        _run("tuned", n_iter, results_subdir=sub, resume=args.resume, splits_filter=splits, rf_n_jobs=args.rf_n_jobs)
    else:
        _run(str(args.tuning), n_iter, results_subdir=sub, resume=args.resume, splits_filter=splits, rf_n_jobs=args.rf_n_jobs)


if __name__ == "__main__":
    main()

