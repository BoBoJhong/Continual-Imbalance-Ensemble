"""
Phase 1 - Medical 年份切割基準線實驗（LogisticRegression）
==========================================================
資料集：UCI Diabetes 130-US Hospitals (1999-2008，偽時序)

固定 Test = 2007-2008；訓練窗 1999–2006。依 `MEDICAL_YEAR_SPLITS` 共 7 組 Old/New 切割；validation 為
訓練段內 **逐年**各抽約 20%。Retrain（Old+New 全量）僅在第一個 split 跑一次（與 `medical_year_splits_xgb.py` 一致）。

採樣策略：none / undersampling / oversampling / hybrid

輸出（與醫療 XGB 同檔名規格，前綴 med_lr）：
  - medical_year_splits_lr_raw.csv
  - med_lr_compact_summary.csv
  - med_lr_compact_{AUC|F1|Recall}_only.csv
  - med_lr_table_{metric}_{old|retrain|new}.csv
  - med_lr_tuning_log.csv（tuned 模式且 export_compact 偵測到調參欄位時）

用法（預設只跑調參 tuned，省時間；要跑未調參請加 --tuning default）：
  python experiments/phase1_baseline/medical_year_splits_logistic_regression.py
  python experiments/phase1_baseline/medical_year_splits_logistic_regression.py --tune-n-iter 48
  python experiments/phase1_baseline/medical_year_splits_logistic_regression.py --tuning default
  python experiments/phase1_baseline/medical_year_splits_logistic_regression.py --tuning both --tune-n-iter 48
  python experiments/phase1_baseline/medical_year_splits_logistic_regression.py --results-subdir med_lr_run_20260201
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

from experiments._shared.baseline_val_search import export_tuning_log, search_lr_on_val, tuning_meta
from experiments._shared.common_dataset import MEDICAL_YEAR_SPLITS, get_medical_year_split
from src.data import DataPreprocessor, ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import LogisticRegressionWrapper
from src.utils import get_config_loader, get_logger, set_seed

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]
METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]
OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "logistic_regression"

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

    tune_seed = 42 + (zlib.adler32(f"{split_label}|{tag}|{strategy}".encode()) % 100000)
    tune_ex: dict = {}
    if use_tuning:
        cfg = get_config_loader()
        base = dict(cfg.get("model_config", "logistic_regression.base_params", {}))
        imb = dict(cfg.get("model_config", "logistic_regression.imbalance_params", {}))
        base_params = {**base, **imb}

        best, auc_s = search_lr_on_val(
            X_r,
            np.asarray(y_r),
            X_val,
            np.asarray(y_val),
            base_params,
            n_iter=n_tune_iter,
            seed=tune_seed,
        )
        model = LogisticRegressionWrapper(
            name=f"{tag}_{strategy}",
            **{**base_params, **best},
        )
        tune_ex = tuning_meta(best, auc_s)
    else:
        model = LogisticRegressionWrapper(name=f"{tag}_{strategy}")

    model.fit(X_r, y_r)

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
    y_old: pd.Series | np.ndarray,
    year_old: pd.Series | np.ndarray,
    X_new: pd.DataFrame,
    y_new: pd.Series | np.ndarray,
    year_new: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
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
        out = output_dir / f"med_lr_table_{metric}_old.csv"
        pivot_old.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        df_rt = df_raw[df_raw["method"] == "Retrain"]
        pivot_rt = df_rt.pivot_table(index="method", columns="col", values=metric, aggfunc="mean").reindex(
            columns=sampling_cols
        )
        pivot_rt.index = [f"full_{TOTAL_TRAIN_YEARS}yr"]
        pivot_rt["avg"] = pivot_rt.mean(axis=1)
        pivot_rt.index.name = "retrain_scope"
        out = output_dir / f"med_lr_table_{metric}_retrain.csv"
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
        out = output_dir / f"med_lr_table_{metric}_new.csv"
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
        out = output_dir / "med_lr_compact_summary.csv"
        pd.concat(rows, ignore_index=True).to_csv(out, index=False, float_format="%.4f")
        logger.info(f"Saved -> {out.name}")

    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pv = _mean_pivot_by_method_sampling(df_raw, m)
        out = output_dir / f"med_lr_compact_{m}_only.csv"
        pv.to_csv(out, float_format="%.4f")
        logger.info(f"Saved -> {out.name}")

    if "tune_best_params" in df_raw.columns or "tune_val_auc" in df_raw.columns:
        try:
            out = output_dir / "med_lr_tuning_log.csv"
            export_tuning_log(df_raw, out, logger)
            logger.info(f"Saved -> {out.name}")
        except Exception as e:
            logger.warning(f"export_tuning_log failed: {e}")


def _run(tuning_mode: str, n_tune_iter: int, *, results_subdir: str | None = None) -> None:
    logger = get_logger("Medical_YearSplits_LR", console=True, file=True)
    set_seed(42)

    results_subdir = (results_subdir or "").strip()
    base = OUTPUT_DIR / results_subdir if results_subdir else OUTPUT_DIR
    out_dir = base / "tuned" if tuning_mode == "tuned" else base
    out_dir.mkdir(parents=True, exist_ok=True)

    use_tuning = tuning_mode == "tuned"
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Tuning: {tuning_mode} (n_iter={n_tune_iter})")

    all_rows: list[dict] = []
    retrain_done = False

    for label, old_end_year in MEDICAL_YEAR_SPLITS:
        include_retrain = not retrain_done
        logger.info(f"\n{'='*60}")
        logger.info(f"Split: {label}  (Old<={old_end_year}, New={old_end_year + 1}-{TRAIN_END_YEAR}, Test=2007-2008)")
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
            if include_retrain:
                retrain_done = True
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            import traceback

            logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果，請確認醫療資料檔存在。")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = out_dir / "medical_year_splits_lr_raw.csv"
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
    p = argparse.ArgumentParser(description="Phase1 medical year splits — Logistic Regression baseline")
    p.add_argument(
        "--tuning",
        type=str,
        default="tuned",
        choices=["default", "tuned", "both"],
        help="預設 tuned（validation 調參）；default=僅預設超參；both=兩者都跑",
    )
    p.add_argument("--tune-n-iter", type=int, default=48, help="LR hyperparameter search iterations when tuning")
    p.add_argument(
        "--results-subdir",
        type=str,
        default="",
        help="Write outputs into results/phase1_baseline/logistic_regression/<subdir>/ (avoid overwriting).",
    )
    args = p.parse_args()

    sub = (args.results_subdir or "").strip() or None
    if args.tuning == "both":
        _run("default", int(args.tune_n_iter), results_subdir=sub)
        _run("tuned", int(args.tune_n_iter), results_subdir=sub)
    else:
        _run(str(args.tuning), int(args.tune_n_iter), results_subdir=sub)


if __name__ == "__main__":
    main()
