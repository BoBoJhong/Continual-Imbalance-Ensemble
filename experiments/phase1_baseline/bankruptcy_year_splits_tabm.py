"""
Phase 1 - Bankruptcy 年份切割基準線實驗（TabM）
================================================
與 bankruptcy_year_splits_xgb.py 相同協定：
  固定 Test = 2015-2018；訓練窗 1999–2014；15 組 Old/New 切割（YEAR_SPLITS）。
  Validation：訓練段內依 fyear 逐年各抽約 20% 合併為校準集。
  Retrain 僅在全部迭代中第一次執行，故跑滿 15 折時 raw = 124 列。

訓練策略：
  - Old      : 只用歷史 (Old) 資料訓練
  - New      : 只用新營運 (New) 資料訓練
  - Retrain  : 歷史 + 新營運全量合併後訓練（只產生一組）

採樣策略：none / undersampling / oversampling / hybrid

安裝依賴：
  pip install tabm rtdl_num_embeddings

用法：
  python experiments/phase1_baseline/bankruptcy_year_splits_tabm.py
  python experiments/phase1_baseline/bankruptcy_year_splits_tabm.py --splits split_2+14 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split
from experiments._shared.baseline_val_search import (
    export_tuning_log,
    search_tabm_on_val,
    tuning_meta,
)
from src.data import DataPreprocessor, ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import TabMWrapper
from src.utils import get_logger, set_seed

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]
METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]
TABM_BASE_DIR = project_root / "results" / "phase1_baseline" / "tabm"
OUTPUT_DIR = TABM_BASE_DIR
CHECKPOINT_PATH = OUTPUT_DIR / "bankruptcy_year_splits_tabm_checkpoint.json"
PARTIAL_RAW_PATH = OUTPUT_DIR / "bankruptcy_year_splits_tabm_raw.partial.csv"


def _stage_key(split: str, method: str, sampling: str) -> str:
    return f"{split}|{method}|{sampling}"


def _deduplicate_rows(rows):
    merged = {}
    for row in rows:
        key = _stage_key(str(row["split"]), str(row["method"]), str(row["sampling"]))
        merged[key] = row
    return list(merged.values())


def _load_progress(logger):
    rows = []
    completed_stage_keys = set()

    if PARTIAL_RAW_PATH.exists():
        try:
            rows = pd.read_csv(PARTIAL_RAW_PATH).to_dict("records")
            rows = _deduplicate_rows(rows)
            completed_stage_keys.update(
                _stage_key(str(r["split"]), str(r["method"]), str(r["sampling"])) for r in rows
            )
            logger.info(f"載入 partial raw：{PARTIAL_RAW_PATH.name} ({len(rows)} rows)")
        except Exception as exc:
            logger.warning(f"讀取 partial raw 失敗，將忽略：{exc}")

    if CHECKPOINT_PATH.exists():
        try:
            data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            completed_stage_keys.update(data.get("completed_stage_keys", []))
            logger.info(f"載入 checkpoint：{CHECKPOINT_PATH.name} ({len(completed_stage_keys)} stages)")
        except Exception as exc:
            logger.warning(f"讀取 checkpoint 失敗，將忽略：{exc}")

    return rows, completed_stage_keys


def _save_progress(rows, completed_stage_keys):
    if rows:
        pd.DataFrame(rows).to_csv(PARTIAL_RAW_PATH, index=False, float_format="%.6f")

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_count": len(rows),
        "completed_stage_keys": sorted(completed_stage_keys),
    }
    tmp_path = CHECKPOINT_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(CHECKPOINT_PATH)


def _clear_progress_files(logger):
    for p in (CHECKPOINT_PATH, PARTIAL_RAW_PATH):
        if p.exists():
            p.unlink()
            logger.info(f"已清除進度檔：{p.name}")


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
    n_tune_iter: int = 24,
    device: str = "auto",
    use_amp: bool = True,
):
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
        base = {
            "device": device,
            "use_amp": use_amp,
            "max_epochs": 200,
            "patience": 20,
            "batch_size": 256,
            "predict_batch_size": 4096,
            "val_batch_size": 8192,
        }
        best, auc_s = search_tabm_on_val(
            X_r,
            y_r,
            X_val,
            np.asarray(y_val),
            base,
            n_iter=n_tune_iter,
            seed=tune_seed,
        )
        if best:
            model = TabMWrapper(
                name=f"{tag}_{strategy}",
                seed=tune_seed,
                **base,
                **best,
            )
        else:
            model = TabMWrapper(
                name=f"{tag}_{strategy}",
                seed=tune_seed,
                **base,
            )
            best, auc_s = {}, float("nan")
        model.fit(X_r, y_r)
        tune_ex = tuning_meta(best, auc_s)
    else:
        model = TabMWrapper(
            name=f"{tag}_{strategy}",
            seed=42,
            device=device,
            use_amp=use_amp,
        )
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
    all_rows,
    completed_stage_keys,
    include_retrain=True,
    *,
    use_tuning: bool = False,
    n_tune_iter: int = 24,
    device: str = "auto",
    use_amp: bool = True,
):
    sampler = ImbalanceSampler()

    def save_stage(method, strat, metrics):
        key = _stage_key(label, method, strat)
        all_rows.append({"split": label, "method": method, "sampling": strat, **metrics})
        completed_stage_keys.add(key)
        _save_progress(all_rows, completed_stage_keys)

    def is_done(method, strat):
        return _stage_key(label, method, strat) in completed_stage_keys

    for strat in SAMPLING_STRATEGIES:
        if is_done("Old", strat):
            logger.info(f"    Old          {strat:12s} [checkpoint skip]")
            continue
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
            device=device,
            use_amp=use_amp,
        )
        save_stage("Old", strat, m)

    for strat in SAMPLING_STRATEGIES:
        if is_done("New", strat):
            logger.info(f"    New          {strat:12s} [checkpoint skip]")
            continue
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
            device=device,
            use_amp=use_amp,
        )
        save_stage("New", strat, m)

    if include_retrain:
        X_fit_re, y_fit_re, X_val_re, y_val_re = _build_retrain_fit_val(
            X_old, y_old, year_old, X_new, y_new, year_new
        )
        logger.info(f"  Retrain: fit={len(X_fit_re)} val={len(X_val_re)}")
        for strat in SAMPLING_STRATEGIES:
            if is_done("Retrain", strat):
                logger.info(f"    Retrain      {strat:12s} [checkpoint skip]")
                continue
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
                device=device,
                use_amp=use_amp,
            )
            save_stage("Retrain", strat, m)


def format_tables(df_raw, logger, split_labels_filter=None):
    split_yr = {label: (old_end - 1998, 2014 - old_end) for label, old_end in YEAR_SPLITS}

    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]

    split_labels = [label for label, _ in YEAR_SPLITS]
    if split_labels_filter is not None:
        split_labels = [s for s in split_labels if s in split_labels_filter]
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
        out = OUTPUT_DIR / f"bk_tabm_table_{metric}_old.csv"
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
        out = OUTPUT_DIR / f"bk_tabm_table_{metric}_retrain.csv"
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
        out = OUTPUT_DIR / f"bk_tabm_table_{metric}_new.csv"
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


def export_compact_report(df_raw: pd.DataFrame, logger) -> None:
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
    path_all = OUTPUT_DIR / "bk_tabm_compact_summary.csv"
    long_df.to_csv(path_all, index=False, float_format="%.4f")
    logger.info(f"  Saved -> {path_all.name}")

    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pivot = _mean_pivot_by_method_sampling(df_raw, m)
        path_m = OUTPUT_DIR / f"bk_tabm_compact_{m}_only.csv"
        pivot.to_csv(path_m, float_format="%.4f")
        logger.info(f"  Saved -> {path_m.name}")


def _run_one_output_dir(
    output_dir: Path,
    *,
    split_iter,
    use_tuning: bool,
    n_tune_iter: int,
    device: str,
    use_amp: bool,
    use_resume: bool,
    logger,
) -> None:
    global OUTPUT_DIR, CHECKPOINT_PATH, PARTIAL_RAW_PATH
    OUTPUT_DIR = output_dir
    CHECKPOINT_PATH = OUTPUT_DIR / "bankruptcy_year_splits_tabm_checkpoint.json"
    PARTIAL_RAW_PATH = OUTPUT_DIR / "bankruptcy_year_splits_tabm_raw.partial.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"結果與進度檔目錄: {OUTPUT_DIR}")

    if use_resume:
        all_rows, completed_stage_keys = _load_progress(logger)
    else:
        _clear_progress_files(logger)
        all_rows, completed_stage_keys = [], set()

    retrain_done = any("|Retrain|" in k for k in completed_stage_keys)

    for label, old_end_year in split_iter:
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Split: {label}  (Old<={old_end_year}, New={old_end_year + 1}-2014, Test=2015-2018)"
        )
        logger.info("=" * 60)
        try:
            X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, _year_test = (
                get_bankruptcy_year_split(
                    logger,
                    old_end_year=old_end_year,
                    return_years=True,
                )
            )
            run_split(
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
                all_rows,
                completed_stage_keys,
                include_retrain=(not retrain_done),
                use_tuning=use_tuning,
                n_tune_iter=n_tune_iter,
                device=device,
                use_amp=use_amp,
            )
            retrain_done = True
        except Exception as exc:
            logger.error(f"[ERROR] {label}: {exc}")
            logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果，請確認資料檔存在且已安裝 tabm / PyTorch。")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = OUTPUT_DIR / "bankruptcy_year_splits_tabm_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")

    expected_methods = {"Old", "New", "Retrain"}
    got_methods = set(df_raw["method"].unique())
    n_splits_ran = len({r["split"] for r in all_rows})
    expect_rows = 12 + 8 * (n_splits_ran - 1) if n_splits_ran > 0 else 0
    if len(df_raw) != expect_rows or got_methods != expected_methods:
        logger.warning(
            "輸出與目前三策略協定不一致：rows=%s（預期 n_splits=%s → %s 列）、method=%s（預期 %s）。",
            len(df_raw),
            n_splits_ran,
            expect_rows,
            sorted(got_methods),
            sorted(expected_methods),
        )

    logger.info("\n產出指標 pivot 表格...")
    filter_labels = [lab for lab, _ in split_iter] if args.splits else None
    format_tables(df_raw, logger, split_labels_filter=filter_labels)

    logger.info("\n產出精簡摘要（跨各 split 平均）...")
    export_compact_report(df_raw, logger)

    if use_tuning:
        export_tuning_log(df_raw, OUTPUT_DIR / "bk_tabm_tuning_log.csv", logger)

    logger.info("\n=== 完成 ===")
    summary = _mean_pivot_by_method_sampling(df_raw, "AUC")
    logger.info("\nAUC 摘要（method × sampling 平均，跨各年份切割）:\n" + summary.to_string())
    _clear_progress_files(logger)


def main():
    parser = argparse.ArgumentParser(description="Bankruptcy 年份切割 — TabM baseline（對齊 XGB 三策略）")
    parser.add_argument(
        "--results-subdir",
        default="",
        metavar="NAME",
        help="結果寫入 tabm/<NAME>/；若 --tuning tuned/both，會再加上 tuned/ 子目錄",
    )
    parser.add_argument(
        "--tuning",
        choices=["default", "tuned", "both"],
        default="default",
        help="default=僅預設超參數；tuned=僅 validation AUC 調參；both=兩者皆跑",
    )
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=24,
        help="TabM validation AUC 隨機搜尋候選數（預設 24；越大越慢）",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="只跑指定 split（例如 split_2+14）；預設跑 YEAR_SPLITS 全部",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("TABM_DEVICE", "auto"),
        choices=["auto", "cuda", "cpu", "mps"],
        help="訓練裝置：auto=優先 CUDA，其次 MPS，否則 CPU",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="停用 CUDA mixed precision（僅在 CUDA 上有影響）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略 checkpoint / partial raw，從頭跑",
    )
    args = parser.parse_args()

    logger = get_logger("BK_YearSplits_TabM", console=True, file=True)
    set_seed(42)
    sub = (args.results_subdir or "").strip().replace("\\", "/").strip("/")
    if sub and (Path(sub).name != sub or ".." in Path(sub).parts):
        raise SystemExit("--results-subdir 請使用單一資料夾名稱，勿含路徑跳脫")

    split_iter = list(YEAR_SPLITS)
    if args.splits:
        want = set(args.splits)
        split_iter = [(lab, yr) for lab, yr in YEAR_SPLITS if lab in want]
        unknown = want - {lab for lab, _ in split_iter}
        if unknown:
            logger.error(f"未知的 split 標籤: {sorted(unknown)}")
        if not split_iter:
            logger.error("沒有可執行的 split，請檢查 --splits")
            return

    base_output_dir = (TABM_BASE_DIR / sub) if sub else TABM_BASE_DIR
    use_amp = not args.no_amp
    use_resume = (not args.no_resume) and os.getenv("TABM_RESUME", "1") != "0"

    if args.tuning == "default":
        runs = [(False, base_output_dir)]
    elif args.tuning == "tuned":
        runs = [(True, base_output_dir / "tuned")]
    else:
        runs = [(False, base_output_dir), (True, base_output_dir / "tuned")]

    for use_tune, out_dir in runs:
        tag = "validation AUC 調參" if use_tune else "預設超參數"
        logger.info(f"\n{'#'*60}\n模式: {tag}\n輸出: {out_dir}\n{'#'*60}")
        _run_one_output_dir(
            out_dir,
            split_iter=split_iter,
            use_tuning=use_tune,
            n_tune_iter=args.tune_n_iter,
            device=args.device,
            use_amp=use_amp,
            use_resume=use_resume,
            logger=logger,
        )


if __name__ == "__main__":
    main()
