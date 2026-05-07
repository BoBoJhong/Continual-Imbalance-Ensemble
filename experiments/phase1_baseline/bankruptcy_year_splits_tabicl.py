"""\
Phase 1 - Bankruptcy 年份切割基準線實驗（TabICL / TabPFN）
============================================================

目標：為破產資料集新增一條 **TabICL** baseline 流程，對齊 `bankruptcy_year_splits_xgb.py`：

- 固定 Test = 2015-2018；訓練窗 1999–2014（16 年）
- 依 `common_bankruptcy.YEAR_SPLITS` 逐年滑動 old_end，共 15 組 Old/New 切割
- Validation：訓練段內依 **fyear 逐年**各抽約 20% 作 validation（`_split_fit_val_by_year`）
- Scaler 僅在 train fold 上 fit；採樣僅施加於 train fold
- 閾值：validation 上做 F1 grid search（0.05–0.95，步長 0.01），再套用到 test
- 策略：Old / New / Retrain（Retrain 僅在第一個 split 跑一次）
- 採樣：none / undersampling / oversampling / hybrid

TabICL 實作：使用 `src.models.TabICLWrapper`（底層 TabPFNClassifier）。
注意：TabPFN 對 context（訓練集）大小有實務限制；此腳本提供 `--max-train-samples` 以 cap。

輸出（與 XGB 同型）：
  - results/phase1_baseline/tabicl/<tag>/bankruptcy_year_splits_tabicl_raw.csv
  - bk_tabicl_table_{metric}_{old|new|retrain}.csv
  - bk_tabicl_compact_summary.csv
  - bk_tabicl_compact_{AUC|F1|Recall}_only.csv

安裝依賴：pip install tabpfn
"""

import argparse
import json
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

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler, DataPreprocessor
from src.models import TabICLWrapper
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split


SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]
METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]

OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "tabicl"
CHECKPOINT_PATH = OUTPUT_DIR / "bankruptcy_year_splits_tabicl_checkpoint.json"
PARTIAL_RAW_PATH = OUTPUT_DIR / "bankruptcy_year_splits_tabicl_raw.partial.csv"


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
    *,
    device: str,
    n_estimators: int,
    max_train_samples: int,
    subsample: str,
    norm_methods: list[str],
    feat_shuffle_method: str,
    class_shift: bool,
    outlier_threshold: float,
    softmax_temperature: float,
    average_logits: bool,
    use_hierarchical: bool,
    batch_size: int,
    use_amp: bool,
    checkpoint_version: str,
    n_jobs: int | None,
    model_verbose: bool,
    X_val_raw=None,
    y_val=None,
    year_train=None,
    split_label: str = "",
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
            X_fit_raw, y_fit, X_val_raw, y_val = _split_fit_val(X_train_raw, y_train_arr)
    else:
        X_fit_raw = X_train_raw
        y_fit = y_train_arr

    pre = DataPreprocessor()
    X_fit, X_val = pre.scale_features(X_fit_raw, X_val_raw, fit=True)
    _, X_test = pre.scale_features(X_fit_raw, X_test_raw, fit=False)

    X_r, y_r = sampler.apply_sampling(X_fit, np.asarray(y_fit), strategy=strategy)

    fit_seed = 42 + (zlib.adler32(f"{split_label}|{tag}|{strategy}".encode()) % 100000)
    model = TabICLWrapper(
        name=f"{tag}_{strategy}",
        device=device,
        n_estimators=n_estimators,
        seed=fit_seed,
        max_train_samples=max_train_samples,
        subsample=subsample,
        norm_methods=norm_methods,
        feat_shuffle_method=feat_shuffle_method,
        class_shift=class_shift,
        outlier_threshold=outlier_threshold,
        softmax_temperature=softmax_temperature,
        average_logits=average_logits,
        use_hierarchical=use_hierarchical,
        batch_size=batch_size,
        use_amp=use_amp,
        checkpoint_version=checkpoint_version,
        n_jobs=n_jobs,
        verbose=model_verbose,
    )
    model.fit(X_r, y_r)

    y_proba_val = model.predict_proba(X_val)
    threshold, val_f1 = _select_threshold_from_validation(np.asarray(y_val), y_proba_val)

    y_t = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    metrics = compute_metrics(y_t, model.predict_proba(X_test), threshold=threshold)
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
    *,
    include_retrain=True,
    device: str,
    n_estimators: int,
    max_train_samples: int,
    subsample: str,
    norm_methods: list[str],
    feat_shuffle_method: str,
    class_shift: bool,
    outlier_threshold: float,
    softmax_temperature: float,
    average_logits: bool,
    use_hierarchical: bool,
    batch_size: int,
    use_amp: bool,
    checkpoint_version: str,
    n_jobs: int | None,
    model_verbose: bool,
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
            device=device,
            n_estimators=n_estimators,
            max_train_samples=max_train_samples,
            subsample=subsample,
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            class_shift=class_shift,
            outlier_threshold=outlier_threshold,
            softmax_temperature=softmax_temperature,
            average_logits=average_logits,
            use_hierarchical=use_hierarchical,
            batch_size=batch_size,
            use_amp=use_amp,
            checkpoint_version=checkpoint_version,
            n_jobs=n_jobs,
            model_verbose=model_verbose,
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
            device=device,
            n_estimators=n_estimators,
            max_train_samples=max_train_samples,
            subsample=subsample,
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            class_shift=class_shift,
            outlier_threshold=outlier_threshold,
            softmax_temperature=softmax_temperature,
            average_logits=average_logits,
            use_hierarchical=use_hierarchical,
            batch_size=batch_size,
            use_amp=use_amp,
            checkpoint_version=checkpoint_version,
            n_jobs=n_jobs,
            model_verbose=model_verbose,
        )
        rows.append({"split": label, "method": "New", "sampling": strat, **m})

    if include_retrain:
        X_fit_re, y_fit_re, X_val_re, y_val_re = _build_retrain_fit_val(
            X_old, y_old, year_old, X_new, y_new, year_new
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
                device=device,
                n_estimators=n_estimators,
                max_train_samples=max_train_samples,
                subsample=subsample,
                norm_methods=norm_methods,
                feat_shuffle_method=feat_shuffle_method,
                class_shift=class_shift,
                outlier_threshold=outlier_threshold,
                softmax_temperature=softmax_temperature,
                average_logits=average_logits,
                use_hierarchical=use_hierarchical,
                batch_size=batch_size,
                use_amp=use_amp,
                checkpoint_version=checkpoint_version,
                n_jobs=n_jobs,
                model_verbose=model_verbose,
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
        out = output_dir / f"bk_tabicl_table_{metric}_old.csv"
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
        out = output_dir / f"bk_tabicl_table_{metric}_retrain.csv"
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
        out = output_dir / f"bk_tabicl_table_{metric}_new.csv"
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
    path_all = output_dir / "bk_tabicl_compact_summary.csv"
    long_df.to_csv(path_all, index=False, float_format="%.4f")
    logger.info(f"  Saved -> {path_all.name}")

    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pivot = _mean_pivot_by_method_sampling(df_raw, m)
        path_m = output_dir / f"bk_tabicl_compact_{m}_only.csv"
        pivot.to_csv(path_m, float_format="%.4f")
        logger.info(f"  Saved -> {path_m.name}")


def _save_raw_snapshot(rows: list[dict], output_dir: Path, logger) -> None:
    """Persist current accumulated rows as a checkpoint-like raw snapshot."""
    if not rows:
        return
    raw_path = output_dir / "bankruptcy_year_splits_tabicl_raw.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"  Snapshot saved -> {raw_path.name}  ({len(rows)} rows)")


def _deduplicate_rows(rows: list[dict]) -> list[dict]:
    """Keep the latest row per (split, method, sampling)."""
    merged: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        k = (str(r.get("split", "")), str(r.get("method", "")), str(r.get("sampling", "")))
        merged[k] = r
    return list(merged.values())


def _load_progress(output_dir: Path, logger):
    rows: list[dict] = []
    completed_splits: set[str] = set()
    retrain_done = False

    partial_raw = output_dir / PARTIAL_RAW_PATH.name
    checkpoint = output_dir / CHECKPOINT_PATH.name

    if partial_raw.exists():
        try:
            rows = pd.read_csv(partial_raw).to_dict("records")
            rows = _deduplicate_rows(rows)
            logger.info(f"Loaded partial raw: {partial_raw.name} ({len(rows)} rows)")
        except Exception as exc:
            logger.warning(f"Failed to load partial raw, ignore and continue: {exc}")

    if checkpoint.exists():
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            completed_splits = set(payload.get("completed_splits", []))
            retrain_done = bool(payload.get("retrain_done", False))
            logger.info(
                f"Loaded checkpoint: {checkpoint.name} "
                f"(completed_splits={len(completed_splits)}, retrain_done={retrain_done})"
            )
        except Exception as exc:
            logger.warning(f"Failed to load checkpoint, ignore and continue: {exc}")

    return rows, completed_splits, retrain_done


def _save_progress(
    output_dir: Path,
    rows: list[dict],
    completed_splits: set[str],
    retrain_done: bool,
) -> None:
    partial_raw = output_dir / PARTIAL_RAW_PATH.name
    checkpoint = output_dir / CHECKPOINT_PATH.name
    if rows:
        pd.DataFrame(rows).to_csv(partial_raw, index=False, float_format="%.6f")

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_count": len(rows),
        "completed_splits": sorted(completed_splits),
        "retrain_done": bool(retrain_done),
    }
    tmp = checkpoint.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(checkpoint)


def _clear_progress_files(output_dir: Path, logger) -> None:
    for p in (
        output_dir / CHECKPOINT_PATH.name,
        output_dir / PARTIAL_RAW_PATH.name,
    ):
        if p.exists():
            p.unlink()
            logger.info(f"Removed progress file: {p.name}")


def _run_one_output_dir(
    output_dir: Path,
    *,
    logger,
    use_resume: bool,
    device: str,
    n_estimators: int,
    max_train_samples: int,
    subsample: str,
    norm_methods: list[str],
    feat_shuffle_method: str,
    class_shift: bool,
    outlier_threshold: float,
    softmax_temperature: float,
    average_logits: bool,
    use_hierarchical: bool,
    batch_size: int,
    use_amp: bool,
    checkpoint_version: str,
    n_jobs: int | None,
    model_verbose: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_resume:
        all_rows, completed_splits, retrain_done = _load_progress(output_dir, logger)
    else:
        _clear_progress_files(output_dir, logger)
        all_rows, completed_splits, retrain_done = [], set(), False

    for label, old_end_year in YEAR_SPLITS:
        if label in completed_splits:
            logger.info(f"\n[checkpoint skip] {label} already completed")
            continue
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Split: {label}  (Old<={old_end_year}, New={old_end_year + 1}-2014, Test=2015-2018)"
        )
        logger.info("=" * 60)
        try:
            X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, year_test = get_bankruptcy_year_split(
                logger, old_end_year=old_end_year, return_years=True
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
                device=device,
                n_estimators=n_estimators,
                max_train_samples=max_train_samples,
                subsample=subsample,
                norm_methods=norm_methods,
                feat_shuffle_method=feat_shuffle_method,
                class_shift=class_shift,
                outlier_threshold=outlier_threshold,
                softmax_temperature=softmax_temperature,
                average_logits=average_logits,
                use_hierarchical=use_hierarchical,
                batch_size=batch_size,
                use_amp=use_amp,
                checkpoint_version=checkpoint_version,
                n_jobs=n_jobs,
                model_verbose=model_verbose,
            )
            all_rows.extend(rows)
            _save_raw_snapshot(all_rows, output_dir, logger)
            retrain_done = True
            completed_splits.add(label)
            _save_progress(output_dir, all_rows, completed_splits, retrain_done)
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            import traceback

            logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果，請確認資料檔存在與依賴已安裝。")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = output_dir / "bankruptcy_year_splits_tabicl_raw.csv"
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")

    logger.info("\n產出指標 pivot 表格...")
    format_tables(df_raw, logger, output_dir)

    logger.info("\n產出精簡摘要（跨各 split 平均）...")
    export_compact_report(df_raw, logger, output_dir)

    logger.info("\n=== 完成 ===")
    summary = _mean_pivot_by_method_sampling(df_raw, "AUC")
    logger.info("\nAUC 摘要（method × sampling 平均，跨各年份切割）:\n" + summary.to_string())
    _clear_progress_files(output_dir, logger)


def main():
    parser = argparse.ArgumentParser(description="Phase1 bankruptcy year splits — TabICL (TabPFN) baseline")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="TabPFN 推論/訓練 device（auto 會優先用 cuda）",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=32,
        help="TabICL ensemble 成員數（等同官方推薦 n_estimators）",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=1024,
        help="TabICL context 上限（訓練資料筆數 cap；避免 TabPFN 因樣本過多而變慢/爆記憶體）",
    )
    parser.add_argument(
        "--subsample",
        choices=["stratified", "random"],
        default="stratified",
        help="當訓練資料超過 max-train-samples 時的抽樣策略",
    )
    parser.add_argument(
        "--norm-methods",
        nargs="+",
        default=["none", "power"],
        help="TabICL 正規化候選（預設: none power）",
    )
    parser.add_argument("--feat-shuffle-method", type=str, default="latin")
    parser.add_argument("--no-class-shift", action="store_true", help="停用 class cyclic shift")
    parser.add_argument("--outlier-threshold", type=float, default=4.0)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--no-average-logits", action="store_true", help="停用 logits averaging")
    parser.add_argument("--no-use-hierarchical", action="store_true", help="停用 hierarchical routing")
    parser.add_argument("--batch-size", type=int, default=8, help="TabICL 推論/集成批次大小")
    parser.add_argument("--no-amp", action="store_true", help="停用自動混合精度")
    parser.add_argument(
        "--checkpoint-version",
        type=str,
        default="tabicl-classifier-v1.1-0506.ckpt",
        help="TabICL checkpoint 版本字串",
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--model-verbose", action="store_true")
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="結果輸出子資料夾標籤（例如 rerun_20260406）。會寫到 tabicl/<tag>/",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略 checkpoint / partial raw，從頭重跑",
    )

    args = parser.parse_args()

    logger = get_logger("BK_YearSplits_TabICL", console=True, file=True)
    set_seed(42)

    tag = args.output_tag.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / tag

    logger.info(f"\n{'#'*60}\n模式: TabICL (TabPFN)\n輸出: {out_dir}\n{'#'*60}")
    _run_one_output_dir(
        out_dir,
        logger=logger,
        use_resume=(not args.no_resume),
        device=args.device,
        n_estimators=args.n_estimators,
        max_train_samples=args.max_train_samples,
        subsample=args.subsample,
        norm_methods=args.norm_methods,
        feat_shuffle_method=args.feat_shuffle_method,
        class_shift=(not args.no_class_shift),
        outlier_threshold=args.outlier_threshold,
        softmax_temperature=args.softmax_temperature,
        average_logits=(not args.no_average_logits),
        use_hierarchical=(not args.no_use_hierarchical),
        batch_size=args.batch_size,
        use_amp=(not args.no_amp),
        checkpoint_version=args.checkpoint_version,
        n_jobs=args.n_jobs,
        model_verbose=args.model_verbose,
    )


if __name__ == "__main__":
    main()
