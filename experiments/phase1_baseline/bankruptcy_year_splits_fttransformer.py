"""
Phase 1 - Bankruptcy 年份切割基準線實驗（FT-Transformer）
=======================================================
架構：FTTransformer (rtdl)，Feature Tokenizer + Transformer Encoder
            d_token=64，n_blocks=3，attention_n_heads=8
不平衡處理：同 MLP/XGB 版，由 ImbalanceSampler 採樣後餵入模型

固定 Test = 2015-2018，年份切割依 common_bankruptcy.YEAR_SPLITS
訓練策略（與 XGB 對齊）：Old / New / Retrain / Finetune
採樣策略：none / undersampling / oversampling / hybrid

安裝依賴：pip install rtdl torch
（Python 3.14 + numpy 2.x 若與 rtdl 的 numpy<2 衝突，可：python -m pip install "rtdl==0.0.13" --no-deps --user）

執行時日誌固定寫入 logs/BK_FTTransformer_run.log（每次覆寫），handler 會即時 flush，勿再將 stdout 重導向同一檔以免重複。

最終輸出：
    - results/phase1_baseline/fttransformer/bankruptcy_year_splits_fttransformer_raw.csv
    - results/phase1_baseline/fttransformer/bk_fttransformer_table_{metric}_{old|retrain|finetune|new}.csv
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler, DataPreprocessor
from src.models import FTTransformerWrapper
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]
METRICS             = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]
OUTPUT_DIR          = project_root / "results" / "phase1_baseline" / "fttransformer"


def _select_threshold_from_validation(y_val, y_proba_val):
    """用 validation set 搜尋最佳 F1 閾值。"""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


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
            try:
                X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
                    X_train_raw, y_train_arr, test_size=0.2, random_state=42, stratify=y_train_arr
                )
            except ValueError:
                X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
                    X_train_raw, y_train_arr, test_size=0.2, random_state=42
                )
    else:
        X_fit_raw = X_train_raw
        y_fit = y_train_arr

    pre = DataPreprocessor()
    X_fit, X_val = pre.scale_features(X_fit_raw, X_val_raw, fit=True)
    _, X_test = pre.scale_features(X_fit_raw, X_test_raw, fit=False)

    X_r, y_r = sampler.apply_sampling(X_fit, np.asarray(y_fit), strategy=strategy)
    model = FTTransformerWrapper(name=f"{tag}_{strategy}")
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


def _finetune_eval(X_old, y_old, year_old, X_new, y_new, year_new, X_test_raw, y_test, sampler, strategy, tag, logger):
    """FT 微調：先 Old 後 New 接續訓練，閾值在 old_val+new_val 上選擇。"""
    X_old_fit_raw, y_old_fit, X_old_val_raw, y_old_val = _split_fit_val_by_year(X_old, y_old, year_old)
    X_new_fit_raw, y_new_fit, X_new_val_raw, y_new_val = _split_fit_val_by_year(X_new, y_new, year_new)

    pre = DataPreprocessor()
    X_old_fit, X_old_val = pre.scale_features(X_old_fit_raw, X_old_val_raw, fit=True)
    _, X_new_fit = pre.scale_features(X_old_fit_raw, X_new_fit_raw, fit=False)
    _, X_new_val = pre.scale_features(X_old_fit_raw, X_new_val_raw, fit=False)
    _, X_test = pre.scale_features(X_old_fit_raw, X_test_raw, fit=False)

    X_r1, y_r1 = sampler.apply_sampling(X_old_fit, np.asarray(y_old_fit), strategy=strategy)
    model = FTTransformerWrapper(name=f"{tag}_{strategy}")
    model.fit(X_r1, y_r1)

    X_r2, y_r2 = sampler.apply_sampling(X_new_fit, np.asarray(y_new_fit), strategy=strategy)
    model.fit(X_r2, y_r2, continue_training=True)

    X_val_all = np.concatenate([np.asarray(X_old_val), np.asarray(X_new_val)], axis=0)
    y_val_all = np.concatenate([np.asarray(y_old_val), np.asarray(y_new_val)], axis=0)
    y_proba_val = model.predict_proba(X_val_all)
    threshold, val_f1 = _select_threshold_from_validation(y_val_all, y_proba_val)

    y_t = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    metrics = compute_metrics(y_t, model.predict_proba(X_test), threshold=threshold)
    logger.info(
        f"    {tag:12s} {strategy:12s} [thr={threshold:.3f}, valF1={val_f1:.4f}]: "
        f"AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}"
    )
    return metrics


def run_split(label, X_old, y_old, year_old, X_new, y_new, year_new, X_test, y_test, logger, include_retrain=True):
    sampler = ImbalanceSampler()
    rows = []

    # Old
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_old, y_old, X_test, y_test, sampler, strat, "Old", logger, year_train=year_old)
        rows.append({"split": label, "method": "Old", "sampling": strat, **m})

    if include_retrain:
        # Retrain：Old + New 全量訓練，validation 由 old/new 各自切分後合併
        X_fit_re, y_fit_re, X_val_re, y_val_re = _build_retrain_fit_val(X_old, y_old, year_old, X_new, y_new, year_new)
        logger.info(f"  Retrain full concat: fit={len(X_fit_re)} val={len(X_val_re)}")
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
            )
            rows.append({"split": label, "method": "Retrain", "sampling": strat, **m})

    # Finetune
    for strat in SAMPLING_STRATEGIES:
        m = _finetune_eval(
            X_old,
            y_old,
            year_old,
            X_new,
            y_new,
            year_new,
            X_test,
            y_test,
            sampler,
            strat,
            "Finetune",
            logger,
        )
        rows.append({"split": label, "method": "Finetune", "sampling": strat, **m})

    # New
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_new, y_new, X_test, y_test, sampler, strat, "New", logger, year_train=year_new)
        rows.append({"split": label, "method": "New", "sampling": strat, **m})

    return rows


def format_tables(df_raw, logger):
    split_yr = {label: (old_end - 1998, 2014 - old_end) for label, old_end in YEAR_SPLITS}
    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]
    split_labels  = [label for label, _ in YEAR_SPLITS]
    sampling_cols = SAMPLING_STRATEGIES

    for metric in METRICS:
        for method, suffix, idx_fn in [
            ("Old", "old", lambda s: f"{split_yr[s][0]}yr"),
            ("Retrain", "retrain", lambda s: f"{split_yr[s][0]}+{split_yr[s][1]}"),
            ("Finetune", "finetune", lambda s: f"{split_yr[s][0]}+{split_yr[s][1]}"),
            ("New", "new", lambda s: f"{split_yr[s][1]}yr"),
        ]:
            if method == "Retrain":
                pivot = (
                    df_raw[df_raw["method"] == method]
                    .pivot_table(index="method", columns="col", values=metric, aggfunc="mean")
                    .reindex(columns=sampling_cols)
                )
                pivot.index = ["full_16yr"]
                pivot["avg"] = pivot.mean(axis=1)
                pivot.index.name = "retrain_scope"
            else:
                pivot = (
                    df_raw[df_raw["method"] == method]
                    .pivot(index="split", columns="col", values=metric)
                    .reindex(index=split_labels, columns=sampling_cols)
                )
                pivot.index = [idx_fn(s) for s in pivot.index]
                pivot["avg"] = pivot.mean(axis=1)
                pivot.loc["avg"] = pivot.mean()
                if method == "Old":
                    pivot.index.name = "old_years"
                elif method == "Finetune":
                    pivot.index.name = "old+new_years_finetune"
                else:
                    pivot.index.name = "new_years"
            out = OUTPUT_DIR / f"bk_fttransformer_table_{metric}_{suffix}.csv"
            pivot.to_csv(out, float_format="%.4f")
            logger.info(f"  Saved -> {out.name}")


def _mean_pivot_by_method_sampling(df_raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    t = (
        df_raw.groupby(["method", "sampling"])[metric]
        .mean()
        .unstack("sampling")
        .reindex(index=["Old", "New", "Retrain", "Finetune"], columns=SAMPLING_REPORT_ORDER)
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
    path_all = OUTPUT_DIR / "bk_fttransformer_compact_summary.csv"
    long_df.to_csv(path_all, index=False, float_format="%.4f")
    logger.info(f"  Saved -> {path_all.name}")

    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pivot = _mean_pivot_by_method_sampling(df_raw, m)
        path_m = OUTPUT_DIR / f"bk_fttransformer_compact_{m}_only.csv"
        pivot.to_csv(path_m, float_format="%.4f")
        logger.info(f"  Saved -> {path_m.name}")


def main():
    logger = get_logger(
        "BK_YearSplits_FTTransformer",
        console=True,
        file=True,
        log_filename="BK_FTTransformer_run.log",
        log_file_mode="w",
    )
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    retrain_done = False
    for label, old_end_year in YEAR_SPLITS:
        logger.info(f"\n{'='*60}\nSplit: {label}  (Old<={old_end_year}, New={old_end_year+1}-2014, Test=2015-2018)\n{'='*60}")
        try:
            X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, year_test = get_bankruptcy_year_split(
                logger,
                old_end_year=old_end_year,
                return_years=True,
            )
            all_rows.extend(
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
                    include_retrain=(not retrain_done),
                )
            )
            retrain_done = True
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            import traceback; logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果"); return

    df_raw = pd.DataFrame(all_rows)
    raw_path = OUTPUT_DIR / "bankruptcy_year_splits_fttransformer_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")
    logger.info("\n產出指標 pivot 表格...")
    format_tables(df_raw, logger)
    logger.info("\n產出 compact 報表...")
    export_compact_report(df_raw, logger)
    logger.info("\n=== 完成 ===")


if __name__ == "__main__":
    # 重導向 stdout 到檔案時，避免 block buffering 導致 log 長時間不更新
    for _stream in (sys.stdout, sys.stderr):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(line_buffering=True)
            except Exception:
                pass
    main()
