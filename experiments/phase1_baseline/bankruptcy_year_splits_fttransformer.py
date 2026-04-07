"""
Phase 1 - Bankruptcy 年份切割基準線實驗（FT-Transformer）
=======================================================
架構：FTTransformer (rtdl)，Feature Tokenizer + Transformer Encoder
            d_token=64，n_blocks=3，attention_n_heads=8
不平衡處理：同 MLP/XGB 版，由 ImbalanceSampler 採樣後餵入模型

固定 Test = 2015-2018，年份切割依 common_bankruptcy.YEAR_SPLITS（15 組，含 New 僅 1 年之 split_15+1）。
訓練策略：**Old / New / Retrain**（不跑 Finetune，與本專案其他 baseline 的「滑動窗」對照方式一致）。
跑滿時 raw **124** 列：首個 split 為 4+4+4，其餘 14 個 split 各 4+4（Retrain 僅第一次迭代）。
採樣策略：none / undersampling / oversampling / hybrid

列名對照（與 XGB pivot 相同）：`split_k+(16-k)` → Old 段 **k** 年、New 段 **16−k** 年（訓練窗 1999–2014）。
故 **bk_*_table_*_old** 列為 **1yr→15yr**；**bk_*_table_*_new** 列為 **15yr→1yr**。

安裝依賴：pip install rtdl torch
（Python 3.14 + numpy 2.x 若與 rtdl 的 numpy<2 衝突，可：python -m pip install "rtdl==0.0.13" --no-deps --user）

裝置：預設 --device auto（有 CUDA 則用 GPU，否則 MPS/CPU）。可設環境變數 FTTRANSFORMER_DEVICE。
訓練在 CUDA 上預設開啟 AMP；不平衡二元分類使用 BCE pos_weight=auto（n_neg/n_pos）。

執行時日誌固定寫入 logs/BK_FTTransformer_run.log（每次覆寫），handler 會即時 flush，勿再將 stdout 重導向同一檔以免重複。
若先前 partial 含 Finetune 列，建議 `--no-resume` 重跑以得到僅 Old/New/Retrain 的 raw。

最終輸出（預設根目錄；可用 --results-subdir <名稱> 寫入 fttransformer/<名稱>/）：
    - bankruptcy_year_splits_fttransformer_raw.csv
    - bk_fttransformer_table_{metric}_{old|retrain|new}.csv
"""
import sys
import os
import argparse
import json
from datetime import datetime
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
FT_BASE_DIR = project_root / "results" / "phase1_baseline" / "fttransformer"
# main() 可依 --results-subdir 覆寫以下三個路徑
OUTPUT_DIR = FT_BASE_DIR
CHECKPOINT_PATH = OUTPUT_DIR / "bankruptcy_year_splits_fttransformer_checkpoint.json"
PARTIAL_RAW_PATH = OUTPUT_DIR / "bankruptcy_year_splits_fttransformer_raw.partial.csv"


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
        except Exception as e:
            logger.warning(f"讀取 partial raw 失敗，將忽略：{e}")

    if CHECKPOINT_PATH.exists():
        try:
            data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            completed_stage_keys.update(data.get("completed_stage_keys", []))
            logger.info(f"載入 checkpoint：{CHECKPOINT_PATH.name} ({len(completed_stage_keys)} stages)")
        except Exception as e:
            logger.warning(f"讀取 checkpoint 失敗，將忽略：{e}")

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
    *,
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
    model = FTTransformerWrapper(
        name=f"{tag}_{strategy}",
        device=device,
        pos_weight="auto",
        use_amp=use_amp,
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

    # Old
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
            device=device,
            use_amp=use_amp,
        )
        save_stage("Old", strat, m)

    # New（與 XGB 等相同順序：Old → New → Retrain；本腳本不跑 Finetune）
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
            device=device,
            use_amp=use_amp,
        )
        save_stage("New", strat, m)

    if include_retrain:
        X_fit_re, y_fit_re, X_val_re, y_val_re = _build_retrain_fit_val(X_old, y_old, year_old, X_new, y_new, year_new)
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
                device=device,
                use_amp=use_amp,
            )
            save_stage("Retrain", strat, m)


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
    global OUTPUT_DIR, CHECKPOINT_PATH, PARTIAL_RAW_PATH

    parser = argparse.ArgumentParser(description="Phase1 bankruptcy year splits — FT-Transformer baseline")
    parser.add_argument(
        "--results-subdir",
        default="",
        metavar="NAME",
        help="結果寫入 fttransformer/<NAME>/（含 checkpoint、partial、raw、表格）；空=寫在 fttransformer 根目錄",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("FTTRANSFORMER_DEVICE", "auto"),
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

    sub = (args.results_subdir or "").strip().replace("\\", "/").strip("/")
    if sub and (Path(sub).name != sub or ".." in Path(sub).parts):
        raise SystemExit("--results-subdir 請使用單一資料夾名稱，勿含路徑跳脫")
    OUTPUT_DIR = (FT_BASE_DIR / sub) if sub else FT_BASE_DIR
    CHECKPOINT_PATH = OUTPUT_DIR / "bankruptcy_year_splits_fttransformer_checkpoint.json"
    PARTIAL_RAW_PATH = OUTPUT_DIR / "bankruptcy_year_splits_fttransformer_raw.partial.csv"

    logger = get_logger(
        "BK_YearSplits_FTTransformer",
        console=True,
        file=True,
        log_filename="BK_FTTransformer_run.log",
        log_file_mode="w",
    )
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"結果與進度檔目錄: {OUTPUT_DIR}")

    use_amp = not args.no_amp
    use_resume = (not args.no_resume) and os.getenv("FTTRANSFORMER_RESUME", "1") != "0"
    if use_resume:
        all_rows, completed_stage_keys = _load_progress(logger)
    else:
        _clear_progress_files(logger)
        all_rows, completed_stage_keys = [], set()

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device_info = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
        logger.info(f"Model device -> --device={args.device} (env FTTRANSFORMER_DEVICE 可覆寫預設 auto)")
        logger.info(
            f"Runtime torch={torch.__version__}, cuda_available={cuda_ok}, "
            f"reported_name={device_info}, use_amp={use_amp}"
        )
    except Exception as e:
        logger.warning(f"無法取得 torch/cuda 狀態：{e}")

    retrain_done = False
    for label, old_end_year in YEAR_SPLITS:
        logger.info(f"\n{'='*60}\nSplit: {label}  (Old<={old_end_year}, New={old_end_year+1}-2014, Test=2015-2018)\n{'='*60}")
        try:
            X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, year_test = get_bankruptcy_year_split(
                logger,
                old_end_year=old_end_year,
                return_years=True,
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
                device=args.device,
                use_amp=use_amp,
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
    summary = _mean_pivot_by_method_sampling(df_raw, "AUC")
    logger.info("\nAUC 摘要（method × sampling 平均，跨各年份切割）:\n" + summary.to_string())
    _clear_progress_files(logger)
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
