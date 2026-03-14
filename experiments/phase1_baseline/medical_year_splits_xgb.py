"""
Phase 1 - Medical 年份切割基準線實驗（XGBoost）
=======================================================
資料集：UCI Diabetes 130-US Hospitals (1999-2008，偽時序)
固定 Test = 2007-2008，對 1999-2006 定義 7 組 Old/New 切割（每組步距 1 年），
跑 3 種訓練策略 × 4 種採樣 = 12 種組合，共 84 rows 原始結果。

訓練策略：
  - Old      : 只用 Old 資料訓練
  - Old+New  : 合併 Old + New 訓練
  - New      : 只用 New 資料訓練

採樣策略：none / undersampling / oversampling / hybrid

最終輸出：
  - results/phase1_baseline/xgb/medical_year_splits_xgb_raw.csv
  - results/phase1_baseline/xgb/med_xgb_table_{metric}_{old|oldnew|new}.csv
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler, DataPreprocessor
from src.models import XGBoostWrapper
from src.evaluation import compute_metrics
from experiments._shared.common_dataset import MEDICAL_YEAR_SPLITS, get_medical_year_split

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
METRICS             = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
OUTPUT_DIR          = project_root / "results" / "phase1_baseline" / "xgb"

# 年份起點固定 1999
BASE_YEAR = 1999


# ---------------------------------------------------------------------------
# 訓練函式（統一回傳 dict of metrics）
# ---------------------------------------------------------------------------

def _train_eval(X_train, y_train, X_test, y_test, sampler, strategy, tag, logger):
    """套用採樣 → 訓練 XGBoost → 評估，回傳指標 dict。"""
    X_r, y_r = sampler.apply_sampling(X_train, np.asarray(y_train), strategy=strategy)
    model = XGBoostWrapper(name=f"{tag}_{strategy}")
    model.fit(X_r, y_r)
    y_t = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    metrics = compute_metrics(y_t, model.predict_proba(X_test))
    logger.info(f"    {tag:12s} {strategy:12s}: AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}")
    return metrics


def run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger):
    """對一組切割跑全部 12 種組合，回傳 list of row dict。"""
    sampler = ImbalanceSampler()
    rows = []

    # 各策略各自 fit scaler，確保縮放參數只來自當次訓練集
    def _scale(X_train, X_test_raw):
        pre = DataPreprocessor()
        X_tr_s, X_te_s = pre.scale_features(X_train, X_test_raw, fit=True)
        return X_tr_s, X_te_s

    # Old
    X_old_s, X_test_s_old = _scale(X_old, X_test)
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_old_s, y_old, X_test_s_old, y_test, sampler, strat, "Old", logger)
        rows.append({"split": label, "method": "Old", "sampling": strat, **m})

    # Old+New  (合併，以合併資料 fit scaler)
    X_combined = pd.concat([X_old, X_new], ignore_index=True)
    y_combined = pd.concat(
        [y_old.reset_index(drop=True), y_new.reset_index(drop=True)], ignore_index=True
    )
    X_combined_s, X_test_s_comb = _scale(X_combined, X_test)
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_combined_s, y_combined, X_test_s_comb, y_test, sampler, strat, "Old+New", logger)
        rows.append({"split": label, "method": "Old+New", "sampling": strat, **m})

    # New（scaler fit 在 New 資料上）
    X_new_s, X_test_s_new = _scale(X_new, X_test)
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_new_s, y_new, X_test_s_new, y_test, sampler, strat, "New", logger)
        rows.append({"split": label, "method": "New", "sampling": strat, **m})

    return rows


# ---------------------------------------------------------------------------
# 格式化 pivot 表格
# ---------------------------------------------------------------------------

def format_tables(df_raw, logger):
    """
    依訓練策略分別產出三張表，每個指標共 3 張 × 7 metrics = 21 個 CSV：

    Old 表：列 = old 年數（1yr~7yr），欄 = 採樣策略
    OldNew 表：列 = old+new 組合（1+7~7+1），欄 = 採樣策略
    New 表：列 = new 年數（7yr~1yr），欄 = 採樣策略
    """
    # split label → (old_yr, new_yr)
    split_yr = {label: (old_end - BASE_YEAR + 1, 2006 - old_end)
                for label, old_end in MEDICAL_YEAR_SPLITS}

    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]

    split_labels  = [label for label, _ in MEDICAL_YEAR_SPLITS]
    sampling_cols = SAMPLING_STRATEGIES

    for metric in METRICS:
        # ---------- Old 表 ----------
        df_old = df_raw[df_raw["method"] == "Old"]
        pivot_old = (
            df_old.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_old.index = [f"{split_yr[s][0]}yr" for s in pivot_old.index]
        pivot_old.loc["avg"] = pivot_old.mean()
        pivot_old.index.name = "old_years"
        out = OUTPUT_DIR / f"med_xgb_table_{metric}_old.csv"
        pivot_old.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        # ---------- OldNew 表 ----------
        df_on = df_raw[df_raw["method"] == "Old+New"]
        pivot_on = (
            df_on.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_on.index = [f"{split_yr[s][0]}+{split_yr[s][1]}" for s in pivot_on.index]
        pivot_on.loc["avg"] = pivot_on.mean()
        pivot_on.index.name = "old+new_years"
        out = OUTPUT_DIR / f"med_xgb_table_{metric}_oldnew.csv"
        pivot_on.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        # ---------- New 表 ----------
        df_new = df_raw[df_raw["method"] == "New"]
        pivot_new = (
            df_new.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_new.index = [f"{split_yr[s][1]}yr" for s in pivot_new.index]
        pivot_new.loc["avg"] = pivot_new.mean()
        pivot_new.index.name = "new_years"
        out = OUTPUT_DIR / f"med_xgb_table_{metric}_new.csv"
        pivot_new.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    logger = get_logger("Medical_YearSplits_XGB", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for label, old_end_year in MEDICAL_YEAR_SPLITS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Split: {label}  (Old<={old_end_year}, New={old_end_year+1}-2006, Test=2007-2008)")
        logger.info("="*60)
        try:
            X_old, y_old, X_new, y_new, X_test, y_test = get_medical_year_split(
                logger, old_end_year=old_end_year
            )
            rows = run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger)
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            import traceback; logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果，請確認資料檔存在（執行 python scripts/download_real_medical_data.py）。")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = OUTPUT_DIR / "medical_year_splits_xgb_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")

    logger.info("\n產出指標 pivot 表格...")
    format_tables(df_raw, logger)

    logger.info("\n=== 完成 ===")
    # 簡要摘要
    summary = (
        df_raw.groupby(["method", "sampling"])["AUC"]
        .mean()
        .unstack("sampling")
        .reindex(["Old", "Old+New", "New"])
    )
    logger.info("\nAUC 摘要（method × sampling 平均，跨7組切割）:\n" + summary.to_string())


if __name__ == "__main__":
    main()
