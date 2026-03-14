"""
Phase 1 - Stock 年份切割基準線實驗（XGBoost）
=======================================================
資料集：S&P 500 (SPX)，2000-2020 日頻資料
固定 Test = 2017-2020，訓練窗口 2001-2016（16 年），定義 7 組對稱切割（步距 2 年）：
跑 3 種訓練策略 × 4 種採樣 = 12 種組合，共 84 rows 原始結果。

訓練策略：
  - Old      : 只用 Old 資料訓練
  - Old+New  : 合併 Old + New 訓練
  - New      : 只用 New 資料訓練

採樣策略：none / undersampling / oversampling / hybrid

最終輸出：
  - results/phase1_baseline/stock_year_splits_xgb_raw.csv    (84 rows)
  - results/phase1_baseline/xgb/stock_year_splits_xgb_raw.csv
  - results/phase1_baseline/xgb/stk_xgb_table_{metric}_{old|oldnew|new}.csv
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
from experiments._shared.common_dataset import (
    STOCK_YEAR_SPLITS, STOCK_BASE_YEAR, get_stock_year_split,
)

# 訓練窗口固定 2001-2016（16 年，與破產 1999-2014 相同長度）
# 切割為 2+14, 4+12, 6+10, 8+8(中心), 10+6, 12+4, 14+2

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
METRICS             = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
OUTPUT_DIR          = project_root / "results" / "phase1_baseline" / "xgb"
TICKER              = "spx"     # S&P 500


# ---------------------------------------------------------------------------
# 訓練函式
# ---------------------------------------------------------------------------

def _train_eval(X_train, y_train, X_test, y_test, sampler, strategy, tag, logger):
    """套用採樣 → 訓練 XGBoost → 評估，回傳指標 dict。

    使用訓練集正類比例作為分類閾值（class-prior threshold），
    避免股票崩盤事件極少（~7%）時 threshold=0.5 導致全部預測為 0。
    """
    y_train_arr = np.asarray(y_train)
    X_r, y_r = sampler.apply_sampling(X_train, y_train_arr, strategy=strategy)
    model = XGBoostWrapper(name=f"{tag}_{strategy}")
    model.fit(X_r, y_r)
    y_t = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    # class-prior threshold：以原始（採樣前）訓練集正類比例決定閾值
    threshold = float(y_train_arr.mean())
    metrics = compute_metrics(y_t, model.predict_proba(X_test), threshold=threshold)
    logger.info(
        f"    {tag:12s} {strategy:12s} [thr={threshold:.3f}]: "
        f"AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}"
    )
    return metrics


def run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger):
    """對一組切割跑全部 12 種組合，回傳 list of row dict。"""
    sampler = ImbalanceSampler()
    rows = []

    def _scale(X_train, X_test_raw):
        pre = DataPreprocessor()
        X_tr_s, X_te_s = pre.scale_features(X_train, X_test_raw, fit=True)
        return X_tr_s, X_te_s

    # Old
    X_old_s, X_test_s_old = _scale(X_old, X_test)
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_old_s, y_old, X_test_s_old, y_test, sampler, strat, "Old", logger)
        rows.append({"split": label, "method": "Old", "sampling": strat, **m})

    # Old+New
    X_combined = pd.concat([X_old, X_new], ignore_index=True)
    y_combined = pd.concat(
        [y_old.reset_index(drop=True), y_new.reset_index(drop=True)], ignore_index=True
    )
    X_combined_s, X_test_s_comb = _scale(X_combined, X_test)
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_combined_s, y_combined, X_test_s_comb, y_test, sampler, strat, "Old+New", logger)
        rows.append({"split": label, "method": "Old+New", "sampling": strat, **m})

    # New
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
    Old 表：列 = old 年數（2yr~14yr），欄 = 採樣策略
    OldNew 表：列 = old+new 組合（2+14~14+2），欄 = 採樣策略
    New 表：列 = new 年數（14yr~2yr），欄 = 採樣策略
    """
    # split label → (old_yr, new_yr)
    split_yr = {label: (old_end - STOCK_BASE_YEAR + 1, 2016 - old_end)
                for label, old_end in STOCK_YEAR_SPLITS}

    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]

    split_labels  = [label for label, _ in STOCK_YEAR_SPLITS]
    sampling_cols = SAMPLING_STRATEGIES

    for metric in METRICS:
        # ---------- Old ----------
        df_old = df_raw[df_raw["method"] == "Old"]
        pivot_old = (
            df_old.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_old.index = [f"{split_yr[s][0]}yr" for s in pivot_old.index]
        pivot_old.loc["avg"] = pivot_old.mean()
        pivot_old.index.name = "old_years"
        out = OUTPUT_DIR / f"stk_xgb_table_{metric}_old.csv"
        pivot_old.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        # ---------- OldNew ----------
        df_on = df_raw[df_raw["method"] == "Old+New"]
        pivot_on = (
            df_on.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_on.index = [f"{split_yr[s][0]}+{split_yr[s][1]}" for s in pivot_on.index]
        pivot_on.loc["avg"] = pivot_on.mean()
        pivot_on.index.name = "old+new_years"
        out = OUTPUT_DIR / f"stk_xgb_table_{metric}_oldnew.csv"
        pivot_on.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        # ---------- New ----------
        df_new = df_raw[df_raw["method"] == "New"]
        pivot_new = (
            df_new.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_new.index = [f"{split_yr[s][1]}yr" for s in pivot_new.index]
        pivot_new.loc["avg"] = pivot_new.mean()
        pivot_new.index.name = "new_years"
        out = OUTPUT_DIR / f"stk_xgb_table_{metric}_new.csv"
        pivot_new.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    logger = get_logger("Stock_YearSplits_XGB", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for label, old_end_year in STOCK_YEAR_SPLITS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Split: {label}  (Old=2001-{old_end_year}, New={old_end_year+1}-2016, Test=2017-2020)  ticker={TICKER}")
        logger.info("="*60)
        try:
            X_old, y_old, X_new, y_new, X_test, y_test = get_stock_year_split(
                logger, old_end_year=old_end_year, ticker=TICKER
            )
            rows = run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger)
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            import traceback; logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果，請確認資料檔存在（data/raw/stock/stock_spx.csv）。")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = OUTPUT_DIR / "stock_year_splits_xgb_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")

    logger.info("\n產出指標 pivot 表格...")
    format_tables(df_raw, logger)

    logger.info("\n=== 完成 ===")
    summary = (
        df_raw.groupby(["method", "sampling"])["AUC"]
        .mean()
        .unstack("sampling")
        .reindex(["Old", "Old+New", "New"])
    )
    logger.info("\nAUC 摘要（method × sampling 平均，跨7組切割，2+14…14+2）:\n" + summary.to_string())


if __name__ == "__main__":
    main()
