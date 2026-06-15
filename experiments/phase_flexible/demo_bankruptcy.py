"""
Flexible ROSS Demo：以破產資料示範「時間粒度可彈性切換」
=========================================================

這個檔案示範 FlexibleROSS 如何在相同資料集上用不同粒度執行：

  模式 A：granularity='year'   （和原始研究相同，年切）
  模式 B：granularity='sample' （不依時間，改以樣本數量切割）

與 phase4/phase5 完全獨立，不影響既有實驗。

執行方式：
    python experiments/phase_flexible/demo_bankruptcy.py
    python experiments/phase_flexible/demo_bankruptcy.py --mode year
    python experiments/phase_flexible/demo_bankruptcy.py --mode sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_logger, set_seed
from experiments.phase_flexible.flexible_ross import FlexibleROSS

# ── 資料路徑 ──────────────────────────────────────────────────────────────────
US_CSV = project_root / "data" / "raw" / "bankruptcy" / "american_bankruptcy_dataset.csv"
OUTPUT_DIR = project_root / "results" / "phase_flexible"

# ── 固定切割點（與原研究相同） ───────────────────────────────────────────────
TRAIN_START = 1999
TRAIN_END   = 2014
VAL_START   = 2012
VAL_END     = 2014
TEST_START  = 2015
TEST_END    = 2018


def load_data(logger) -> pd.DataFrame:
    """載入 US 破產資料，回傳含 fyear 與 target 的完整 DataFrame。"""
    if not US_CSV.exists():
        raise FileNotFoundError(
            f"找不到資料：{US_CSV}\n"
            "請下載 american_bankruptcy_dataset.csv 放到 data/raw/bankruptcy/"
        )
    df = pd.read_csv(US_CSV)
    df["target"] = (df["status_label"] == "failed").astype(int)
    drop_cols = ["company_name", "status_label"]
    if "Division" in df.columns:
        drop_cols.append("Division")
    df = df.drop(columns=drop_cols)
    logger.info(f"資料載入完成：{df.shape}，破產率={df['target'].mean()*100:.2f}%")
    return df


def run_year_mode(df: pd.DataFrame, logger) -> dict:
    """
    模式 A：granularity='year'
    與原始研究相同：搜尋 1999~2011 內哪一年最佳，驗證集 2012~2014，測試集 2015~2018。
    """
    logger.info("\n" + "="*60)
    logger.info("模式 A：granularity='year'（與原研究相同）")
    logger.info("="*60)

    # 切割：訓練搜尋範圍（不含驗證、測試），驗證，測試
    train_mask = (df["fyear"] >= TRAIN_START) & (df["fyear"] < VAL_START)
    val_mask   = (df["fyear"] >= VAL_START)   & (df["fyear"] <= VAL_END)
    test_mask  = (df["fyear"] >= TEST_START)  & (df["fyear"] <= TEST_END)

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_val   = df.loc[val_mask].reset_index(drop=True)
    df_test  = df.loc[test_mask].reset_index(drop=True)

    logger.info(
        f"Train={len(df_train)}（fyear {TRAIN_START}~{VAL_START-1}） "
        f"Val={len(df_val)}（{VAL_START}~{VAL_END}） "
        f"Test={len(df_test)}（{TEST_START}~{TEST_END}）"
    )

    ross = FlexibleROSS(
        time_col="fyear",
        granularity="year",
        min_old_units=3,    # Old 至少 3 年
        min_new_units=1,    # New 至少 1 年
        boundary_metric="AUC",
        weight_metric="F1",
    )

    results = ross.run(
        df_train=df_train,
        target_col="target",
        df_val=df_val,
        df_test=df_test,
    )
    return results


def run_sample_mode(df: pd.DataFrame, logger) -> dict:
    """
    模式 B：granularity='sample'
    不依年份，改以每 500 筆為一個時間單位，同樣用最後 20% 當驗證集。
    示範：資料不一定要有時間欄也能跑 ROSS。
    """
    logger.info("\n" + "="*60)
    logger.info("模式 B：granularity='sample'（不依年份，以樣本數切）")
    logger.info("="*60)

    # 先照時間排序（確保順序性），再切 train/val/test
    df_sorted = df.sort_values("fyear").reset_index(drop=True)

    # 用原本的時間切割（邏輯同模式 A），但 FlexibleROSS 改以 sample 模式
    train_mask = (df_sorted["fyear"] >= TRAIN_START) & (df_sorted["fyear"] < VAL_START)
    val_mask   = (df_sorted["fyear"] >= VAL_START)   & (df_sorted["fyear"] <= VAL_END)
    test_mask  = (df_sorted["fyear"] >= TEST_START)  & (df_sorted["fyear"] <= TEST_END)

    df_train = df_sorted.loc[train_mask].reset_index(drop=True)
    df_val   = df_sorted.loc[val_mask].reset_index(drop=True)
    df_test  = df_sorted.loc[test_mask].reset_index(drop=True)

    # sample 模式下需要一個「假時間欄」代表樣本順序
    df_train = df_train.copy()
    df_train["sample_idx"] = np.arange(len(df_train))

    logger.info(
        f"Train={len(df_train)} 筆（sample 模式，每 500 筆一個單位） "
        f"Val={len(df_val)}  Test={len(df_test)}"
    )

    ross = FlexibleROSS(
        time_col="sample_idx",
        granularity="sample",
        min_old_units=3,      # Old 至少 3 個 chunk（3×500 = 1500 筆）
        min_new_units=1,      # New 至少 1 個 chunk
        boundary_metric="AUC",
        weight_metric="F1",
        sample_step=500,
    )

    results = ross.run(
        df_train=df_train,
        target_col="target",
        df_val=df_val,
        df_test=df_test,
    )
    return results


def save_results(results: dict, tag: str, logger) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 儲存候選邊界
    cand_path = OUTPUT_DIR / f"flexible_ross_{tag}_candidates.csv"
    results["candidates"].to_csv(cand_path, index=False, float_format="%.6f")

    # 儲存加權掃描
    sweep_path = OUTPUT_DIR / f"flexible_ross_{tag}_weight_sweep.csv"
    results["weight_sweep"].to_csv(sweep_path, index=False, float_format="%.6f")

    logger.info(f"  候選邊界 → {cand_path}")
    logger.info(f"  加權掃描 → {sweep_path}")

    # 印出測試成績
    if results["test_metrics"]:
        m = results["test_metrics"]
        logger.info(
            f"\n[{tag}] 最終測試成績 "
            f"（boundary={results['best_boundary_label']}, w_new={results['best_w_new']:.2f}）\n"
            f"  AUC={m['AUC']:.4f}  F1={m['F1']:.4f}  "
            f"Recall={m['Recall']:.4f}  Precision={m['Precision']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Flexible ROSS Demo")
    parser.add_argument(
        "--mode",
        choices=["year", "sample", "both"],
        default="both",
        help="執行哪個示範模式（預設：both）",
    )
    args = parser.parse_args()

    logger = get_logger("FlexibleROSS_Demo", console=True, file=True)
    set_seed(42)

    df = load_data(logger)

    if args.mode in ("year", "both"):
        results_year = run_year_mode(df, logger)
        save_results(results_year, tag="year", logger=logger)

    if args.mode in ("sample", "both"):
        results_sample = run_sample_mode(df, logger)
        save_results(results_sample, tag="sample", logger=logger)

    if args.mode == "both":
        logger.info("\n" + "="*60)
        logger.info("兩種模式成績比較")
        logger.info("="*60)

        def fmt(r):
            m = r.get("test_metrics", {})
            return (
                f"boundary={r['best_boundary_label']!s:>12s}  "
                f"w_new={r['best_w_new']:.2f}  "
                f"AUC={m.get('AUC', 0):.4f}  F1={m.get('F1', 0):.4f}"
            ) if m else "（無測試集成績）"

        logger.info(f"Year   模式：{fmt(results_year)}")
        logger.info(f"Sample 模式：{fmt(results_sample)}")


if __name__ == "__main__":
    main()
