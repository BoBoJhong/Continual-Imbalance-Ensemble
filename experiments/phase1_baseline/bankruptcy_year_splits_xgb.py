"""
Phase 1 - Bankruptcy 年份切割基準線實驗（XGBoost）
=======================================================
固定 Test = 2015-2018，對 1999-2014 定義 7 組 Old/New 切割（每組步距 2 年），
跑 3 種訓練策略 × 4 種採樣 = 12 種組合，共 84 rows 原始結果。

訓練策略：
  - Old      : 只用 Old 資料訓練
  - Old+New  : 合併 Old + New 訓練
  - New      : 只用 New 資料訓練

採樣策略：none / undersampling / oversampling / hybrid

最終輸出：
  - results/phase1_baseline/xgb/bankruptcy_year_splits_xgb_raw.csv    (84 rows)
  - results/phase1_baseline/xgb/bk_xgb_table_{metric}_old.csv
  - results/phase1_baseline/xgb/bk_xgb_table_{metric}_oldnew.csv
  - results/phase1_baseline/xgb/bk_xgb_table_{metric}_new.csv
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
from src.models import XGBoostWrapper
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
METRICS             = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
OUTPUT_DIR          = project_root / "results" / "phase1_baseline" / "xgb"

# 欄標題對應：method 縮寫
METHOD_PREFIX = {"Old": "Old", "Old+New": "OldNew", "New": "New"}


# ---------------------------------------------------------------------------
# 訓練函式（統一回傳 dict of metrics）
# ---------------------------------------------------------------------------

def _select_threshold_from_validation(y_val, y_proba_val):
    """用 validation set 搜尋最佳 F1 閾值，避免固定規則。"""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)

def _train_eval(X_train_raw, y_train, X_test_raw, y_test, sampler, strategy, tag, logger):
    """先切 train/val，再只用 train fold fit scaler，避免 validation leakage。"""
    y_train_arr = np.asarray(y_train)

    try:
        X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
            X_train_raw, y_train_arr, test_size=0.2, random_state=42, stratify=y_train_arr
        )
    except ValueError:
        X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
            X_train_raw, y_train_arr, test_size=0.2, random_state=42
        )

    pre = DataPreprocessor()
    X_fit, X_val = pre.scale_features(X_fit_raw, X_val_raw, fit=True)
    _, X_test = pre.scale_features(X_fit_raw, X_test_raw, fit=False)

    X_r, y_r = sampler.apply_sampling(X_fit, np.asarray(y_fit), strategy=strategy)
    model = XGBoostWrapper(name=f"{tag}_{strategy}")
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


def run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger):
    """對一組切割跑全部 12 種組合，回傳 list of row dict。"""
    sampler = ImbalanceSampler()
    rows = []

    def _balanced_oldnew(Xo, yo, Xn, yn, split_label):
        """建立 split-aware 的 Old+New 訓練集：Old/New 各取相同筆數。"""
        n = min(len(Xo), len(Xn))
        if n == 0:
            return (
                pd.concat([Xo, Xn], ignore_index=True),
                pd.concat([yo.reset_index(drop=True), yn.reset_index(drop=True)], ignore_index=True),
            )

        split_seed = 42 + sum(ord(c) for c in str(split_label))
        rng = np.random.default_rng(split_seed)
        idx_old = rng.choice(len(Xo), size=n, replace=False)
        idx_new = rng.choice(len(Xn), size=n, replace=False)

        X_bal = pd.concat(
            [Xo.iloc[idx_old].reset_index(drop=True), Xn.iloc[idx_new].reset_index(drop=True)],
            ignore_index=True,
        )
        y_bal = pd.concat(
            [yo.iloc[idx_old].reset_index(drop=True), yn.iloc[idx_new].reset_index(drop=True)],
            ignore_index=True,
        )
        return X_bal, y_bal

    # Old
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_old, y_old, X_test, y_test, sampler, strat, "Old", logger)
        rows.append({"split": label, "method": "Old", "sampling": strat, **m})

    # Old+New  (split-aware 平衡混合，Old/New 各取相同筆數)
    X_combined, y_combined = _balanced_oldnew(X_old, y_old, X_new, y_new, label)
    logger.info(f"  Old+New balanced mix: Old={len(X_old)} New={len(X_new)} -> each={len(X_combined)//2}")
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_combined, y_combined, X_test, y_test, sampler, strat, "Old+New", logger)
        rows.append({"split": label, "method": "Old+New", "sampling": strat, **m})

    # New
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_new, y_new, X_test, y_test, sampler, strat, "New", logger)
        rows.append({"split": label, "method": "New", "sampling": strat, **m})

    return rows


# ---------------------------------------------------------------------------
# 格式化 pivot 表格
# ---------------------------------------------------------------------------

def format_tables(df_raw, logger):
    """
    依訓練策略分別產出三張表，每個指標共 3 張 × 7 metrics = 21 個 CSV：

    Old 表：列 = old 年數（2yr~14yr），欄 = 採樣策略
    OldNew 表：列 = old+new 組合（2+14~14+2），欄 = 採樣策略
    New 表：列 = new 年數（14yr~2yr），欄 = 採樣策略

    row label 只描述實際用到的資料，避免誤導。
    """
    # split label → (old_yr, new_yr)
    split_yr = {label: (old_end - 1998, 2014 - old_end)
                for label, old_end in YEAR_SPLITS}

    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]   # 欄 = 採樣策略

    split_labels = [label for label, _ in YEAR_SPLITS]
    sampling_cols = SAMPLING_STRATEGIES  # none / undersampling / oversampling / hybrid

    for metric in METRICS:
        # ---------- Old 表 ----------
        df_old = df_raw[df_raw["method"] == "Old"]
        pivot_old = (
            df_old.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_old.index = [f"{split_yr[s][0]}yr" for s in pivot_old.index]
        pivot_old["avg"] = pivot_old.mean(axis=1)
        pivot_old.loc["avg"] = pivot_old.mean()
        pivot_old.index.name = "old_years"
        out = OUTPUT_DIR / f"bk_xgb_table_{metric}_old.csv"
        pivot_old.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        # ---------- OldNew 表 ----------
        df_on = df_raw[df_raw["method"] == "Old+New"]
        pivot_on = (
            df_on.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_on.index = [f"{split_yr[s][0]}+{split_yr[s][1]}" for s in pivot_on.index]
        pivot_on["avg"] = pivot_on.mean(axis=1)
        pivot_on.loc["avg"] = pivot_on.mean()
        pivot_on.index.name = "old+new_years"
        out = OUTPUT_DIR / f"bk_xgb_table_{metric}_oldnew.csv"
        pivot_on.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        # ---------- New 表 ----------
        df_new = df_raw[df_raw["method"] == "New"]
        pivot_new = (
            df_new.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_new.index = [f"{split_yr[s][1]}yr" for s in pivot_new.index]
        pivot_new["avg"] = pivot_new.mean(axis=1)
        pivot_new.loc["avg"] = pivot_new.mean()
        pivot_new.index.name = "new_years"
        out = OUTPUT_DIR / f"bk_xgb_table_{metric}_new.csv"
        pivot_new.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    logger = get_logger("BK_YearSplits_XGB", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for label, old_end_year in YEAR_SPLITS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Split: {label}  (Old<=2{old_end_year}, New=2{old_end_year+1}-2014, Test=2015-2018)")
        logger.info("="*60)
        try:
            X_old, y_old, X_new, y_new, X_test, y_test = get_bankruptcy_year_split(
                logger, old_end_year=old_end_year
            )
            rows = run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger)
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            import traceback; logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果，請確認資料檔存在。")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = OUTPUT_DIR / "bankruptcy_year_splits_xgb_raw.csv"
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
    logger.info("\nAUC 摘要（method × sampling 平均，跨6組切割）:\n" + summary.to_string())


if __name__ == "__main__":
    main()
