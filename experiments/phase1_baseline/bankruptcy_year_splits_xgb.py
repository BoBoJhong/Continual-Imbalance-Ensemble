"""
Phase 1 - Bankruptcy 年份切割基準線實驗（XGBoost）
=======================================================
固定 Test = 2015-2018，對 1999-2014 定義 7 組 Old/New 切割（每組步距 2 年），
跑 4 種訓練策略 × 4 種採樣 = 16 種組合／每組切割（7 組 → 112 rows 原始結果）。

訓練策略（對齊 docs/研究方向.md Baselines + 集成對照）：
  - Old      : 只用歷史 (Old) 資料訓練（集成之 base learner，非 retrain）
  - New      : 只用新營運 (New) 資料訓練（集成之 base learner）
  - Retrain  : 歷史 + 新營運「全量」合併後訓練（Re-training baseline）
  - Finetune : 先以 Old 訓練，再以 New 接續訓練同一 booster（xgb_model=）；
               微調階段僅 New 樣本；閾值於 New 的 validation 上選取

採樣策略：none / undersampling / oversampling / hybrid

最終輸出：
  - results/phase1_baseline/xgb/bankruptcy_year_splits_xgb_raw.csv
  - results/phase1_baseline/xgb/bk_xgb_compact_AUC_only.csv / _F1_only.csv / _Recall_only.csv
  - results/phase1_baseline/xgb/bk_xgb_compact_summary.csv   （精簡：AUC/F1/G_Mean/Recall/Precision）
  - results/phase1_baseline/xgb/bk_xgb_table_{metric}_old.csv
  - results/phase1_baseline/xgb/bk_xgb_table_{metric}_retrain.csv
  - results/phase1_baseline/xgb/bk_xgb_table_{metric}_finetune.csv
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
# 精簡報告欄位順序（與常見論文表格一致：hybrid / none / over / under）
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]
METRICS             = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
# 精簡摘要檔包含的指標（全指標仍寫在 raw 與各 bk_xgb_table_*）
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
# 各寫一份 method×採樣 小表（方便貼簡報）
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]
OUTPUT_DIR          = project_root / "results" / "phase1_baseline" / "xgb"

# 欄標題對應：method 縮寫
METHOD_PREFIX = {"Old": "Old", "Retrain": "Retrain", "Finetune": "Finetune", "New": "New"}


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


def _scale_pos_weight(y: np.ndarray) -> float:
    yv = np.asarray(y).ravel()
    unique, counts = np.unique(yv, return_counts=True)
    if len(unique) != 2:
        return 1.0
    neg_count = counts[0] if unique[0] == 0 else counts[1]
    pos_count = counts[1] if unique[1] == 1 else counts[0]
    return float(neg_count / max(pos_count, 1))


def _finetune_eval(X_old, y_old, X_new, y_new, X_test_raw, y_test, sampler, strategy, tag, logger):
    """
    Fine-tuning：Scaler 僅在 Old 的 train fold 上 fit；先訓練 Old，再以 xgb_model 接續訓練 New。
    分類閾值在 New 的 validation 上選（與研究方向「微調階段僅新資料」一致）。
    """
    y_old_arr = np.asarray(y_old)
    y_new_arr = np.asarray(y_new)

    try:
        X_old_fit_raw, X_old_val_raw, y_old_fit, y_old_val = train_test_split(
            X_old, y_old_arr, test_size=0.2, random_state=42, stratify=y_old_arr
        )
    except ValueError:
        X_old_fit_raw, X_old_val_raw, y_old_fit, y_old_val = train_test_split(
            X_old, y_old_arr, test_size=0.2, random_state=42
        )

    try:
        X_new_fit_raw, X_new_val_raw, y_new_fit, y_new_val = train_test_split(
            X_new, y_new_arr, test_size=0.2, random_state=42, stratify=y_new_arr
        )
    except ValueError:
        X_new_fit_raw, X_new_val_raw, y_new_fit, y_new_val = train_test_split(
            X_new, y_new_arr, test_size=0.2, random_state=42
        )

    pre = DataPreprocessor()
    X_old_fit, X_old_val = pre.scale_features(X_old_fit_raw, X_old_val_raw, fit=True)
    _, X_new_fit = pre.scale_features(X_old_fit_raw, X_new_fit_raw, fit=False)
    _, X_new_val = pre.scale_features(X_old_fit_raw, X_new_val_raw, fit=False)
    _, X_test = pre.scale_features(X_old_fit_raw, X_test_raw, fit=False)

    X_r1, y_r1 = sampler.apply_sampling(X_old_fit, np.asarray(y_old_fit), strategy=strategy)
    model = XGBoostWrapper(name=f"{tag}_{strategy}")
    model.fit(X_r1, y_r1)

    X_r2, y_r2 = sampler.apply_sampling(X_new_fit, np.asarray(y_new_fit), strategy=strategy)
    model.model.set_params(scale_pos_weight=_scale_pos_weight(y_r2))
    model.model.fit(
        X_r2,
        y_r2,
        eval_set=[(X_r2, y_r2)],
        verbose=False,
        xgb_model=model.model.get_booster(),
    )

    y_proba_val = model.predict_proba(X_new_val)
    threshold, val_f1 = _select_threshold_from_validation(np.asarray(y_new_val), y_proba_val)

    y_t = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    metrics = compute_metrics(y_t, model.predict_proba(X_test), threshold=threshold)
    logger.info(
        f"    {tag:12s} {strategy:12s} [thr={threshold:.3f}, valF1={val_f1:.4f}]: "
        f"AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}"
    )
    return metrics


def run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger):
    """對一組切割跑全部 16 種組合，回傳 list of row dict。"""
    sampler = ImbalanceSampler()
    rows = []

    # Old
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_old, y_old, X_test, y_test, sampler, strat, "Old", logger)
        rows.append({"split": label, "method": "Old", "sampling": strat, **m})

    # New
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_new, y_new, X_test, y_test, sampler, strat, "New", logger)
        rows.append({"split": label, "method": "New", "sampling": strat, **m})

    # Retrain：歷史 + 新營運全量合併（研究方向之 Re-training）
    X_re = pd.concat([X_old, X_new], ignore_index=True)
    y_re = pd.concat(
        [y_old.reset_index(drop=True), y_new.reset_index(drop=True)], ignore_index=True
    )
    logger.info(f"  Retrain: concat Old+New full n={len(X_re)}")
    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_re, y_re, X_test, y_test, sampler, strat, "Retrain", logger)
        rows.append({"split": label, "method": "Retrain", "sampling": strat, **m})

    # Finetune
    for strat in SAMPLING_STRATEGIES:
        m = _finetune_eval(X_old, y_old, X_new, y_new, X_test, y_test, sampler, strat, "Finetune", logger)
        rows.append({"split": label, "method": "Finetune", "sampling": strat, **m})

    return rows


# ---------------------------------------------------------------------------
# 格式化 pivot 表格
# ---------------------------------------------------------------------------

def format_tables(df_raw, logger):
    """
    依訓練策略分別產出四張表，每個指標共 4 張 × 7 metrics = 28 個 CSV：

    Old / Retrain / Finetune / New 表：列 = 各 split 之年數標籤，欄 = 採樣策略
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

        # ---------- Retrain 表 ----------
        df_rt = df_raw[df_raw["method"] == "Retrain"]
        pivot_rt = (
            df_rt.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_rt.index = [f"{split_yr[s][0]}+{split_yr[s][1]}" for s in pivot_rt.index]
        pivot_rt["avg"] = pivot_rt.mean(axis=1)
        pivot_rt.loc["avg"] = pivot_rt.mean()
        pivot_rt.index.name = "old+new_years_retrain_full"
        out = OUTPUT_DIR / f"bk_xgb_table_{metric}_retrain.csv"
        pivot_rt.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        # ---------- Finetune 表 ----------
        df_ft = df_raw[df_raw["method"] == "Finetune"]
        pivot_ft = (
            df_ft.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_ft.index = [f"{split_yr[s][0]}+{split_yr[s][1]}" for s in pivot_ft.index]
        pivot_ft["avg"] = pivot_ft.mean(axis=1)
        pivot_ft.loc["avg"] = pivot_ft.mean()
        pivot_ft.index.name = "old+new_years_finetune"
        out = OUTPUT_DIR / f"bk_xgb_table_{metric}_finetune.csv"
        pivot_ft.to_csv(out, float_format="%.4f")
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


def _mean_pivot_by_method_sampling(df_raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """method × sampling，數值為跨各 split 之平均；欄位順序：hybrid, none, oversampling, undersampling。"""
    t = (
        df_raw.groupby(["method", "sampling"])[metric]
        .mean()
        .unstack("sampling")
        .reindex(index=["Old", "New", "Retrain", "Finetune"], columns=SAMPLING_REPORT_ORDER)
    )
    return t.round(4)


def export_compact_report(df_raw: pd.DataFrame, logger) -> None:
    """
    精簡報告：
      - bk_xgb_compact_summary.csv：長表，欄位 metric, method, hybrid, none, oversampling, undersampling
      - bk_xgb_compact_{AUC|F1|Recall}_only.csv：各指標之 method×採樣 小表（跨 split 平均）
    """
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
    path_all = OUTPUT_DIR / "bk_xgb_compact_summary.csv"
    long_df.to_csv(path_all, index=False, float_format="%.4f")
    logger.info(f"  Saved -> {path_all.name}")

    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pivot = _mean_pivot_by_method_sampling(df_raw, m)
        path_m = OUTPUT_DIR / f"bk_xgb_compact_{m}_only.csv"
        pivot.to_csv(path_m, float_format="%.4f")
        logger.info(f"  Saved -> {path_m.name}")


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
        logger.info(
            f"Split: {label}  (Old<={old_end_year}, New={old_end_year + 1}-2014, Test=2015-2018)"
        )
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

    logger.info("\n產出精簡摘要（跨各 split 平均）...")
    export_compact_report(df_raw, logger)

    logger.info("\n=== 完成 ===")
    summary = _mean_pivot_by_method_sampling(df_raw, "AUC")
    logger.info("\nAUC 摘要（method × sampling 平均，跨各年份切割）:\n" + summary.to_string())


if __name__ == "__main__":
    main()
