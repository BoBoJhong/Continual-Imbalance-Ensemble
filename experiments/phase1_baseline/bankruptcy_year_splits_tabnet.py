"""
Phase 1 - Bankruptcy 年份切割基準線實驗（TabNet）
=======================================================
架構：TabNetClassifier (pytorch-tabnet)
      Sequential Attention，n_d=n_a=32，n_steps=5
不平衡處理：同 MLP/XGB 版，由 ImbalanceSampler 採樣後餵入模型

固定 Test = 2015-2018，7 組切割 × 3 策略 × 4 採樣 = 84 rows

安裝依賴：pip install pytorch-tabnet

最終輸出：
  - results/phase1_baseline/bankruptcy_year_splits_tabnet_raw.csv
  - results/phase1_baseline/bk_tabnet_table_{metric}_{old|oldnew|new}.csv
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler, DataPreprocessor
from src.models import TabNetWrapper
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
METRICS             = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
OUTPUT_DIR          = project_root / "results" / "phase1_baseline" / "tabnet"


def _train_eval(X_train, y_train, X_test, y_test, sampler, strategy, tag, logger):
    X_r, y_r = sampler.apply_sampling(X_train, np.asarray(y_train), strategy=strategy)
    model = TabNetWrapper(name=f"{tag}_{strategy}")
    model.fit(X_r, y_r)
    y_t = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    metrics = compute_metrics(y_t, model.predict_proba(X_test))
    logger.info(f"    {tag:12s} {strategy:12s}: AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}")
    return metrics


def run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger):
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
    y_combined = pd.concat([y_old.reset_index(drop=True), y_new.reset_index(drop=True)], ignore_index=True)
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


def format_tables(df_raw, logger):
    split_yr = {label: (old_end - 1998, 2014 - old_end) for label, old_end in YEAR_SPLITS}
    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]
    split_labels  = [label for label, _ in YEAR_SPLITS]
    sampling_cols = SAMPLING_STRATEGIES

    for metric in METRICS:
        for method, suffix, idx_fn in [
            ("Old",     "old",    lambda s: f"{split_yr[s][0]}yr"),
            ("Old+New", "oldnew", lambda s: f"{split_yr[s][0]}+{split_yr[s][1]}"),
            ("New",     "new",    lambda s: f"{split_yr[s][1]}yr"),
        ]:
            pivot = (
                df_raw[df_raw["method"] == method]
                .pivot(index="split", columns="col", values=metric)
                .reindex(index=split_labels, columns=sampling_cols)
            )
            pivot.index = [idx_fn(s) for s in pivot.index]
            pivot["avg"] = pivot.mean(axis=1)
            pivot.loc["avg"] = pivot.mean()
            out = OUTPUT_DIR / f"bk_tabnet_table_{metric}_{suffix}.csv"
            pivot.to_csv(out, float_format="%.4f")
            logger.info(f"  Saved -> {out.name}")


def main():
    logger = get_logger("BK_YearSplits_TabNet", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for label, old_end_year in YEAR_SPLITS:
        logger.info(f"\n{'='*60}\nSplit: {label}  (Old<=2{old_end_year}, New=2{old_end_year+1}-2014, Test=2015-2018)\n{'='*60}")
        try:
            X_old, y_old, X_new, y_new, X_test, y_test = get_bankruptcy_year_split(logger, old_end_year=old_end_year)
            all_rows.extend(run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger))
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            import traceback; logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果"); return

    df_raw = pd.DataFrame(all_rows)
    raw_path = OUTPUT_DIR / "bankruptcy_year_splits_tabnet_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")
    logger.info("\n產出指標 pivot 表格...")
    format_tables(df_raw, logger)
    logger.info("\n=== 完成 ===")


if __name__ == "__main__":
    main()
