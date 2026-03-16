"""
Phase 1 - Bankruptcy 年份切割基準線實驗（FT-Transformer）
=======================================================
架構：FTTransformer (rtdl)，Feature Tokenizer + Transformer Encoder
      d_token=64，n_blocks=3，attention_n_heads=8
不平衡處理：同 MLP/XGB 版，由 ImbalanceSampler 採樣後餵入模型

固定 Test = 2015-2018，7 組切割 × 3 策略 × 4 採樣 = 84 rows

安裝依賴：pip install rtdl torch

最終輸出：
  - results/phase1_baseline/bankruptcy_year_splits_fttransformer_raw.csv
  - results/phase1_baseline/bk_fttransformer_table_{metric}_{old|oldnew|new}.csv
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
METRICS             = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
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


def run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger):
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

    # Old+New（split-aware 平衡混合，Old/New 各取相同筆數）
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
            if method == "Old":
                pivot.index.name = "old_years"
            elif method == "Old+New":
                pivot.index.name = "old+new_years"
            else:
                pivot.index.name = "new_years"
            out = OUTPUT_DIR / f"bk_fttransformer_table_{metric}_{suffix}.csv"
            pivot.to_csv(out, float_format="%.4f")
            logger.info(f"  Saved -> {out.name}")


def main():
    logger = get_logger("BK_YearSplits_FTTransformer", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for label, old_end_year in YEAR_SPLITS:
        logger.info(f"\n{'='*60}\nSplit: {label}  (Old<={old_end_year}, New={old_end_year+1}-2014, Test=2015-2018)\n{'='*60}")
        try:
            X_old, y_old, X_new, y_new, X_test, y_test = get_bankruptcy_year_split(logger, old_end_year=old_end_year)
            all_rows.extend(run_split(label, X_old, y_old, X_new, y_new, X_test, y_test, logger))
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
    logger.info("\n=== 完成 ===")


if __name__ == "__main__":
    main()
