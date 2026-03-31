"""
Phase 1 - Bankruptcy 年份切割基準線實驗（PyTorch MLP 深度學習）
================================================================
與 bankruptcy_year_splits_xgb.py **相同協定**（請與該檔同步維護）：
  固定 Test = 2015-2018；訓練窗 1999–2014；**15 組** Old/New 切割（`YEAR_SPLITS`）。
  Validation：訓練段內依 **fyear 逐年**各抽約 20% 合併為校準集（與 XGB/LR/RF 一致）。
  **Retrain 僅在全部迭代中第一次執行**，故跑滿 15 折時 raw = **184** 列。

勿與下列檔案混淆：
  - bankruptcy_year_splits_mlp.py：sklearn MLPClassifier，**三策略** Old / Old+New / New，
    輸出 bankruptcy_year_splits_mlp_raw.csv（84 rows），**不**與 XGB 四策略對齊。
  - 本檔：PyTorch MLP，**四策略**與 XGB 相同，method **不應** 出現「Old+New」。

訓練策略（對齊 docs/研究方向.md Baselines + 集成對照）：
  - Old      : 只用歷史 (Old) 資料訓練
  - New      : 只用新營運 (New) 資料訓練
  - Retrain  : 歷史 + 新營運「全量」合併後訓練（實驗中只產生一組）
  - Finetune : 先以 Old 訓練，再以 continue_fit 於 New 接續訓練同一 MLP；
               微調階段之 BCE pos_weight 與 XGB 第二段相同，以採樣後整段 y_r2 計算；
               閾值於 **合併之 Old+New validation** 上選取

採樣策略：none / undersampling / oversampling / hybrid

用法：
  python experiments/phase1_baseline/bankruptcy_year_splits_torch_mlp.py
  python experiments/phase1_baseline/bankruptcy_year_splits_torch_mlp.py --splits split_2+14

最終輸出：
  - results/phase1_baseline/torch_mlp/bankruptcy_year_splits_torch_mlp_raw.csv
  - results/phase1_baseline/torch_mlp/bk_torch_mlp_compact_AUC_only.csv / _F1_only.csv / _Recall_only.csv
  - results/phase1_baseline/torch_mlp/bk_torch_mlp_compact_summary.csv
  - results/phase1_baseline/torch_mlp/bk_torch_mlp_table_{metric}_old.csv
  - results/phase1_baseline/torch_mlp/bk_torch_mlp_table_{metric}_retrain.csv
  - results/phase1_baseline/torch_mlp/bk_torch_mlp_table_{metric}_finetune.csv
  - results/phase1_baseline/torch_mlp/bk_torch_mlp_table_{metric}_new.csv
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split
from src.data import DataPreprocessor, ImbalanceSampler
from src.evaluation import compute_metrics
from src.models.torch_mlp_wrapper import TorchTabularMLPWrapper
from src.utils import get_logger, set_seed

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
SAMPLING_REPORT_ORDER = ["hybrid", "none", "oversampling", "undersampling"]
METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
COMPACT_ONLY_METRICS = ["AUC", "F1", "Recall"]
OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "torch_mlp"

METHOD_PREFIX = {"Old": "Old", "Retrain": "Retrain", "Finetune": "Finetune", "New": "New"}


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
    X_val_raw=None,
    y_val=None,
    year_train=None,
):
    """先切 train/val（或沿用外傳 val），再只用 train fold fit scaler，避免 validation leakage。"""
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
                    X_train_raw,
                    y_train_arr,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_train_arr,
                )
            except ValueError:
                X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
                    X_train_raw,
                    y_train_arr,
                    test_size=0.2,
                    random_state=42,
                )
    else:
        X_fit_raw = X_train_raw
        y_fit = y_train_arr

    pre = DataPreprocessor()
    X_fit, X_val = pre.scale_features(X_fit_raw, X_val_raw, fit=True)
    _, X_test = pre.scale_features(X_fit_raw, X_test_raw, fit=False)

    X_r, y_r = sampler.apply_sampling(X_fit, np.asarray(y_fit), strategy=strategy)
    model = TorchTabularMLPWrapper(name=f"{tag}_{strategy}", seed=42)
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
    """與 bankruptcy_year_splits_xgb._scale_pos_weight 相同（Finetune 第二段對齊）。"""
    yv = np.asarray(y).ravel()
    unique, counts = np.unique(yv, return_counts=True)
    if len(unique) != 2:
        return 1.0
    neg_count = counts[0] if unique[0] == 0 else counts[1]
    pos_count = counts[1] if unique[1] == 1 else counts[0]
    return float(neg_count / max(pos_count, 1))


def _finetune_eval(X_old, y_old, year_old, X_new, y_new, year_new, X_test_raw, y_test, sampler, strategy, tag, logger):
    """
    Fine-tuning：Scaler 僅在 Old 的 train fold 上 fit；先訓練 Old，再以 continue_fit 接續訓練 New。
    第二段 pos_weight 以採樣後整段 y_r2 計算（對齊 XGB 之 scale_pos_weight(y_r2)）。
    閾值在合併之 Old+New validation 上選（與 bankruptcy_year_splits_xgb 一致）。
    """
    X_old_fit_raw, y_old_fit, X_old_val_raw, y_old_val = _split_fit_val_by_year(X_old, y_old, year_old)
    X_new_fit_raw, y_new_fit, X_new_val_raw, y_new_val = _split_fit_val_by_year(X_new, y_new, year_new)

    pre = DataPreprocessor()
    X_old_fit, X_old_val = pre.scale_features(X_old_fit_raw, X_old_val_raw, fit=True)
    _, X_new_fit = pre.scale_features(X_old_fit_raw, X_new_fit_raw, fit=False)
    _, X_new_val = pre.scale_features(X_old_fit_raw, X_new_val_raw, fit=False)
    _, X_test = pre.scale_features(X_old_fit_raw, X_test_raw, fit=False)

    X_r1, y_r1 = sampler.apply_sampling(X_old_fit, np.asarray(y_old_fit), strategy=strategy)
    model = TorchTabularMLPWrapper(name=f"{tag}_{strategy}", seed=42)
    model.fit(X_r1, y_r1)

    X_r2, y_r2 = sampler.apply_sampling(X_new_fit, np.asarray(y_new_fit), strategy=strategy)
    model.continue_fit(
        X_r2,
        y_r2,
        lr_factor=0.25,
        pos_weight_override=_scale_pos_weight(y_r2),
    )

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
    """對一組切割跑 Old/New/Finetune（與可選的 Retrain），回傳 list of row dict。"""
    sampler = ImbalanceSampler()
    rows = []

    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_old, y_old, X_test, y_test, sampler, strat, "Old", logger, year_train=year_old)
        rows.append({"split": label, "method": "Old", "sampling": strat, **m})

    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(X_new, y_new, X_test, y_test, sampler, strat, "New", logger, year_train=year_new)
        rows.append({"split": label, "method": "New", "sampling": strat, **m})

    if include_retrain:
        X_fit_re, y_fit_re, X_val_re, y_val_re = _build_retrain_fit_val(X_old, y_old, year_old, X_new, y_new, year_new)
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
            )
            rows.append({"split": label, "method": "Retrain", "sampling": strat, **m})

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

    return rows


def format_tables(df_raw, logger, split_labels_filter=None):
    """
    依訓練策略分別產出四張表，每個指標共 4 張 × len(METRICS) 個 CSV。
    Old / Finetune / New：列 = 各 split；Retrain 為單列（與 XGB 之 bk_xgb_table_*_retrain 一致）。
    split_labels_filter：若指定（例如 --splits 子集），僅輸出該些列；預設全部 YEAR_SPLITS。
    """
    split_yr = {label: (old_end - 1998, 2014 - old_end) for label, old_end in YEAR_SPLITS}

    df_raw = df_raw.copy()
    df_raw["col"] = df_raw["sampling"]

    split_labels = [label for label, _ in YEAR_SPLITS]
    if split_labels_filter is not None:
        split_labels = [s for s in split_labels if s in split_labels_filter]
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
        out = OUTPUT_DIR / f"bk_torch_mlp_table_{metric}_old.csv"
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
        out = OUTPUT_DIR / f"bk_torch_mlp_table_{metric}_retrain.csv"
        pivot_rt.to_csv(out, float_format="%.4f")
        logger.info(f"  Saved -> {out.name}")

        df_ft = df_raw[df_raw["method"] == "Finetune"]
        pivot_ft = (
            df_ft.pivot(index="split", columns="col", values=metric)
            .reindex(index=split_labels, columns=sampling_cols)
        )
        pivot_ft.index = [f"{split_yr[s][0]}+{split_yr[s][1]}" for s in pivot_ft.index]
        pivot_ft["avg"] = pivot_ft.mean(axis=1)
        pivot_ft.loc["avg"] = pivot_ft.mean()
        pivot_ft.index.name = "old+new_years_finetune"
        out = OUTPUT_DIR / f"bk_torch_mlp_table_{metric}_finetune.csv"
        pivot_ft.to_csv(out, float_format="%.4f")
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
        out = OUTPUT_DIR / f"bk_torch_mlp_table_{metric}_new.csv"
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
      - bk_torch_mlp_compact_summary.csv：長表，欄位 metric, method, hybrid, none, oversampling, undersampling
      - bk_torch_mlp_compact_{AUC|F1|Recall}_only.csv：各指標之 method×採樣 小表（跨 split 平均）
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
    path_all = OUTPUT_DIR / "bk_torch_mlp_compact_summary.csv"
    long_df.to_csv(path_all, index=False, float_format="%.4f")
    logger.info(f"  Saved -> {path_all.name}")

    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pivot = _mean_pivot_by_method_sampling(df_raw, m)
        path_m = OUTPUT_DIR / f"bk_torch_mlp_compact_{m}_only.csv"
        pivot.to_csv(path_m, float_format="%.4f")
        logger.info(f"  Saved -> {path_m.name}")


def main():
    parser = argparse.ArgumentParser(description="Bankruptcy 年份切割 — PyTorch MLP baseline（對齊 XGB 四策略）")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="只跑指定 split（例如 split_2+14）；預設跑 YEAR_SPLITS 全部",
    )
    args = parser.parse_args()

    logger = get_logger("BK_YearSplits_TorchMLP", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_iter = list(YEAR_SPLITS)
    if args.splits:
        want = set(args.splits)
        split_iter = [(lab, yr) for lab, yr in YEAR_SPLITS if lab in want]
        unknown = want - {lab for lab, _ in split_iter}
        if unknown:
            logger.error(f"未知的 split 標籤: {sorted(unknown)}")
        if not split_iter:
            logger.error("沒有可執行的 split，請檢查 --splits")
            return

    all_rows = []
    retrain_done = False

    for label, old_end_year in split_iter:
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Split: {label}  (Old<={old_end_year}, New={old_end_year + 1}-2014, Test=2015-2018)"
        )
        logger.info("=" * 60)
        try:
            X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, _year_test = (
                get_bankruptcy_year_split(
                    logger, old_end_year=old_end_year, return_years=True
                )
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
            )
            all_rows.extend(rows)
            retrain_done = True
        except Exception as e:
            logger.error(f"[ERROR] {label}: {e}")
            logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果，請確認資料檔存在且已安裝 PyTorch（pip install torch）。")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = OUTPUT_DIR / "bankruptcy_year_splits_torch_mlp_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果已儲存 -> {raw_path.name}  ({len(df_raw)} rows)")

    expected_methods = {"Old", "New", "Retrain", "Finetune"}
    got_methods = set(df_raw["method"].unique())
    n_splits_ran = len({r["split"] for r in all_rows})
    # 第一次迭代含 Retrain（+4 列），其餘各 12 列（Old/New/Finetune ×4）
    expect_rows = 16 + 12 * (n_splits_ran - 1) if n_splits_ran > 0 else 0
    if len(df_raw) != expect_rows or got_methods != expected_methods:
        logger.warning(
            "輸出與 XGB 四策略協定不一致：rows=%s（預期 n_splits=%s → %s 列）、method=%s（預期 %s）。"
            "若曾見 Old+New 或 84/112 rows，代表舊版或誤跑 bankruptcy_year_splits_mlp.py；請重跑本腳本。",
            len(df_raw),
            n_splits_ran,
            expect_rows,
            sorted(got_methods),
            sorted(expected_methods),
        )

    logger.info("\n產出指標 pivot 表格...")
    filter_labels = [lab for lab, _ in split_iter] if args.splits else None
    format_tables(df_raw, logger, split_labels_filter=filter_labels)

    logger.info("\n產出精簡摘要（跨各 split 平均）...")
    export_compact_report(df_raw, logger)

    logger.info("\n=== 完成 ===")
    summary = _mean_pivot_by_method_sampling(df_raw, "AUC")
    logger.info("\nAUC 摘要（method × sampling 平均，跨各年份切割）:\n" + summary.to_string())


if __name__ == "__main__":
    main()
