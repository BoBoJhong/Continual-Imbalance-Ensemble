"""
Phase 1 – MOVER 年份切割基準線實驗（XGBoost）
=================================================
資料集：MOVER 術後併發症（2017–2023，EPIC）
標籤：any_complication（二元，正例率 ~3.75%）

年份分配：
  2017：45 筆（極小，併入 Old 下限）
  2018：2,662  2019：11,546  2020：15,337  2021：15,811  → 訓練窗
  2022：15,618 → 固定 Test（未來資料，概念漂移重點）
  2023：3,335  → 次要 Drift check（不計入主實驗，但寫入額外欄位）

固定 Test = 2022；訓練窗 2018–2021（4 年）。
依 old_end_year 滑動，共 3 組：
  split_1+3：Old=2018（1yr），          New=2019-2021（3yr）
  split_2+2：Old=2018-2019（2yr），     New=2020-2021（2yr）
  split_3+1：Old=2018-2020（3yr），     New=2021（1yr）

訓練策略：Old / New / Retrain（Old+New）
採樣策略：none / undersampling / oversampling / hybrid

輸出：results/phase1_baseline/xgb_mover/
  mover_year_splits_xgb_raw.csv         ← 所有結果
  mover_xgb_compact_summary.csv
  mover_xgb_compact_{AUC|F1|Recall}_only.csv
  mover_xgb_table_{metric}_{old|retrain|new}.csv
  mover_xgb_drift2023_summary.csv       ← 2023 drift 評估

用法：
  python experiments/phase1_baseline/mover_year_splits_xgb.py
  python experiments/phase1_baseline/mover_year_splits_xgb.py --tuning tuned --tune-n-iter 48
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataPreprocessor, ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_logger, set_seed

from experiments._shared.baseline_val_search import (
    export_tuning_log,
    search_xgb_on_val,
    tuning_meta,
)

# ── 常數 ────────────────────────────────────────────────────────────────────
MOVER_LABELED_PATH = project_root / "data/processed/MOVER/model/mover_labeled_fixed.csv"

FEATURE_COLS = [
    # 連續
    "age_years",
    "ASA_RATING_C",
    "or_duration_hours",
    "anesthesia_duration_hours",
    "procedure_event_count",
    "procedure_event_n_unique",
    "evt_span_minutes",
    # 事件 flag（0/1）
    "evt_sign_in",
    "evt_intubation",
    "evt_extubation",
    "evt_induction",
    "evt_emergence",
    "evt_case_delayed",
    "evt_iv_antibiotics",
    "evt_lma_placed",
    "evt_tee_echo_placed",
    "evt_tourniquet_inflated",
    "evt_transported_pacu_icu",
    "evt_two_anti_emetics",
    # 類別（字串，需 encode）
    "SEX",
    "PRIMARY_ANES_TYPE_NM",
    "PATIENT_CLASS_GROUP",
    "procedure_group",
]
TARGET_COL = "any_complication"
YEAR_COL   = "surgery_year"

CAT_COLS = ["SEX", "PRIMARY_ANES_TYPE_NM", "PATIENT_CLASS_GROUP", "procedure_group"]

# 滑動切割表：(label, old_end_year)
MOVER_YEAR_SPLITS = [
    ("split_1+3", 2018),   # Old=2018      (1yr), New=2019-2021 (3yr)
    ("split_2+2", 2019),   # Old=2018-2019 (2yr), New=2020-2021 (2yr)
    ("split_3+1", 2020),   # Old=2018-2020 (3yr), New=2021      (1yr)
]
BASE_YEAR      = 2018   # 2017 (45 rows) 太小，從 2018 起算
TRAIN_END_YEAR = 2021
TEST_YEAR      = 2022   # 固定未來資料為 Test
DRIFT_YEAR     = 2023   # 次要 Drift check

SAMPLING_STRATEGIES     = ["none", "undersampling", "oversampling", "hybrid"]
SAMPLING_REPORT_ORDER   = ["hybrid", "none", "oversampling", "undersampling"]
METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
COMPACT_SUMMARY_METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision"]
COMPACT_ONLY_METRICS    = ["AUC", "F1", "Recall"]

OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "xgb_mover"


# ── 類別編碼：fit on Old, transform all ──────────────────────────────────────
def _encode_categoricals(
    X_fit: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """OrdinalEncoder：fit on X_fit，transform X_fit + others。未知類別填 -1。"""
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float32,
    )
    cat_present = [c for c in CAT_COLS if c in X_fit.columns]
    if not cat_present:
        return (X_fit,) + others

    X_fit = X_fit.copy()
    X_fit[cat_present] = enc.fit_transform(X_fit[cat_present].astype(str))
    result = [X_fit]
    for df in others:
        df = df.copy()
        df[cat_present] = enc.transform(df[cat_present].astype(str))
        result.append(df)
    return tuple(result)


# ── 資料載入與切割 ────────────────────────────────────────────────────────────
def load_mover(logger) -> pd.DataFrame:
    if not MOVER_LABELED_PATH.exists():
        raise FileNotFoundError(
            f"找不到 MOVER 建模檔: {MOVER_LABELED_PATH}\n"
            "請先執行 scripts/data/build_mover_dataset.py 與 scripts/data/_fix_mover_labeled.py"
        )
    df = pd.read_csv(MOVER_LABELED_PATH)
    df = df[df["temporal_split"].isin(["train", "val", "test", "drift", "other"])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    logger.info(f"MOVER loaded: {len(df):,} rows  pos_rate={df[TARGET_COL].mean():.4f}")
    return df


def get_mover_year_split(
    df: pd.DataFrame,
    logger,
    old_end_year: int,
):
    """
    依 old_end_year 切 Old / New / Test / Drift：
      Old   = BASE_YEAR <= year <= old_end_year
      New   = old_end_year < year <= TRAIN_END_YEAR
      Test  = year == TEST_YEAR   (2022)
      Drift = year == DRIFT_YEAR  (2023，次要)
    """
    year = df[YEAR_COL].astype(int)
    mask_old   = (year >= BASE_YEAR) & (year <= old_end_year)
    mask_new   = (year > old_end_year) & (year <= TRAIN_END_YEAR)
    mask_test  = (year == TEST_YEAR)
    mask_drift = (year == DRIFT_YEAR)

    def _extract(mask):
        sub = df[mask].reset_index(drop=True)
        return sub[FEATURE_COLS].copy(), sub[TARGET_COL].copy(), sub[YEAR_COL].copy()

    X_old,   y_old,   yr_old   = _extract(mask_old)
    X_new,   y_new,   yr_new   = _extract(mask_new)
    X_test,  y_test,  _        = _extract(mask_test)
    X_drift, y_drift, _        = _extract(mask_drift)

    logger.info(
        f"  Old={len(X_old):,}({y_old.mean()*100:.2f}%+)  "
        f"New={len(X_new):,}({y_new.mean()*100:.2f}%+)  "
        f"Test(2022)={len(X_test):,}({y_test.mean()*100:.2f}%+)  "
        f"Drift(2023)={len(X_drift):,}({y_drift.mean()*100:.2f}%+)"
    )
    return X_old, y_old, yr_old, X_new, y_new, yr_new, X_test, y_test, X_drift, y_drift


# ── Validation split ─────────────────────────────────────────────────────────
def _split_fit_val(X, y, year_arr, *, val_ratio=0.2, random_state=42):
    years = np.asarray(year_arr)
    y_arr = np.asarray(y)
    fit_idx, val_idx = [], []

    for yr in sorted(np.unique(years)):
        idx = np.where(years == yr)[0]
        n = len(idx)
        if n <= 1:
            fit_idx.extend(idx.tolist())
            continue
        n_val = max(1, min(int(round(n * val_ratio)), n - 1))
        stratify = None
        y_sub = y_arr[idx]
        if len(np.unique(y_sub)) >= 2 and n_val >= len(np.unique(y_sub)):
            stratify = y_sub
        try:
            i_fit, i_val = train_test_split(
                idx, test_size=n_val, random_state=random_state + int(yr), stratify=stratify
            )
        except ValueError:
            i_fit, i_val = train_test_split(
                idx, test_size=n_val, random_state=random_state + int(yr)
            )
        fit_idx.extend(i_fit.tolist())
        val_idx.extend(i_val.tolist())

    if not val_idx:
        return train_test_split(
            np.arange(len(X)), test_size=val_ratio, random_state=random_state,
            stratify=(y_arr if len(np.unique(y_arr)) >= 2 else None)
        )
    fi = np.array(sorted(fit_idx))
    vi = np.array(sorted(val_idx))
    return (
        X.iloc[fi].reset_index(drop=True), y_arr[fi],
        X.iloc[vi].reset_index(drop=True), y_arr[vi],
    )


def _scale_pos_weight(y: np.ndarray) -> float:
    unique, counts = np.unique(np.asarray(y), return_counts=True)
    if len(unique) != 2:
        return 1.0
    neg = counts[unique == 0][0] if 0 in unique else 1
    pos = counts[unique == 1][0] if 1 in unique else 1
    return float(neg / max(pos, 1))


def _select_threshold(y_val: np.ndarray, proba_val: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_val, (proba_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


# ── 訓練 + 評估 ───────────────────────────────────────────────────────────────
def _train_eval(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sampler: ImbalanceSampler,
    strategy: str,
    tag: str,
    logger,
    year_train: pd.Series | None = None,
    X_val_ext: pd.DataFrame | None = None,
    y_val_ext: np.ndarray | None = None,
    split_label: str = "",
    use_tuning: bool = False,
    n_tune_iter: int = 48,
    X_drift: pd.DataFrame | None = None,
    y_drift: np.ndarray | None = None,
) -> dict:
    y_tr = np.asarray(y_train)

    if X_val_ext is not None and y_val_ext is not None:
        X_fit_raw, y_fit = X_train, y_tr
        X_val_raw, y_val = X_val_ext, np.asarray(y_val_ext)
    else:
        yr = year_train if year_train is not None else pd.Series(np.zeros(len(X_train)))
        X_fit_raw, y_fit, X_val_raw, y_val = _split_fit_val(X_train, y_tr, yr)

    # 類別編碼（fit on X_fit only）
    has_drift_data = X_drift is not None and len(X_drift) > 0
    if has_drift_data:
        encoded = _encode_categoricals(X_fit_raw, X_val_raw, X_test, X_drift)
        X_fit_enc, X_val_enc, X_test_enc, X_drift_enc = encoded
    else:
        X_fit_enc, X_val_enc, X_test_enc = _encode_categoricals(X_fit_raw, X_val_raw, X_test)
        X_drift_enc = None

    # imblearn 不支援 NaN → 填 -1（XGBoost 本身可處理，但 imblearn 的 resamplers 不行）
    X_fit_enc = X_fit_enc.fillna(-1)
    X_val_enc = X_val_enc.fillna(-1)
    X_test_enc = X_test_enc.fillna(-1)
    if X_drift_enc is not None:
        X_drift_enc = X_drift_enc.fillna(-1)

    # 採樣
    X_r, y_r = sampler.apply_sampling(X_fit_enc, y_fit, strategy=strategy)

    tune_seed = 42 + (zlib.adler32(f"{split_label}|{tag}|{strategy}".encode()) % 100000)

    if use_tuning:
        from src.utils import get_config_loader
        cfg = get_config_loader()
        base = dict(cfg.get("model_config", "xgboost.base_params", {}))
        spw = _scale_pos_weight(y_r)
        best, auc_s = search_xgb_on_val(
            X_r, y_r, X_val_enc, y_val, base, spw,
            n_iter=int(n_tune_iter), seed=int(tune_seed),
        )
        if best:
            model = XGBoostWrapper(
                name=f"{tag}_{strategy}", use_imbalance=False,
                **{**base, **best, "scale_pos_weight": spw},
            )
        else:
            model = XGBoostWrapper(name=f"{tag}_{strategy}")
            best, auc_s = {}, float("nan")
        model.fit(X_r, y_r)
        tune_ex = tuning_meta(best, auc_s)
    else:
        model = XGBoostWrapper(name=f"{tag}_{strategy}")
        model.fit(X_r, y_r)
        tune_ex = {}

    proba_val = model.predict_proba(X_val_enc)
    threshold, val_f1 = _select_threshold(y_val, proba_val)

    y_t = np.asarray(y_test)
    metrics = compute_metrics(y_t, model.predict_proba(X_test_enc), threshold=threshold)
    metrics.update(tune_ex)

    # Drift 2023 評估（次要）
    if X_drift_enc is not None and y_drift is not None and len(y_drift) > 0:
        y_dr = np.asarray(y_drift)
        drift_m = compute_metrics(y_dr, model.predict_proba(X_drift_enc), threshold=threshold)
        for k, v in drift_m.items():
            metrics[f"drift_{k}"] = v

    logger.info(
        f"    {tag:12s} {strategy:12s}  thr={threshold:.3f}  valF1={val_f1:.4f} | "
        f"AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}"
        + (f"  drift_AUC={metrics.get('drift_AUC', float('nan')):.4f}"
           if X_drift_enc is not None else "")
    )
    return metrics


# ── 一個 split 的完整執行 ────────────────────────────────────────────────────
def run_split(
    label: str,
    X_old: pd.DataFrame, y_old: pd.Series, yr_old: pd.Series,
    X_new: pd.DataFrame, y_new: pd.Series, yr_new: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    X_drift: pd.DataFrame, y_drift: pd.Series,
    logger,
    include_retrain: bool = True,
    use_tuning: bool = False,
    n_tune_iter: int = 48,
) -> list[dict]:
    sampler = ImbalanceSampler()
    rows: list[dict] = []
    y_dr = np.asarray(y_drift) if y_drift is not None else None

    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(
            X_old, np.asarray(y_old), X_test, np.asarray(y_test),
            sampler, strat, "Old", logger,
            year_train=yr_old, split_label=label,
            use_tuning=use_tuning, n_tune_iter=n_tune_iter,
            X_drift=X_drift, y_drift=y_dr,
        )
        rows.append({"split": label, "method": "Old", "sampling": strat, **m})

    for strat in SAMPLING_STRATEGIES:
        m = _train_eval(
            X_new, np.asarray(y_new), X_test, np.asarray(y_test),
            sampler, strat, "New", logger,
            year_train=yr_new, split_label=label,
            use_tuning=use_tuning, n_tune_iter=n_tune_iter,
            X_drift=X_drift, y_drift=y_dr,
        )
        rows.append({"split": label, "method": "New", "sampling": strat, **m})

    if include_retrain:
        # Retrain = Old + New，各自切 val 再合併
        X_o_fit, y_o_fit, X_o_val, y_o_val = _split_fit_val(X_old, np.asarray(y_old), yr_old)
        X_n_fit, y_n_fit, X_n_val, y_n_val = _split_fit_val(X_new, np.asarray(y_new), yr_new)
        X_re_fit = pd.concat([X_o_fit, X_n_fit], ignore_index=True)
        y_re_fit = np.concatenate([y_o_fit, y_n_fit])
        X_re_val = pd.concat([X_o_val, X_n_val], ignore_index=True)
        y_re_val = np.concatenate([y_o_val, y_n_val])
        logger.info(f"  Retrain fit={len(X_re_fit):,}  val={len(X_re_val):,}")
        for strat in SAMPLING_STRATEGIES:
            m = _train_eval(
                X_re_fit, y_re_fit, X_test, np.asarray(y_test),
                sampler, strat, "Retrain", logger,
                X_val_ext=X_re_val, y_val_ext=y_re_val,
                split_label=label, use_tuning=use_tuning, n_tune_iter=n_tune_iter,
                X_drift=X_drift, y_drift=y_dr,
            )
            rows.append({"split": label, "method": "Retrain", "sampling": strat, **m})

    return rows


# ── 輸出格式化 ───────────────────────────────────────────────────────────────
def _yr_label(label: str) -> str:
    """'split_1+2' → '1yr'（Old 年數）"""
    try:
        old_yrs = label.split("_")[1].split("+")[0]
        return f"{old_yrs}yr"
    except Exception:
        return label


def format_tables(df_raw: pd.DataFrame, logger, output_dir: Path) -> None:
    df = df_raw.copy()
    split_labels = [s for s, _ in MOVER_YEAR_SPLITS]

    for metric in METRICS:
        if metric not in df.columns:
            continue
        for method, suffix in [("Old", "old"), ("Retrain", "retrain"), ("New", "new")]:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            pv = (
                sub.pivot_table(index="split", columns="sampling", values=metric, aggfunc="mean")
                .reindex(index=split_labels, columns=SAMPLING_STRATEGIES)
            )
            pv.index = [_yr_label(s) for s in pv.index]
            pv["avg"] = pv.mean(axis=1)
            pv.loc["avg"] = pv.mean()
            pv.index.name = f"{suffix}_years"
            out = output_dir / f"mover_xgb_table_{metric}_{suffix}.csv"
            pv.to_csv(out, float_format="%.4f")
            logger.info(f"  Saved -> {out.name}")


def export_compact(df_raw: pd.DataFrame, logger, output_dir: Path) -> None:
    rows = []
    for m in COMPACT_SUMMARY_METRICS:
        if m not in df_raw.columns:
            continue
        pv = (
            df_raw.groupby(["method", "sampling"])[m]
            .mean()
            .unstack("sampling")
            .reindex(index=["Old", "New", "Retrain"], columns=SAMPLING_REPORT_ORDER)
        ).round(4).reset_index()
        pv.insert(0, "metric", m)
        rows.append(pv)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            output_dir / "mover_xgb_compact_summary.csv", index=False, float_format="%.4f"
        )
    for m in COMPACT_ONLY_METRICS:
        if m not in df_raw.columns:
            continue
        pv = (
            df_raw.groupby(["method", "sampling"])[m]
            .mean()
            .unstack("sampling")
            .reindex(index=["Old", "New", "Retrain"], columns=SAMPLING_REPORT_ORDER)
        ).round(4)
        pv.to_csv(output_dir / f"mover_xgb_compact_{m}_only.csv", float_format="%.4f")
        logger.info(f"  Saved -> mover_xgb_compact_{m}_only.csv")

    if "tune_best_params" in df_raw.columns or "tune_val_auc" in df_raw.columns:
        try:
            export_tuning_log(df_raw, output_dir / "mover_xgb_tuning_log.csv")
        except Exception as e:
            logger.warning(f"export_tuning_log: {e}")


# ── 主流程 ───────────────────────────────────────────────────────────────────
def _run(tuning_mode: str, n_tune_iter: int, results_subdir: str = "") -> None:
    logger = get_logger("MOVER_YearSplits_XGB", console=True, file=True)
    set_seed(42)

    results_subdir = results_subdir.strip()
    if results_subdir:
        out_dir = OUTPUT_DIR / results_subdir
    else:
        out_dir = OUTPUT_DIR / "tuned" if tuning_mode == "tuned" else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    use_tuning = tuning_mode == "tuned"
    logger.info(f"Output: {out_dir}")
    logger.info(f"Tuning: {tuning_mode}  n_iter={n_tune_iter}")

    df_all = load_mover(logger)
    all_rows: list[dict] = []

    for split_idx, (label, old_end_year) in enumerate(MOVER_YEAR_SPLITS):
        include_retrain = (split_idx == 0)
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Split: {label}  "
            f"Old={BASE_YEAR}–{old_end_year}  "
            f"New={old_end_year+1}–{TRAIN_END_YEAR}  "
            f"Test(fixed)={TEST_YEAR}  "
            f"Drift(secondary)={DRIFT_YEAR}"
        )
        logger.info("=" * 60)
        try:
            X_old, y_old, yr_old, X_new, y_new, yr_new, X_test, y_test, X_drift, y_drift = \
                get_mover_year_split(df_all, logger, old_end_year)

            rows = run_split(
                label,
                X_old, y_old, yr_old,
                X_new, y_new, yr_new,
                X_test, y_test,
                X_drift, y_drift,
                logger,
                include_retrain=include_retrain,
                use_tuning=use_tuning,
                n_tune_iter=int(n_tune_iter),
            )
            all_rows.extend(rows)
        except Exception as e:
            import traceback
            logger.error(f"[ERROR] {label}: {e}\n{traceback.format_exc()}")

    if not all_rows:
        logger.error("No results produced.")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = out_dir / "mover_year_splits_xgb_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\nRaw results -> {raw_path}  ({len(df_raw)} rows)")

    format_tables(df_raw, logger, out_dir)
    export_compact(df_raw, logger, out_dir)

    logger.info("\n=== 完成 ===")
    summary = (
        df_raw.groupby(["method", "sampling"])["AUC"]
        .mean()
        .unstack("sampling")
        .reindex(["Old", "New", "Retrain"])
    )
    logger.info("\nAUC 摘要（method × sampling，跨 split 平均）:\n" + summary.to_string())


def main() -> None:
    p = argparse.ArgumentParser(description="MOVER XGBoost year-split baseline")
    p.add_argument("--tuning", default="default", choices=["default", "tuned", "both"])
    p.add_argument("--tune-n-iter", type=int, default=48)
    p.add_argument("--results-subdir", type=str, default="")
    args = p.parse_args()

    if args.tuning == "both":
        _run("default", args.tune_n_iter, args.results_subdir)
        _run("tuned",   args.tune_n_iter, args.results_subdir)
    else:
        _run(args.tuning, args.tune_n_iter, args.results_subdir)


if __name__ == "__main__":
    main()
