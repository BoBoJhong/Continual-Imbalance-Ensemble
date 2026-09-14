"""
Phase 4 – Concept Drift Detection（Bankruptcy，US 1999–2018）
=============================================================
資料集：美國企業破產預測（Kaggle american_bankruptcy_dataset.csv）
時間範圍：1999–2018；固定測試集 2015–2018

核心問題
--------
Phase 1–3 使用人工固定切割點（Old=1999–2011，New=2012–2014）。
Phase 4 改由三種漂移偵測器自動找出 drift 發生年份，
並以此更新模型與集成，比較效能是否優於固定切割。

串流模擬邏輯
-----------
1. Burn-in：用 1999–BURN_IN_END 訓練初始模型
2. 逐年前進 (BURN_IN_END+1 → 2014)：
     a. 用當前模型預測該年資料 → 計算 1 - AUC 作為誤差信號
     b. 餵進偵測器；若觸發 → 記錄 drift_year，結束串流
3. 以 drift_year 為 Old/New 邊界：
     - Single model：用 [drift_year, 2014] 重訓（類似 Phase 1 "New" 策略）
     - Ensemble：Old pool 用 [1999, drift_year-1]，New pool 用 [drift_year, 2014]
4. 全部方法在同一測試集 (2015–2018) 上評估

比較基準（對齊 Phase 1 固定切割）
---------------------------------
  Fixed_Old   : Old data only (1999–FIXED_OLD_END)
  Fixed_New   : New data only (FIXED_NEW_START–2014)
  Fixed_Retrain: 全量 1999–2014
  Fixed_Ensemble_all6 : Phase 2 固定集成（Old pool + New pool，固定邊界）

輸出
----
  results/phase4_drift/bk_drift_detection_points.csv  ← 各偵測器觸發年份
  results/phase4_drift/bk_drift_vs_fixed_comparison.csv ← 完整對照表

用法
----
  python experiments/phase4_drift/bankruptcy_drift_stream.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataPreprocessor, ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_logger, set_seed

from experiments._shared.common_bankruptcy import US_CSV
from experiments.phase4_drift._detectors import make_detectors

# ── 常數 ────────────────────────────────────────────────────────────────────
BURN_IN_END      = 2001   # 初始模型訓練至此年（含）
TRAIN_END_YEAR   = 2014
TEST_START_YEAR  = 2015
TEST_END_YEAR    = 2018

FIXED_OLD_END    = 2011   # Phase 1 固定 Old 邊界
FIXED_NEW_START  = 2012   # Phase 1 固定 New 起點

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
POOL_SAMPLING       = ["undersampling", "oversampling", "hybrid"]   # 集成用（3 個模型）

# 集成組合（對齊 Phase 2）
ENSEMBLE_COMBOS = {
    "old3":     ["old_undersampling", "old_oversampling", "old_hybrid"],
    "new3":     ["new_undersampling", "new_oversampling", "new_hybrid"],
    "pair_hy":  ["old_hybrid", "new_hybrid"],
    "all6":     ["old_undersampling", "old_oversampling", "old_hybrid",
                 "new_undersampling", "new_oversampling", "new_hybrid"],
}

OUTPUT_DIR = project_root / "results" / "phase4_drift"
METRICS    = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]


# ── 資料載入 ─────────────────────────────────────────────────────────────────
def load_bankruptcy_with_year(logger, *, keep_company: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    載入 US 破產資料，保留 fyear 欄以供年份切割用。
    回傳 (X_with_fyear, y)。
    """
    if not US_CSV.exists():
        raise FileNotFoundError(
            f"找不到 US 破產資料: {US_CSV}\n"
            "請下載 american_bankruptcy_dataset.csv 放到 data/raw/bankruptcy/"
        )
    df = pd.read_csv(US_CSV)
    if df["status_label"].isna().any() or not set(df["status_label"].unique()) <= {"alive", "failed"}:
        raise ValueError("Unexpected bankruptcy labels; expected exactly alive/failed values.")
    y = (df["status_label"] == "failed").astype(int)
    excluded = ["status_label", "Division"] + ([] if keep_company else ["company_name"])
    drop_cols = [c for c in excluded if c in df.columns]
    X = df.drop(columns=drop_cols)
    logger.info(
        f"US Bankruptcy loaded: {len(X):,} rows  "
        f"fyear=[{X['fyear'].min()},{X['fyear'].max()}]  "
        f"bankruptcy_rate={y.mean()*100:.2f}%"
    )
    return X, y


# ── 前處理工具 ───────────────────────────────────────────────────────────────
def _preprocess(X_train_raw: pd.DataFrame, *others: pd.DataFrame):
    """
    Fit imputation statistics and StandardScaler on X_train_raw only.

    Validation/test frames are only transformed with training-derived
    statistics. Returns (X_train_scaled, *others_scaled, scaler).
    """
    fill_values = X_train_raw.mean(numeric_only=True)
    Xtr = X_train_raw.fillna(fill_values)
    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns)
    scaler.training_fill_values_ = fill_values

    result = [Xtr_s]
    for df in others:
        df_c = df.fillna(fill_values)
        result.append(pd.DataFrame(scaler.transform(df_c), columns=df_c.columns))
    result.append(scaler)
    return tuple(result)


# ── 閾值搜尋 ─────────────────────────────────────────────────────────────────
def _select_threshold(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_val, (proba_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ── 單一模型訓練＋評估 ────────────────────────────────────────────────────────
def train_eval_single(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    sampling: str,
    tag: str,
    logger,
) -> dict:
    """
    訓練一個 XGBoost 模型（指定採樣策略），在測試集評估後回傳指標 dict。
    內部用訓練集最後 20% 做 validation 搜尋閾值。
    """
    # 訓練 / validation 分割（最後 20%，不 shuffle 保持時序）
    n_val = max(1, int(len(X_train_raw) * 0.2))
    X_fit_raw, X_val_raw = X_train_raw.iloc[:-n_val], X_train_raw.iloc[-n_val:]
    y_fit, y_val = y_train[:-n_val],     y_train[-n_val:]
    X_fit, X_val, X_te_s, _ = _preprocess(X_fit_raw, X_val_raw, X_test_raw)

    sampler = ImbalanceSampler()
    X_r, y_r = sampler.apply_sampling(X_fit, y_fit, strategy=sampling)

    model = XGBoostWrapper(name=f"{tag}_{sampling}", use_imbalance=False)
    model.fit(X_r, y_r)

    threshold = _select_threshold(y_val, model.predict_proba(X_val))
    metrics   = compute_metrics(y_test, model.predict_proba(X_te_s), threshold=threshold)

    logger.info(
        f"  [{tag}|{sampling:12s}] thr={threshold:.2f}  "
        f"AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  "
        f"Recall={metrics['Recall']:.4f}"
    )
    return metrics


# ── 集成訓練＋評估 ────────────────────────────────────────────────────────────
def train_eval_ensemble(
    X_old_raw: pd.DataFrame,
    y_old: np.ndarray,
    X_new_raw: pd.DataFrame,
    y_new: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    tag: str,
    logger,
) -> dict[str, dict]:
    """
    訓練 Old pool（3 模型）+ New pool（3 模型），回傳各集成組合的指標 dict。
    Scaler 以 Old 資料 fit（對齊 Phase 2 設計）。
    閾值以 Old+New 混合 val 搜尋（各自取最後 20% 合併）。
    """
    sampler = ImbalanceSampler()
    pool: dict[str, XGBoostWrapper] = {}

    n_old_val = max(1, int(len(X_old_raw) * 0.2))
    n_new_val = max(1, int(len(X_new_raw) * 0.2))
    X_old_fit_raw, X_old_val_raw = X_old_raw.iloc[:-n_old_val], X_old_raw.iloc[-n_old_val:]
    X_new_fit_raw, X_new_val_raw = X_new_raw.iloc[:-n_new_val], X_new_raw.iloc[-n_new_val:]
    X_old_fit, X_new_fit, X_old_val, X_new_val, X_te_s, _ = _preprocess(
        X_old_fit_raw,
        X_new_fit_raw,
        X_old_val_raw,
        X_new_val_raw,
        X_test_raw,
    )
    X_val = pd.concat([X_old_val, X_new_val], ignore_index=True)
    y_val = np.concatenate([y_old[-n_old_val:], y_new[-n_new_val:]])

    for s in POOL_SAMPLING:
        # Old model
        X_r, y_r = sampler.apply_sampling(X_old_fit, y_old[:-n_old_val], strategy=s)
        m = XGBoostWrapper(name=f"old_{s}_{tag}", use_imbalance=False)
        m.fit(X_r, y_r)
        pool[f"old_{s}"] = m

        # New model
        X_r, y_r = sampler.apply_sampling(X_new_fit, y_new[:-n_new_val], strategy=s)
        m = XGBoostWrapper(name=f"new_{s}_{tag}", use_imbalance=False)
        m.fit(X_r, y_r)
        pool[f"new_{s}"] = m

    results: dict[str, dict] = {}
    for combo_name, keys in ENSEMBLE_COMBOS.items():
        # 用 val 集搜尋集成閾值
        val_probas = np.mean([pool[k].predict_proba(X_val) for k in keys], axis=0)
        threshold  = _select_threshold(y_val, val_probas)

        te_probas = np.mean([pool[k].predict_proba(X_te_s) for k in keys], axis=0)
        metrics   = compute_metrics(y_test, te_probas, threshold=threshold)
        results[combo_name] = metrics
        logger.info(
            f"  [{tag}|ens_{combo_name:8s}] thr={threshold:.2f}  "
            f"AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  "
            f"Recall={metrics['Recall']:.4f}"
        )

    return results


# ── 串流模擬：找 drift year ──────────────────────────────────────────────────
def _build_init_model(
    X_all: pd.DataFrame,
    y_all: pd.Series,
) -> tuple:
    """
    用 burn-in 資料訓練初始模型，回傳 (model, scaler, threshold, burn_in_err_rate)。
    此函式在三個偵測器間共享，避免重複訓練。
    """
    mask_init = X_all["fyear"] <= BURN_IN_END
    X_init = X_all[mask_init].drop(columns=["fyear"])
    y_init = np.asarray(y_all[mask_init])

    scaler_init = StandardScaler()
    fill_values = X_init.mean(numeric_only=True)
    X_init_clean = X_init.fillna(fill_values)
    X_init_s = pd.DataFrame(
        scaler_init.fit_transform(X_init_clean), columns=X_init_clean.columns
    )
    scaler_init.training_fill_values_ = fill_values

    sampler = ImbalanceSampler()
    X_r, y_r = sampler.apply_sampling(X_init_s, y_init, strategy="hybrid")
    init_model = XGBoostWrapper(name="drift_init", use_imbalance=False)
    init_model.fit(X_r, y_r)

    # 在 burn-in 資料上找最佳 F1 閾值
    proba_init = init_model.predict_proba(X_init_s)
    threshold_init = _select_threshold(y_init, proba_init)

    # burn-in 誤差率（作為 PHT 固定基準）
    preds_init = (proba_init >= threshold_init).astype(int)
    burn_in_err = float(np.mean(preds_init != y_init))

    return init_model, scaler_init, threshold_init, burn_in_err, len(y_init), float(y_init.mean())


def find_drift_year(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    detector,
    detector_name: str,
    logger,
    init_model=None,
    scaler_init=None,
    threshold_init: float = 0.5,
    burn_in_err: float = 0.05,
) -> int | None:
    """
    以初始 burn-in 模型（已訓練好），逐年前進，將每一筆預測的二元誤差（0/1）
    **逐筆**餵給偵測器（instance-level stream）。

    正確粒度：每年約 5,000 筆 → Hoeffding bound ~0.02，偵測器才能有效運作。
    （餵年級 AUC 只有 13 個點，Hoeffding bound ≈ 1.0，永遠不會觸發。）

    PHT 若設有 reference_mean=None，在此動態注入 burn-in 固定誤差率。
    """
    # 動態注入 PHT 基準
    if hasattr(detector, "reference_mean") and detector.reference_mean is None:
        detector.reference_mean = burn_in_err
        logger.info(
            f"  [{detector_name}] PHT fixed reference_mean = {burn_in_err:.4f} "
            f"（burn-in error rate）"
        )

    stream_years = sorted(
        int(y) for y in X_all["fyear"].unique()
        if BURN_IN_END < y <= TRAIN_END_YEAR
    )

    # --- 逐年：逐筆餵入 ---
    for year in stream_years:
        mask_yr = X_all["fyear"] == year
        X_yr = X_all[mask_yr].drop(columns=["fyear"])
        y_yr = np.asarray(y_all[mask_yr])

        if len(y_yr) == 0:
            continue

        X_yr_clean = X_yr.fillna(scaler_init.training_fill_values_)
        # 保持 burn-in scaler，才能偵測到分布偏移
        X_yr_s = pd.DataFrame(
            scaler_init.transform(X_yr_clean), columns=X_yr_clean.columns
        )

        proba_yr  = init_model.predict_proba(X_yr_s)
        preds_yr  = (proba_yr >= threshold_init).astype(int)
        inst_errors = (preds_yr != y_yr).astype(float)   # 0/1 per instance

        yr_err  = float(inst_errors.mean())
        try:
            yr_auc = float(roc_auc_score(y_yr, proba_yr)) if y_yr.sum() > 0 else 0.5
        except Exception:
            yr_auc = 0.5

        # 逐筆餵入（遇到觸發立即停止）
        drift_in_year = False
        for err in inst_errors:
            if detector.update(err):
                drift_in_year = True
                break

        logger.info(
            f"  [{detector_name}] year={year}  "
            f"n={len(y_yr):,}  err={yr_err:.4f}  AUC={yr_auc:.4f}"
            + ("  ← DRIFT DETECTED" if drift_in_year else "")
        )

        if drift_in_year:
            logger.info(f"  [{detector_name}] Drift confirmed at year {year}!")
            return year

    logger.info(f"  [{detector_name}] No drift detected in instance stream.")
    return None


# ── 固定基準（Phase 1 對齊） ─────────────────────────────────────────────────
def run_fixed_baselines(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    logger,
) -> list[dict]:
    rows = []
    logger.info("\n" + "="*60)
    logger.info("固定基準（Phase 1 / Phase 2 對齊）")
    logger.info("="*60)

    # -- 單一模型：Old only --
    mask_old = (X_all["fyear"] >= 1999) & (X_all["fyear"] <= FIXED_OLD_END)
    X_old_r = X_all[mask_old].drop(columns=["fyear"])
    y_old   = np.asarray(y_all[mask_old])
    for s in SAMPLING_STRATEGIES:
        m = train_eval_single(X_old_r, y_old, X_test_raw, y_test, s, f"Fixed_Old({FIXED_OLD_END})", logger)
        rows.append({"method": "Fixed_Old", "detector": "none", "drift_year": "-",
                     "model_type": "single", "sampling/combo": s, **m})

    # -- 單一模型：New only --
    mask_new = (X_all["fyear"] >= FIXED_NEW_START) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_new_r = X_all[mask_new].drop(columns=["fyear"])
    y_new   = np.asarray(y_all[mask_new])
    for s in SAMPLING_STRATEGIES:
        m = train_eval_single(X_new_r, y_new, X_test_raw, y_test, s, f"Fixed_New({FIXED_NEW_START})", logger)
        rows.append({"method": "Fixed_New", "detector": "none", "drift_year": "-",
                     "model_type": "single", "sampling/combo": s, **m})

    # -- 單一模型：Retrain --
    mask_all = (X_all["fyear"] >= 1999) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_all_r = X_all[mask_all].drop(columns=["fyear"])
    y_all_r = np.asarray(y_all[mask_all])
    for s in SAMPLING_STRATEGIES:
        m = train_eval_single(X_all_r, y_all_r, X_test_raw, y_test, s, "Fixed_Retrain", logger)
        rows.append({"method": "Fixed_Retrain", "detector": "none", "drift_year": "-",
                     "model_type": "single", "sampling/combo": s, **m})

    # -- 集成：固定 Old/New（Phase 2 切割） --
    logger.info(f"\n  固定集成  Old=1999-{FIXED_OLD_END}  New={FIXED_NEW_START}-{TRAIN_END_YEAR}")
    ens_results = train_eval_ensemble(
        X_old_r, y_old, X_new_r, y_new, X_test_raw, y_test,
        tag=f"FixedEns({FIXED_OLD_END}/{FIXED_NEW_START})", logger=logger,
    )
    for combo_name, m in ens_results.items():
        rows.append({"method": "Fixed_Ensemble", "detector": "none", "drift_year": "-",
                     "model_type": "ensemble", "sampling/combo": combo_name, **m})

    return rows


# ── 漂移偵測實驗 ─────────────────────────────────────────────────────────────
def run_drift_experiment(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    detector_name: str,
    detector,
    logger,
    init_model=None,
    scaler_init=None,
    threshold_init: float = 0.5,
    burn_in_err: float = 0.05,
) -> list[dict]:
    logger.info(f"\n{'='*60}")
    logger.info(f"偵測器：{detector_name}")
    logger.info("="*60)

    rows: list[dict] = []

    # 1. 串流找 drift_year（逐筆實例級）
    drift_year = find_drift_year(
        X_all, y_all, detector, detector_name, logger,
        init_model=init_model,
        scaler_init=scaler_init,
        threshold_init=threshold_init,
        burn_in_err=burn_in_err,
    )
    fallback = drift_year is None
    if fallback:
        # 未偵測到漂移：退回全量 Retrain
        drift_year = BURN_IN_END + 1
        logger.info(f"  未偵測到漂移，退回 Retrain（drift_year 設為 {drift_year}）")

    actual_drift_display = drift_year if not fallback else "none"

    # 2. 切割 Old / New by drift_year
    mask_old = (X_all["fyear"] >= 1999) & (X_all["fyear"] < drift_year)
    mask_new = (X_all["fyear"] >= drift_year) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_old_r = X_all[mask_old].drop(columns=["fyear"])
    y_old_np = np.asarray(y_all[mask_old])
    X_new_r = X_all[mask_new].drop(columns=["fyear"])
    y_new_np = np.asarray(y_all[mask_new])
    logger.info(
        f"  Old: {len(X_old_r):,} rows ({y_old_np.mean()*100:.1f}%+)  "
        f"New: {len(X_new_r):,} rows ({y_new_np.mean()*100:.1f}%+)"
    )

    # 3. 單一模型（New window，各採樣策略）
    if len(X_new_r) > 0:
        for s in SAMPLING_STRATEGIES:
            m = train_eval_single(
                X_new_r, y_new_np, X_test_raw, y_test, s,
                f"{detector_name}_New", logger,
            )
            rows.append({
                "method": f"{detector_name}_Single_New",
                "detector": detector_name,
                "drift_year": actual_drift_display,
                "model_type": "single",
                "sampling/combo": s,
                **m,
            })

    # 4. 集成（drift-triggered Old pool + New pool）
    if len(X_old_r) > 0 and len(X_new_r) > 0:
        logger.info(f"\n  集成（drift-triggered）Old=1999-{drift_year-1}  New={drift_year}-{TRAIN_END_YEAR}")
        ens_results = train_eval_ensemble(
            X_old_r, y_old_np, X_new_r, y_new_np, X_test_raw, y_test,
            tag=f"{detector_name}_Drift", logger=logger,
        )
        for combo_name, m in ens_results.items():
            rows.append({
                "method": f"{detector_name}_Ensemble",
                "detector": detector_name,
                "drift_year": actual_drift_display,
                "model_type": "ensemble",
                "sampling/combo": combo_name,
                **m,
            })

    return rows


# ── 輸出格式化 ───────────────────────────────────────────────────────────────
def _save_results(all_rows: list[dict], detection_log: list[dict], logger) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 完整對照表
    df = pd.DataFrame(all_rows)
    cols = ["method", "detector", "drift_year", "model_type", "sampling/combo"] + METRICS
    df = df[[c for c in cols if c in df.columns]]
    path_cmp = OUTPUT_DIR / "bk_drift_vs_fixed_comparison.csv"
    df.to_csv(path_cmp, index=False, float_format="%.4f")
    logger.info(f"\n完整對照表 → {path_cmp}  ({len(df)} 列)")

    # 漂移偵測點
    df_pts = pd.DataFrame(detection_log)
    path_pts = OUTPUT_DIR / "bk_drift_detection_points.csv"
    df_pts.to_csv(path_pts, index=False)
    logger.info(f"漂移偵測點 → {path_pts}")

    # 摘要（各 method 的最佳單一 F1）
    logger.info("\n=== Test oracle 摘要：不可當作 Validation-selected 方法 ===")
    pivot_rows = []
    for (method, detector, drift_yr), grp in df.groupby(["method", "detector", "drift_year"]):
        best = grp.sort_values("F1", ascending=False).iloc[0]
        pivot_rows.append({
            "method": method, "detector": detector, "drift_year": drift_yr,
            "best_sampling/combo": best["sampling/combo"],
            "AUC": best["AUC"], "F1": best["F1"],
            "Recall": best["Recall"], "Precision": best["Precision"],
            "selection_source": "test_oracle",
            "evidence_status": "exploratory_not_confirmatory",
        })
    df_summary = pd.DataFrame(pivot_rows).sort_values("F1", ascending=False)
    path_sum = OUTPUT_DIR / "bk_drift_test_oracle_summary.csv"
    df_summary.to_csv(path_sum, index=False, float_format="%.4f")
    logger.info("\n" + df_summary.to_string(index=False))
    logger.info(f"\n摘要 → {path_sum}")


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main() -> None:
    logger = get_logger("Phase4_Drift_BK", console=True, file=True)
    set_seed(42)

    # 載入資料
    X_all, y_all = load_bankruptcy_with_year(logger)

    # 固定測試集
    mask_test = (X_all["fyear"] >= TEST_START_YEAR) & (X_all["fyear"] <= TEST_END_YEAR)
    X_test_raw = X_all[mask_test].drop(columns=["fyear"]).reset_index(drop=True)
    y_test     = np.asarray(y_all[mask_test])
    logger.info(
        f"Test set: {len(X_test_raw):,} rows  "
        f"pos={y_test.mean()*100:.1f}%  ({TEST_START_YEAR}-{TEST_END_YEAR})"
    )

    # 訓練流（1999–2014）
    mask_train = (X_all["fyear"] >= 1999) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_stream = X_all[mask_train].reset_index(drop=True)
    y_stream = y_all[mask_train].reset_index(drop=True)

    all_rows: list[dict] = []
    detection_log: list[dict] = []

    # 1. 固定基準
    fixed_rows = run_fixed_baselines(X_stream, y_stream, X_test_raw, y_test, logger)
    all_rows.extend(fixed_rows)

    # 2. 預先訓練共用 burn-in 模型（三個偵測器共用，節省時間）
    logger.info("\n訓練共用 burn-in 初始模型...")
    init_model, scaler_init, threshold_init, burn_in_err, n_init, pos_init = \
        _build_init_model(X_stream, y_stream)
    logger.info(
        f"  burn-in 1999-{BURN_IN_END}: {n_init:,} rows  "
        f"pos={pos_init*100:.1f}%  thr={threshold_init:.2f}  err={burn_in_err:.4f}"
    )

    # 3. 各偵測器實驗
    detectors = make_detectors(pht_reference_mean=burn_in_err)
    for det_name, detector in detectors.items():
        drift_rows = run_drift_experiment(
            X_stream, y_stream, X_test_raw, y_test,
            det_name, detector, logger,
            init_model=init_model,
            scaler_init=scaler_init,
            threshold_init=threshold_init,
            burn_in_err=burn_in_err,
        )
        all_rows.extend(drift_rows)

        # 記錄偵測點（從結果取）
        drift_yrs = set(
            r["drift_year"] for r in drift_rows if r["drift_year"] != "none"
        )
        detection_log.append({
            "detector": det_name,
            "drift_year": drift_yrs.pop() if drift_yrs else "none",
        })

    # 3. 儲存
    _save_results(all_rows, detection_log, logger)
    logger.info("\n=== Phase 4 Bankruptcy Drift Detection 完成 ===")


if __name__ == "__main__":
    main()
