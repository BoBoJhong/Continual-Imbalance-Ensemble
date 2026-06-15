"""
Phase 4b – Concept Drift Detection with Year-level 1-AUC Signal
================================================================
原始 bankruptcy_drift_stream.py 使用逐筆 binary error（0/1）作為偵測信號，
在高度不平衡資料中，信號被多數類主導，ADWIN/DDM 無法觸發。

本腳本的修正：
1. 信號改為「年級 1-AUC」（每年一個值，直接反映少數類識別能力）
2. PHT 的 burn-in reference_mean 改以 hold-out 驗證集計算（非 in-sample）
3. 僅保留 PHT（Page-Hinkley Test）——適合年級少量序列；
   ADWIN/DDM 設計用於大量 instance-level 串流，年級 13 點統計效力不足。

輸出
----
  results/phase4_drift/bk_drift_auc_signal_detection.csv   ← 各年 1-AUC 軌跡與觸發點
  results/phase4_drift/bk_drift_auc_signal_comparison.csv  ← 與 ROSS / 固定基準對照

用法
----
  python experiments/phase4_drift/bankruptcy_drift_auc_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import ImbalanceSampler
from src.models import XGBoostWrapper
from src.evaluation import compute_metrics
from src.utils import get_logger, set_seed

from experiments._shared.common_bankruptcy import US_CSV
from experiments.phase4_drift._detectors import PageHinkley

# ── 常數 ─────────────────────────────────────────────────────────────────────
BURN_IN_END     = 2004   # 延長至 2004，讓 PHT 有更穩定的 burn-in 基準
TRAIN_END_YEAR  = 2014
TEST_START_YEAR = 2015
TEST_END_YEAR   = 2018

FIXED_OLD_END   = 2011   # 原始人工固定邊界
FIXED_NEW_START = 2012

ROSS_BOUNDARY   = 2009   # ROSS 選出的最佳邊界（來自 bk_validation_ross_selected_boundary.csv）

POOL_SAMPLING   = ["undersampling", "oversampling", "hybrid"]
OUTPUT_DIR      = project_root / "results" / "phase4_drift"
METRICS         = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]


# ── 資料載入 ──────────────────────────────────────────────────────────────────
def load_data(logger):
    if not US_CSV.exists():
        raise FileNotFoundError(f"找不到 US 破產資料: {US_CSV}")
    df = pd.read_csv(US_CSV)
    y = (df["status_label"] == "failed").astype(int)
    drop_cols = [c for c in ["company_name", "status_label", "Division"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    logger.info(
        f"Loaded: {len(X):,} rows  fyear=[{X['fyear'].min()},{X['fyear'].max()}]  "
        f"bankruptcy={y.mean()*100:.2f}%"
    )
    return X, y


# ── 前處理 ────────────────────────────────────────────────────────────────────
def _scale(X_fit_raw: pd.DataFrame, *others: pd.DataFrame):
    """以 X_fit_raw fit scaler，回傳 (X_fit_s, *others_s, scaler)"""
    def _fill(df):
        return df.fillna(df.mean())
    Xf = _fill(X_fit_raw)
    sc = StandardScaler()
    Xf_s = pd.DataFrame(sc.fit_transform(Xf), columns=Xf.columns)
    result = [Xf_s]
    for df in others:
        result.append(pd.DataFrame(sc.transform(_fill(df)), columns=df.columns))
    result.append(sc)
    return tuple(result)


def _best_threshold(y_val, proba_val):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_val, (proba_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ── Burn-in：訓練初始模型，用 hold-out 計算 reference AUC ───────────────────
def build_init_model(X_all: pd.DataFrame, y_all: pd.Series, logger):
    """
    訓練 burn-in 模型，並以 hold-out（最後 20%）計算 reference 1-AUC。
    避免 in-sample 誤差導致 PHT 基準過於樂觀。
    """
    mask = X_all["fyear"] <= BURN_IN_END
    X_raw = X_all[mask].drop(columns=["fyear"]).reset_index(drop=True)
    y_raw = y_all[mask].to_numpy()

    # hold-out 分割（最後 20%，保持時序）
    n_ho = max(1, int(len(X_raw) * 0.2))
    X_tr_raw, X_ho_raw = X_raw.iloc[:-n_ho], X_raw.iloc[-n_ho:]
    y_tr, y_ho = y_raw[:-n_ho], y_raw[-n_ho:]

    X_tr_s, X_ho_s, scaler = _scale(X_tr_raw, X_ho_raw)

    sampler = ImbalanceSampler()
    X_r, y_r = sampler.apply_sampling(X_tr_s, y_tr, strategy="hybrid")
    model = XGBoostWrapper(name="drift_init_auc", use_imbalance=False)
    model.fit(X_r, y_r)

    # hold-out AUC → reference for PHT
    proba_ho = model.predict_proba(X_ho_s)
    try:
        ho_auc = float(roc_auc_score(y_ho, proba_ho)) if y_ho.sum() > 0 else 0.5
    except Exception:
        ho_auc = 0.5
    threshold = _best_threshold(y_ho, proba_ho)

    logger.info(
        f"Burn-in 1999–{BURN_IN_END}: train={len(X_tr_s):,}  hold-out={len(X_ho_s):,}  "
        f"hold-out AUC={ho_auc:.4f}  thr={threshold:.2f}"
    )
    return model, scaler, threshold, ho_auc


# ── PHT 年級 1-AUC 串流 ───────────────────────────────────────────────────────
def find_drift_year_auc(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    model,
    scaler,
    threshold: float,
    ref_auc: float,
    logger,
    pht_threshold: float = 0.5,
    pht_delta: float = 0.002,
) -> tuple[int | None, list[dict]]:
    """
    逐年計算初始模型在當年資料上的 AUC，將 1-AUC 餵給 PHT。
    ref_auc = burn-in hold-out AUC → PHT 的 reference_mean = 1 - ref_auc。

    回傳 (drift_year 或 None, 每年軌跡 list)
    """
    pht = PageHinkley(
        threshold=pht_threshold,
        delta=pht_delta,
        reference_mean=1.0 - ref_auc,   # 固定基準 = 1 - hold-out AUC
    )

    stream_years = sorted(
        int(yr) for yr in X_all["fyear"].unique()
        if BURN_IN_END < yr <= TRAIN_END_YEAR
    )

    trajectory: list[dict] = []
    drift_year = None

    for year in stream_years:
        mask_yr = X_all["fyear"] == year
        X_yr = X_all[mask_yr].drop(columns=["fyear"])
        y_yr = y_all[mask_yr].to_numpy()
        if len(y_yr) == 0:
            continue

        # 保持 burn-in scaler → 捕捉特徵分布偏移
        X_yr_s = pd.DataFrame(
            scaler.transform(X_yr.fillna(X_yr.mean())), columns=X_yr.columns
        )
        proba_yr = model.predict_proba(X_yr_s)
        try:
            yr_auc = float(roc_auc_score(y_yr, proba_yr)) if y_yr.sum() > 0 else 0.5
        except Exception:
            yr_auc = 0.5

        signal = 1.0 - yr_auc   # ← 核心修正：用 1-AUC 而非 binary error
        triggered = pht.update(signal)

        row = {
            "year": year,
            "n": len(y_yr),
            "bankruptcy_rate": float(y_yr.mean()),
            "AUC": yr_auc,
            "signal_1_minus_AUC": signal,
            "pht_cumsum": pht._ph - pht._ph_min,
            "pht_triggered": triggered,
        }
        trajectory.append(row)

        flag = "  ← DRIFT DETECTED" if triggered else ""
        logger.info(
            f"  year={year}  n={len(y_yr):,}  AUC={yr_auc:.4f}  "
            f"1-AUC={signal:.4f}  PHT_stat={row['pht_cumsum']:.3f}{flag}"
        )

        if triggered and drift_year is None:
            drift_year = year
            logger.info(f"  PHT (AUC signal) detected drift at year {year}!")
            # 不立即 break，繼續記錄軌跡供後續分析

    return drift_year, trajectory


# ── 集成訓練 + 評估 ──────────────────────────────────────────────────────────
def train_eval_ensemble(
    X_old_raw: pd.DataFrame, y_old: np.ndarray,
    X_new_raw: pd.DataFrame, y_new: np.ndarray,
    X_test_raw: pd.DataFrame, y_test: np.ndarray,
    label: str, logger,
) -> dict:
    """訓練 Old+New pool，回傳 ensemble_new3 / ensemble_all6 / ensemble_old3 指標。"""
    X_old_s, X_new_s, X_te_s, _ = _scale(X_old_raw, X_new_raw, X_test_raw)

    sampler = ImbalanceSampler()
    pool: dict[str, XGBoostWrapper] = {}

    n_old_val = max(1, int(len(X_old_s) * 0.2))
    n_new_val = max(1, int(len(X_new_s) * 0.2))
    X_val = pd.concat([X_old_s.iloc[-n_old_val:], X_new_s.iloc[-n_new_val:]], ignore_index=True)
    y_val = np.concatenate([y_old[-n_old_val:], y_new[-n_new_val:]])

    for s in POOL_SAMPLING:
        Xr, yr = sampler.apply_sampling(X_old_s.iloc[:-n_old_val], y_old[:-n_old_val], strategy=s)
        m = XGBoostWrapper(name=f"old_{s}_{label}", use_imbalance=False)
        m.fit(Xr, yr)
        pool[f"old_{s}"] = m

        Xr, yr = sampler.apply_sampling(X_new_s.iloc[:-n_new_val], y_new[:-n_new_val], strategy=s)
        m = XGBoostWrapper(name=f"new_{s}_{label}", use_imbalance=False)
        m.fit(Xr, yr)
        pool[f"new_{s}"] = m

    combos = {
        "ensemble_old3": [f"old_{s}" for s in POOL_SAMPLING],
        "ensemble_new3": [f"new_{s}" for s in POOL_SAMPLING],
        "ensemble_all6": [f"old_{s}" for s in POOL_SAMPLING] + [f"new_{s}" for s in POOL_SAMPLING],
    }
    results = {}
    for cname, keys in combos.items():
        val_p = np.mean([pool[k].predict_proba(X_val) for k in keys], axis=0)
        thr   = _best_threshold(y_val, val_p)
        te_p  = np.mean([pool[k].predict_proba(X_te_s) for k in keys], axis=0)
        m     = compute_metrics(y_test, te_p, threshold=thr)
        results[cname] = m
        logger.info(
            f"  [{label}|{cname}] thr={thr:.2f}  AUC={m['AUC']:.4f}  "
            f"F1={m['F1']:.4f}  Recall={m['Recall']:.4f}"
        )
    return results


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    logger = get_logger("Phase4b_AUC_Signal", console=True, file=True)
    set_seed(42)

    X_all, y_all = load_data(logger)

    # 固定測試集
    mask_test = (X_all["fyear"] >= TEST_START_YEAR) & (X_all["fyear"] <= TEST_END_YEAR)
    X_test_raw = X_all[mask_test].drop(columns=["fyear"]).reset_index(drop=True)
    y_test = y_all[mask_test].to_numpy()
    logger.info(f"Test: {len(X_test_raw):,} rows  pos={y_test.mean()*100:.1f}%")

    # 訓練流
    mask_train = (X_all["fyear"] >= 1999) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_stream = X_all[mask_train].reset_index(drop=True)
    y_stream = y_all[mask_train].reset_index(drop=True)

    # 1. Burn-in 初始模型（hold-out reference）
    logger.info(f"\n=== Burn-in 1999–{BURN_IN_END} ===")
    init_model, scaler, threshold, ref_auc = build_init_model(X_stream, y_stream, logger)

    # 2. PHT 年級 1-AUC 串流
    logger.info("\n=== PHT (year-level 1-AUC signal) ===")
    drift_year, trajectory = find_drift_year_auc(
        X_stream, y_stream, init_model, scaler, threshold, ref_auc, logger,
        pht_threshold=0.5,
        pht_delta=0.002,
    )
    if drift_year is None:
        logger.info("PHT 未觸發，使用 BURN_IN_END+1 作為 fallback")
        drift_year_use = BURN_IN_END + 1
    else:
        drift_year_use = drift_year

    # 3. 比較三種邊界的集成效能
    logger.info("\n=== Ensemble Comparison ===")
    all_rows: list[dict] = []

    boundaries = {
        "Fixed_2012": (FIXED_OLD_END, FIXED_NEW_START),        # 人工固定
        f"ROSS_2009": (ROSS_BOUNDARY - 1, ROSS_BOUNDARY),      # ROSS 選出
        f"PHT_AUC_{drift_year_use}": (drift_year_use - 1, drift_year_use),  # PHT 年級 AUC
    }

    for label, (old_end, new_start) in boundaries.items():
        mask_old = (X_stream["fyear"] >= 1999) & (X_stream["fyear"] <= old_end)
        mask_new = (X_stream["fyear"] >= new_start) & (X_stream["fyear"] <= TRAIN_END_YEAR)
        X_old_r = X_stream[mask_old].drop(columns=["fyear"]).reset_index(drop=True)
        y_old_r = y_stream[mask_old].to_numpy()
        X_new_r = X_stream[mask_new].drop(columns=["fyear"]).reset_index(drop=True)
        y_new_r = y_stream[mask_new].to_numpy()

        if len(X_old_r) == 0 or len(X_new_r) == 0:
            logger.warning(f"  [{label}] 資料不足，跳過")
            continue

        logger.info(
            f"\n  [{label}]  Old=1999-{old_end} ({len(X_old_r):,})  "
            f"New={new_start}-{TRAIN_END_YEAR} ({len(X_new_r):,})"
        )
        combo_results = train_eval_ensemble(
            X_old_r, y_old_r, X_new_r, y_new_r, X_test_raw, y_test,
            label=label, logger=logger,
        )
        for cname, metrics in combo_results.items():
            all_rows.append({
                "boundary_method": label,
                "old_end": old_end,
                "new_start": new_start,
                "ensemble": cname,
                **{k: metrics.get(k) for k in METRICS},
            })

    # 4. 儲存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 軌跡檔
    df_traj = pd.DataFrame(trajectory)
    path_traj = OUTPUT_DIR / "bk_drift_auc_signal_detection.csv"
    df_traj.to_csv(path_traj, index=False, float_format="%.6f")
    logger.info(f"\n年級 AUC 軌跡 → {path_traj}")

    # 比較表
    df_cmp = pd.DataFrame(all_rows)
    path_cmp = OUTPUT_DIR / "bk_drift_auc_signal_comparison.csv"
    df_cmp.to_csv(path_cmp, index=False, float_format="%.4f")
    logger.info(f"集成比較 → {path_cmp}")

    # 摘要印出
    logger.info("\n=== 各邊界方法最佳 ensemble（按 F1）===")
    best = df_cmp.sort_values("F1", ascending=False).groupby("boundary_method").first().reset_index()
    logger.info("\n" + best[["boundary_method", "ensemble", "AUC", "F1", "Recall", "Precision"]].to_string(index=False))

    logger.info("\n=== Phase 4b AUC Signal Drift Detection 完成 ===")


if __name__ == "__main__":
    main()
