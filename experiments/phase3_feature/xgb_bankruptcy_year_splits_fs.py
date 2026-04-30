"""
Phase 3 — Study II: 特徵選擇對集成分類器的影響（XGBoost + 年份切割 — Bankruptcy）
=============================================================================
在 Phase 2 靜態集成 (Old×3 + New×3) 的基礎上，嵌入 FeatureSelector。

流程（每組年份切割 × 每個 FS 配置）：
  1. get_bankruptcy_year_split → X_old, y_old, X_new, y_new, X_test, y_test
  2. 在 X_old 上 fit FeatureSelector → transform X_old, X_new, X_test
  3. scale → split fit/val → sample → train Old×3 + New×3
  4. 靜態集成 ensemble_metrics_with_threshold()
  5. 比較 no_fs vs 各 FS 方法 × 保留比例的指標差異

FS 配置：no_fs + 3 方法 (kbest_f / kbest_chi2 / lasso) × 3 比例 (r20 / r50 / r80) = 10 種
年份切割：15 組 (split_1+15 … split_15+1)
每組：Old×3 + New×3 = 6 個模型

輸出：results/phase3_feature/xgb_bankruptcy_fs_raw.csv
      results/phase3_feature/xgb_bankruptcy_fs_summary.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import DataPreprocessor, ImbalanceSampler
from src.models import XGBoostWrapper
from src.features import FeatureSelector
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

SAMPLING_STRATEGIES = ["undersampling", "oversampling", "hybrid"]
SAMPLING_COL_MAP = {
    "undersampling": "under",
    "oversampling": "over",
    "hybrid": "hybrid",
}

FS_METHODS = ["kbest_f", "kbest_chi2", "lasso"]
FS_RATIOS = [0.2, 0.5, 0.8]

# FS 配置清單：(tag, method, ratio) ; no_fs 為 (None, None)
FS_CONFIGS: List[Tuple[str, Optional[str], Optional[float]]] = [
    ("no_fs", None, None),
]
for method in FS_METHODS:
    for ratio in FS_RATIOS:
        tag = f"{method}_r{int(ratio * 100)}"
        FS_CONFIGS.append((tag, method, ratio))

METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]
OUTPUT_DIR = project_root / "results" / "phase3_feature"


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _split_fit_val_by_year(X_raw, y_raw, year_arr, val_ratio=0.2, random_state=42):
    """依年份逐年切 fit/val，避免跨年洩漏。"""
    years = np.asarray(year_arr)
    y_arr = np.asarray(y_raw)
    if years.size == 0 or len(np.unique(years)) <= 1:
        return _split_fit_val(X_raw, y_arr, val_ratio, random_state)

    fit_idx_all = []
    val_idx_all = []
    for yr in sorted(np.unique(years)):
        idx = np.where(years == yr)[0]
        n = len(idx)
        if n <= 1:
            fit_idx_all.extend(idx.tolist())
            continue
        n_val = max(1, min(int(round(n * val_ratio)), n - 1))
        stratify = None
        y_sub = y_arr[idx]
        if len(np.unique(y_sub)) >= 2 and n_val >= len(np.unique(y_sub)):
            stratify = y_sub
        try:
            idx_fit, idx_val = train_test_split(
                idx, test_size=n_val, random_state=random_state + int(yr), stratify=stratify,
            )
        except ValueError:
            idx_fit, idx_val = train_test_split(
                idx, test_size=n_val, random_state=random_state + int(yr),
            )
        fit_idx_all.extend(idx_fit.tolist())
        val_idx_all.extend(idx_val.tolist())

    if not val_idx_all:
        return _split_fit_val(X_raw, y_arr, val_ratio, random_state)

    fit_idx = np.array(sorted(fit_idx_all))
    val_idx = np.array(sorted(val_idx_all))
    return (
        X_raw.iloc[fit_idx].reset_index(drop=True),
        y_arr[fit_idx],
        X_raw.iloc[val_idx].reset_index(drop=True),
        y_arr[val_idx],
    )


def _split_fit_val(X_raw, y_raw, test_size=0.2, random_state=42):
    y_arr = np.asarray(y_raw)
    try:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_raw, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr,
        )
    except ValueError:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_raw, y_arr, test_size=test_size, random_state=random_state,
        )
    return X_fit, np.asarray(y_fit), X_val, np.asarray(y_val)


def _select_threshold_from_validation(y_val, y_proba_val):
    """用 validation set 搜尋最佳 F1 閾值。"""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _apply_feature_selection(
    fs_tag: str,
    fs_method: Optional[str],
    fs_ratio: Optional[float],
    X_old: pd.DataFrame,
    y_old: np.ndarray,
    X_new: pd.DataFrame,
    X_test: pd.DataFrame,
    logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """
    在 Old (Historical) 上 fit FeatureSelector，transform 全部。

    Returns:
        (X_old_fs, X_new_fs, X_test_fs, n_features_before, n_features_after)
    """
    n_before = X_old.shape[1]
    if fs_method is None:
        return X_old, X_new, X_test, n_before, n_before

    k = max(1, int(n_before * fs_ratio))
    fs = FeatureSelector(method=fs_method, k=k)
    X_old_fs = fs.fit_transform(X_old, y_old)
    X_new_fs = fs.transform(X_new)
    X_test_fs = fs.transform(X_test)
    n_after = X_old_fs.shape[1]
    logger.info(f"    FS [{fs_tag}]: {n_before} → {n_after} features")
    return X_old_fs, X_new_fs, X_test_fs, n_before, n_after


# ---------------------------------------------------------------------------
# 核心：對一組年份切割 × 一個 FS 配置，跑 Old×3 + New×3 靜態集成
# ---------------------------------------------------------------------------

def _run_one_split_one_fs(
    label: str,
    X_old: pd.DataFrame,
    y_old: np.ndarray,
    year_old: np.ndarray,
    X_new: pd.DataFrame,
    y_new: np.ndarray,
    year_new: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fs_tag: str,
    fs_method: Optional[str],
    fs_ratio: Optional[float],
    logger,
) -> List[dict]:
    """
    對一組 (split, fs_config) 跑完整靜態集成。

    回傳列表：每個 (ensemble_type, sampling) 一列指標。
    """
    rows: List[dict] = []

    # 1. 特徵選擇（在原始 X_old 上 fit，不含年份、已 handle_missing）
    X_old_fs, X_new_fs, X_test_fs, n_before, n_after = _apply_feature_selection(
        fs_tag, fs_method, fs_ratio, X_old, y_old, X_new, X_test, logger,
    )

    # 2. 切 fit/val（依年份）
    X_old_fit, y_old_fit, X_old_val, y_old_val = _split_fit_val_by_year(X_old_fs, y_old, year_old)
    X_new_fit, y_new_fit, X_new_val, y_new_val = _split_fit_val_by_year(X_new_fs, y_new, year_new)

    # 3. scale（Old scaler fit on Old_fit；New scaler fit on New_fit）
    pre_old = DataPreprocessor()
    X_old_fit_s = pre_old.scale_features(X_old_fit, fit=True)[0]
    X_old_val_s = pre_old.scale_features(X_old_val, fit=False)[0]
    X_test_s_old = pre_old.scale_features(X_test_fs, fit=False)[0]

    pre_new = DataPreprocessor()
    X_new_fit_s = pre_new.scale_features(X_new_fit, fit=True)[0]
    X_new_val_s = pre_new.scale_features(X_new_val, fit=False)[0]
    X_test_s_new = pre_new.scale_features(X_test_fs, fit=False)[0]

    # 合併 val（靜態集成的 threshold 校準用）
    X_val_all = pd.concat(
        [X_old_val_s, pre_old.scale_features(X_new_val, fit=False)[0]],  # Old scaler 上的 new_val
        ignore_index=True,
    )
    y_val_all = np.concatenate([y_old_val, y_new_val])

    sampler = ImbalanceSampler()

    # 4. 訓練 Old×3 + New×3
    old_val_probas: List[np.ndarray] = []
    old_test_probas: List[np.ndarray] = []
    new_val_probas: List[np.ndarray] = []
    new_test_probas: List[np.ndarray] = []

    for s in SAMPLING_STRATEGIES:
        X_r, y_r = sampler.apply_sampling(X_old_fit_s, y_old_fit, strategy=s)
        model = XGBoostWrapper(name=f"old_{label}_{fs_tag}_{s}")
        model.fit(X_r, y_r)
        old_val_probas.append(model.predict_proba(X_val_all))
        old_test_probas.append(model.predict_proba(X_test_s_old))

    for s in SAMPLING_STRATEGIES:
        X_r, y_r = sampler.apply_sampling(X_new_fit_s, y_new_fit, strategy=s)
        model = XGBoostWrapper(name=f"new_{label}_{fs_tag}_{s}")
        model.fit(X_r, y_r)
        # New 模型的 val：用 New scaler 上的合併 val
        X_val_new_all = pd.concat(
            [pre_new.scale_features(X_old_val, fit=False)[0], X_new_val_s],
            ignore_index=True,
        )
        new_val_probas.append(model.predict_proba(X_val_new_all))
        new_test_probas.append(model.predict_proba(X_test_s_new))

    y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    year_combo = label.split("split_", 1)[-1] if str(label).startswith("split_") else str(label)

    # 5. 計算各種靜態集成組合
    ensemble_combos = {
        "Old_3":     (list(range(3)),    []),       # Old under/over/hybrid
        "New_3":     ([],                list(range(3))),   # New under/over/hybrid
        "All_6":     (list(range(3)),    list(range(3))),   # 全部 6 個
        "Old_under_New_under": ([0], [0]),   # 單對
        "Old_hybrid_New_hybrid": ([2], [2]),
    }

    # 各採樣策略的單模型（用於與 no_fs baseline 直接比較）
    for si, s in enumerate(SAMPLING_STRATEGIES):
        sc = SAMPLING_COL_MAP[s]
        # Old single
        val_avg = old_val_probas[si]
        thr = _select_threshold_from_validation(y_val_all, val_avg)
        m = compute_metrics(y_test_arr, old_test_probas[si], threshold=thr)
        rows.append({
            "split": label, "dataset": year_combo, "fs": fs_tag,
            "n_features": n_after, "ensemble": "Old", "sampling": sc,
            **{k: m[k] for k in METRICS if k in m},
        })
        # New single
        val_avg = new_val_probas[si]
        thr = _select_threshold_from_validation(y_val_all, val_avg)
        m = compute_metrics(y_test_arr, new_test_probas[si], threshold=thr)
        rows.append({
            "split": label, "dataset": year_combo, "fs": fs_tag,
            "n_features": n_after, "ensemble": "New", "sampling": sc,
            **{k: m[k] for k in METRICS if k in m},
        })

    # 各集成組合
    for combo_name, (old_idx, new_idx) in ensemble_combos.items():
        val_ps = [old_val_probas[i] for i in old_idx] + [new_val_probas[i] for i in new_idx]
        test_ps = [old_test_probas[i] for i in old_idx] + [new_test_probas[i] for i in new_idx]
        val_avg = np.mean(val_ps, axis=0)
        thr = _select_threshold_from_validation(y_val_all, val_avg)
        test_avg = np.mean(test_ps, axis=0)
        m = compute_metrics(y_test_arr, test_avg, threshold=thr)
        rows.append({
            "split": label, "dataset": year_combo, "fs": fs_tag,
            "n_features": n_after, "ensemble": combo_name, "sampling": "ensemble",
            **{k: m[k] for k in METRICS if k in m},
        })

    return rows


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main():
    logger = get_logger("Phase3_FS_XGB_Bankruptcy", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    total_splits = len(YEAR_SPLITS)
    total_fs = len(FS_CONFIGS)

    logger.info(f"Study II: 特徵選擇 × XGBoost 年份切割 — Bankruptcy")
    logger.info(f"切割數: {total_splits}, FS 配置數: {total_fs}")
    logger.info(f"預估訓練次數: {total_splits * total_fs * 6}")

    for si, (label, old_end_year) in enumerate(YEAR_SPLITS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{si}/{total_splits}] {label}  (Old<={old_end_year}, New={old_end_year+1}-2014, Test=2015-2018)")
        logger.info("=" * 60)

        try:
            X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, year_test = (
                get_bankruptcy_year_split(logger, old_end_year=old_end_year, return_years=True)
            )
            y_old = np.asarray(y_old)
            y_new = np.asarray(y_new)
        except Exception as e:
            logger.error(f"  [ERROR] 資料載入失敗: {e}")
            continue

        for fi, (fs_tag, fs_method, fs_ratio) in enumerate(FS_CONFIGS, 1):
            logger.info(f"  [{fi}/{total_fs}] FS: {fs_tag}")
            try:
                split_rows = _run_one_split_one_fs(
                    label, X_old, y_old, year_old,
                    X_new, y_new, year_new,
                    X_test, y_test,
                    fs_tag, fs_method, fs_ratio,
                    logger,
                )
                all_rows.extend(split_rows)
                # 快速摘要
                for r in split_rows:
                    if r["ensemble"] == "All_6":
                        logger.info(
                            f"    All_6: AUC={r.get('AUC', 0):.4f}  "
                            f"F1={r.get('F1', 0):.4f}  Recall={r.get('Recall', 0):.4f}"
                        )
            except Exception as e:
                logger.error(f"    [ERROR] {fs_tag}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    if not all_rows:
        logger.error("無任何結果。")
        return

    # ---------------------------------------------------------------------------
    # 輸出 CSV
    # ---------------------------------------------------------------------------
    df_raw = pd.DataFrame(all_rows)
    raw_path = OUTPUT_DIR / "xgb_bankruptcy_fs_raw.csv"
    df_raw.to_csv(raw_path, index=False, float_format="%.6f")
    logger.info(f"\n原始結果: {raw_path.name}  ({len(df_raw)} rows)")

    # 精簡摘要：各 FS × ensemble 取跨 split 平均
    summary_rows = []
    for (fs, ens, samp), g in df_raw.groupby(["fs", "ensemble", "sampling"]):
        row = {"fs": fs, "ensemble": ens, "sampling": samp, "n_splits": len(g)}
        for m in METRICS:
            if m in g.columns:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std()
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "xgb_bankruptcy_fs_summary.csv"
    df_summary.to_csv(summary_path, index=False, float_format="%.4f")
    logger.info(f"摘要: {summary_path.name}  ({len(df_summary)} rows)")

    # AUC Diff 表（vs no_fs baseline，僅 All_6 ensemble）
    baseline = df_raw[(df_raw["fs"] == "no_fs") & (df_raw["ensemble"] == "All_6")]
    if not baseline.empty:
        baseline_auc_by_split = baseline.set_index("split")["AUC"]
        diff_rows = []
        for fs_tag in [c[0] for c in FS_CONFIGS if c[0] != "no_fs"]:
            sub = df_raw[(df_raw["fs"] == fs_tag) & (df_raw["ensemble"] == "All_6")]
            if sub.empty:
                continue
            sub_auc_by_split = sub.set_index("split")["AUC"]
            common = baseline_auc_by_split.index.intersection(sub_auc_by_split.index)
            if common.empty:
                continue
            diff = sub_auc_by_split[common] - baseline_auc_by_split[common]
            diff_rows.append({
                "fs": fs_tag,
                "AUC_no_fs": baseline_auc_by_split[common].mean(),
                "AUC_fs": sub_auc_by_split[common].mean(),
                "AUC_diff": diff.mean(),
                "AUC_diff_std": diff.std(),
                "n_positive": (diff > 0).sum(),
                "n_negative": (diff < 0).sum(),
                "n_total": len(common),
            })
        if diff_rows:
            df_diff = pd.DataFrame(diff_rows)
            diff_path = OUTPUT_DIR / "xgb_bankruptcy_fs_auc_diff.csv"
            df_diff.to_csv(diff_path, index=False, float_format="%.4f")
            logger.info(f"AUC Diff 表: {diff_path.name}")
            logger.info("\n=== AUC Diff vs no_fs (All_6 ensemble) ===")
            for _, r in df_diff.iterrows():
                sign = "+" if r["AUC_diff"] > 0 else ""
                logger.info(f"  {r['fs']:20s}: {sign}{r['AUC_diff']:.4f} "
                           f"(+{int(r['n_positive'])} / -{int(r['n_negative'])} / {int(r['n_total'])} splits)")

    logger.info(f"\n=== Study II Bankruptcy 完成 ===")
    logger.info(f"結果目錄: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
