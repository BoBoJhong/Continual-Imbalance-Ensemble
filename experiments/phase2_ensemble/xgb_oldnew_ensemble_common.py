"""
Common utilities for XGB Old/New ensemble (year splits).

靜態集成：`ensemble_metrics_with_threshold`（驗證集上對平均機率搜 F1 最佳閾值）。

動態集成（DES）：`dynamic_ensemble_metrics_with_threshold` — 以 New 時期 scaler 後之驗證集為 DSEL，
在該特徵空間做 kNN；各基學習器在鄰域上的「是否預測正確」由各模型在 DSEL 上之機率與固定閾值（預設 0.5）
決定。支援 KNORA-E、KNORA-U、DES-KNN（與 DESlib 文獻對齊之典型方法）。驗證集閾值校準使用 leave-one-out
鄰居，避免樣本當作自己的近鄰造成洩漏。
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any, List, Dict, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

from src.data import ImbalanceSampler, DataPreprocessor
from src.models import XGBoostWrapper
from src.evaluation import compute_metrics


# 與研究方向一致：Old/New 各 3 個模型 = under / over / hybrid（無 none）
SAMPLING_STRATEGIES: List[str] = ["undersampling", "oversampling", "hybrid"]

# 長表 type：k 模型組合列專用（見 combination_metrics_6_models 說明）
TYPE_K_SUBSETS_MEAN = "k_subsets_mean"
# 每個合法 k 子集一列明細（與下方六槽位順序一致）
TYPE_K_SUBSET_DETAIL = "k_subset"

# 動態 DES：六模型池（Old×3 + New×3）在長表中的 sampling_col 標記
SAMPLING_DYNAMIC_ALL6 = "all6"

# 文獻常見 DES：KNORA-E / KNORA-U（Ko et al., 2008）、DES-KNN 權重（Cruz et al., 2018; DESlib）
DYNAMIC_DES_METHODS: Tuple[Tuple[str, str], ...] = (
    ("KNORA_E", "Dynamic_KNORA_E"),
    ("KNORA_U", "Dynamic_KNORA_U"),
    ("DES_KNN", "Dynamic_DES_KNN"),
)

# 6 槽 = [Old_under, Old_over, Old_hybrid, New_under, New_over, New_hybrid]
SIX_MODEL_SLOT_LABELS: Tuple[str, ...] = (
    "Old_under",
    "Old_over",
    "Old_hybrid",
    "New_under",
    "New_over",
    "New_hybrid",
)


def subset_label_from_indices(sorted_idx: Tuple[int, ...]) -> str:
    """由已排序的槽位索引產生可讀標籤，例如 (0,4) -> Old_under+New_over。"""
    return "+".join(SIX_MODEL_SLOT_LABELS[i] for i in sorted_idx)


def format_oyear_nyear(year_combo: str) -> str:
    """將年份窗標記 '10+6' 轉為 'O10 + N6'；'avg' 保留。"""
    s = str(year_combo).strip()
    if s == "avg":
        return "avg"
    if "+" in s:
        a, b = s.split("+", 1)
        a, b = a.strip(), b.strip()
        if a.isdigit() and b.isdigit():
            return f"O{a} + N{b}"
    return s


def _dataset_sort_key(d: str) -> Tuple[int, int, int]:
    if d == "avg":
        return (2, 0, 0)
    m = re.match(r"^O(\d+)\s*\+\s*N(\d+)$", str(d).strip())
    if m:
        return (0, int(m.group(1)), int(m.group(2)))
    return (1, 0, 0)


def _ensemble_sort_key(e: str) -> Tuple[int, int]:
    if str(e).endswith("models"):
        try:
            k = int(str(e).replace("models", ""))
            return (0, k)
        except ValueError:
            return (0, 99)
    dyn = {m[1]: i for i, m in enumerate(DYNAMIC_DES_METHODS)}
    if str(e) in dyn:
        return (2, dyn[str(e)])
    order = {"New": 1, "Old": 2, "OldNew": 3}
    return (1, order.get(str(e), 99))


def _type_sort_key(t: str) -> Tuple[int, int]:
    order = {
        "under": 0,
        "over": 1,
        "hybrid": 2,
        SAMPLING_DYNAMIC_ALL6: 3,
        TYPE_K_SUBSET_DETAIL: 4,
        TYPE_K_SUBSETS_MEAN: 5,
    }
    return (0, order.get(str(t), 9))


def _subset_sort_key(s: Any) -> Tuple:
    """依 subset_indices 數字序排序；空字串排在前（與 k_subsets_mean 等同列）。"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return (-1,)
    t = str(s).strip()
    if not t:
        return (-1,)
    parts = t.split(",")
    try:
        return tuple(int(x) for x in parts)
    except ValueError:
        return (999, t)


def build_long_metric_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    欄位：dataset（Oyear + Nyear）、ensemble、type、subset、subset_indices（若有）、<metric>。

    type 為 under/over/hybrid：單一採樣；k_subset：每個合法 k 子集一列；k_subsets_mean：同 k 下
    所有子集指標之平均（macro over subsets）。
    """
    work = df.copy()
    if "subset" not in work.columns:
        work["subset"] = ""
    else:
        work["subset"] = work["subset"].fillna("").astype(str)
    if "subset_indices" not in work.columns:
        work["subset_indices"] = ""
    else:
        work["subset_indices"] = work["subset_indices"].fillna("").astype(str)

    id_cols = ["dataset", "ensemble", "sampling_col", "subset", "subset_indices", metric]
    sub = work[id_cols].copy()
    sub = sub.rename(columns={"sampling_col": "type"})
    sub["dataset"] = sub["dataset"].map(format_oyear_nyear)

    avg_parts: List[Dict] = []
    for (ens, typ, sl, six), g in work.groupby(
        ["ensemble", "sampling_col", "subset", "subset_indices"], sort=False
    ):
        avg_parts.append(
            {
                "dataset": format_oyear_nyear("avg"),
                "ensemble": ens,
                "type": typ,
                "subset": "" if sl is None or (isinstance(sl, float) and pd.isna(sl)) else str(sl),
                "subset_indices": "" if six is None or (isinstance(six, float) and pd.isna(six)) else str(six),
                metric: float(g[metric].mean()),
            }
        )
    avg_df = pd.DataFrame(avg_parts)
    out = pd.concat([sub, avg_df], ignore_index=True)
    out[metric] = out[metric].round(4)
    out["_ds"] = out["dataset"].map(_dataset_sort_key)
    out["_en"] = out["ensemble"].map(_ensemble_sort_key)
    out["_ty"] = out["type"].map(_type_sort_key)
    out["_si"] = out["subset_indices"].map(_subset_sort_key)
    out = out.sort_values(["_ds", "_en", "_ty", "_si"], kind="stable").drop(
        columns=["_ds", "_en", "_ty", "_si"]
    )
    return out.reset_index(drop=True)


def format_raw_dataframe_for_export(df: pd.DataFrame, metric_cols: List[str], decimals: int = 4) -> pd.DataFrame:
    """輸出用 raw：dataset 格式化、欄位 sampling_col 改名 type、指標四捨五入。"""
    out = df.copy()
    out["dataset"] = out["dataset"].map(format_oyear_nyear)
    out = out.rename(columns={"sampling_col": "type"})
    for c in metric_cols:
        if c in out.columns:
            out[c] = out[c].round(decimals)
    if "threshold" in out.columns:
        out["threshold"] = out["threshold"].round(decimals)
    return out


def _normalize_sampling_col_column(df: pd.DataFrame) -> pd.DataFrame:
    """raw 匯出檔欄位為 type；實驗內部為 sampling_col。"""
    out = df.copy()
    if "sampling_col" not in out.columns and "type" in out.columns:
        out = out.rename(columns={"type": "sampling_col"})
    return out


def expected_summary_wide_columns() -> List[str]:
    """論文用寬表欄位順序：Old/New/OldNew × 採樣 + 動態 DES × (採樣|all6) + 2～6 models 之子集宏平均。"""
    return expected_summary_wide_columns_static_only() + expected_summary_wide_columns_des_only()


def expected_summary_wide_columns_static_only() -> List[str]:
    cols: List[str] = []
    for ens in ("Old", "New", "OldNew"):
        for st in ("under", "over", "hybrid"):
            cols.append(f"{ens}_{st}")
    for k in range(2, 7):
        cols.append(f"{k}models_subsets_mean")
    return cols


def expected_summary_wide_columns_des_only() -> List[str]:
    cols: List[str] = []
    for _, ens_name in DYNAMIC_DES_METHODS:
        for st in ("under", "over", "hybrid"):
            cols.append(f"{ens_name}_{st}")
        cols.append(f"{ens_name}_{SAMPLING_DYNAMIC_ALL6}")
    return cols


def build_summary_wide(
    df: pd.DataFrame,
    metric: str,
    *,
    columns_order: List[str] | None = None,
) -> pd.DataFrame:
    """
    精簡寬表：列為 dataset（Oyear+Nyear），欄為方法；剔除 k_subset 明細列。
    最後一列 avg 為各切分上該欄位數值的平均。
    """
    work = _normalize_sampling_col_column(df)
    work = work.loc[work["sampling_col"] != TYPE_K_SUBSET_DETAIL].copy()
    ens = work["ensemble"].astype(str)
    sc = work["sampling_col"].astype(str)
    work["_col"] = np.where(
        sc == TYPE_K_SUBSETS_MEAN,
        ens + "_subsets_mean",
        ens + "_" + sc,
    )
    pivot = work.pivot_table(
        index="dataset",
        columns="_col",
        values=metric,
        aggfunc="mean",
    )
    ordered = columns_order if columns_order is not None else expected_summary_wide_columns()
    for c in ordered:
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot = pivot[ordered]
    pivot = pivot.reset_index()
    pivot["dataset"] = pivot["dataset"].map(format_oyear_nyear)
    pivot["_sk"] = pivot["dataset"].map(_dataset_sort_key)
    pivot = pivot.sort_values("_sk", kind="stable").drop(columns=["_sk"])
    method_cols = [c for c in pivot.columns if c != "dataset"]
    avg_row = {c: float(pivot[c].mean()) for c in method_cols}
    avg_row["dataset"] = format_oyear_nyear("avg")
    pivot = pd.concat([pivot, pd.DataFrame([avg_row])], ignore_index=True)
    for c in method_cols:
        pivot[c] = pivot[c].round(4)
    return pivot.reset_index(drop=True)


def combination_metrics_6_models_details(
    y_val: np.ndarray,
    val_probas_6: List[np.ndarray],
    y_test: np.ndarray,
    test_probas_6: List[np.ndarray],
    k: int,
) -> List[Dict[str, Any]]:
    """
    回傳每個「大小 k、至少一 Old、一 New」子集的指標；subset_indices 為排序後的槽位索引字串（0..5）。
    """
    assert len(val_probas_6) == 6 and len(test_probas_6) == 6
    subsets = _subsets_of_size_k(6, k, require_old_new=True, n_old=3)
    rows: List[Dict[str, Any]] = []
    for idx in subsets:
        canon = tuple(sorted(idx))
        vp = [val_probas_6[i] for i in idx]
        tp = [test_probas_6[i] for i in idx]
        m = ensemble_metrics_with_threshold(y_val, vp, y_test, tp)
        rows.append(
            {
                "subset_label": subset_label_from_indices(canon),
                "subset_indices": ",".join(str(i) for i in canon),
                "AUC": float(m["AUC"]),
                "F1": float(m["F1"]),
                "Recall": float(m["Recall"]),
            }
        )
    return rows


def export_ensemble_long_tables_and_raw(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    metric_cols: List[str],
    table_filename_fmt: str,
    raw_csv_name: str,
    logger: logging.Logger,
    summary_wide_suffix: str | None = None,
    summary_wide_filename_fmt: str | None = None,
    summary_wide_columns: List[str] | None = None,
) -> None:
    """寫入各指標長表 CSV、raw CSV；若給 summary_wide_suffix 則另寫精簡寬表（論文用）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metric_cols:
        long_df = build_long_metric_table(df, metric)
        out_csv = output_dir / table_filename_fmt.format(metric=metric)
        long_df.to_csv(out_csv, index=False, float_format="%.4f")
        logger.info(f"\nSaved -> {out_csv}")

    raw_out = format_raw_dataframe_for_export(df, metric_cols, decimals=4)
    raw_path = output_dir / raw_csv_name
    raw_out.to_csv(raw_path, index=False, float_format="%.4f")
    logger.info(f"Saved -> {raw_path}")

    if summary_wide_suffix:
        wide_fmt = summary_wide_filename_fmt or "xgb_oldnew_ensemble_{metric}_summary_wide_{suffix}.csv"
        for metric in metric_cols:
            wide = build_summary_wide(df, metric, columns_order=summary_wide_columns)
            spath = output_dir / wide_fmt.format(metric=metric, suffix=summary_wide_suffix)
            wide.to_csv(spath, index=False, float_format="%.4f")
            logger.info(f"Saved -> {spath}")


def select_threshold_from_validation(y_val: np.ndarray, y_proba_val: np.ndarray) -> float:
    """Use validation set to select best threshold by maximizing F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return float(best_t)


def _as_float64_2d(X: Any) -> np.ndarray:
    if hasattr(X, "values"):
        X = X.values
    return np.asarray(X, dtype=np.float64)


def _stack_positive_probas(probas: List[np.ndarray]) -> np.ndarray:
    """(n, n_models) 正類機率。"""
    cols = []
    for p in probas:
        a = np.asarray(p)
        if a.ndim == 2 and a.shape[1] >= 2:
            cols.append(a[:, 1].astype(np.float64, copy=False))
        else:
            cols.append(a.reshape(-1).astype(np.float64, copy=False))
    return np.column_stack(cols)


def _neighbor_indices_loo(X_dsel: np.ndarray, k: int) -> np.ndarray:
    """每個 DSEL 點的鄰居索引，不含自身；欄數 = min(k, n-1)。"""
    n = X_dsel.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    k_take = min(k, n - 1)
    k_req = min(k_take + 1, n)
    nn = NearestNeighbors(n_neighbors=k_req, metric="minkowski", p=2).fit(X_dsel)
    _, idx = nn.kneighbors(X_dsel)
    out = np.zeros((n, k_take), dtype=np.int64)
    for i in range(n):
        row = idx[i]
        filt = row[row != i][:k_take]
        out[i, : filt.shape[0]] = filt
    return out


def _neighbor_indices_query(
    X_dsel: np.ndarray, X_query: np.ndarray, k: int
) -> np.ndarray:
    n = X_dsel.shape[0]
    k_eff = max(1, min(k, n))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="minkowski", p=2).fit(X_dsel)
    _, idx = nn.kneighbors(X_query)
    return idx


def _dynamic_proba_rows(
    X_dsel: np.ndarray,
    y_dsel: np.ndarray,
    val_proba_mat: np.ndarray,
    X_query: np.ndarray,
    query_proba_mat: np.ndarray,
    method: str,
    k_neighbors: int,
    *,
    loo: bool,
    hard_label_threshold: float,
) -> np.ndarray:
    """
    在 X_dsel 上定義 DSEL；鄰域以 **New scaler** 特徵空間之 kNN。
    各基學習器在鄰居上是否「預測正確」以 val 上之硬判定（proba >= threshold）比對真實標籤。
    """
    y_dsel = np.asarray(y_dsel).astype(np.int64, copy=False)
    val_hard = (val_proba_mat >= hard_label_threshold).astype(np.int64)
    n_q = X_query.shape[0]
    n_models = val_proba_mat.shape[1]
    if query_proba_mat.shape != (n_q, n_models):
        raise ValueError("query_proba_mat 與 X_query 列數或模型數不一致")

    if loo:
        neigh_idx = _neighbor_indices_loo(X_dsel, k_neighbors)
    else:
        neigh_idx = _neighbor_indices_query(X_dsel, X_query, k_neighbors)

    k_eff = neigh_idx.shape[1]
    if k_eff == 0:
        return query_proba_mat.mean(axis=1)

    out = np.zeros(n_q, dtype=np.float64)
    for j in range(n_q):
        idx_j = neigh_idx[j]
        ny = y_dsel[idx_j]
        nh = val_hard[idx_j]
        correct = nh == ny[:, None]
        row = query_proba_mat[j]
        if method == "KNORA_E":
            mask = correct.all(axis=0)
            out[j] = float(row[mask].mean()) if mask.any() else float(row.mean())
        elif method == "KNORA_U":
            mask = correct.any(axis=0)
            out[j] = float(row[mask].mean()) if mask.any() else float(row.mean())
        elif method == "DES_KNN":
            w = correct.mean(axis=0)
            s = float(w.sum())
            out[j] = float((row * w).sum() / s) if s > 0 else float(row.mean())
        else:
            raise ValueError(f"未知 method={method!r}，請用 KNORA_E | KNORA_U | DES_KNN")
    return out


def dynamic_ensemble_metrics_with_threshold(
    y_val: np.ndarray,
    X_val_new_scaled: Any,
    val_probas: List[np.ndarray],
    y_test: np.ndarray,
    X_test_new_scaled: Any,
    test_probas: List[np.ndarray],
    *,
    method: str,
    k_neighbors: int = 7,
    hard_label_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    動態集成（DES）：以驗證集為 DSEL，在 New 特徵空間做 kNN，再依方法融合 test 上各模型正類機率。

    閾值：先在驗證集上以 **leave-one-out 鄰居** 計算動態融合機率，再以 F1 格搜最佳 threshold，
    與靜態 `ensemble_metrics_with_threshold` 精神一致，避免驗證點當自己的鄰居造成洩漏。
    """
    Xv = _as_float64_2d(X_val_new_scaled)
    Xt = _as_float64_2d(X_test_new_scaled)
    vmat = _stack_positive_probas(val_probas)
    tmat = _stack_positive_probas(test_probas)

    val_dyn = _dynamic_proba_rows(
        Xv,
        y_val,
        vmat,
        Xv,
        vmat,
        method,
        k_neighbors,
        loo=True,
        hard_label_threshold=hard_label_threshold,
    )
    best_t = select_threshold_from_validation(np.asarray(y_val), val_dyn)

    test_dyn = _dynamic_proba_rows(
        Xv,
        y_val,
        vmat,
        Xt,
        tmat,
        method,
        k_neighbors,
        loo=False,
        hard_label_threshold=hard_label_threshold,
    )
    metrics = compute_metrics(np.asarray(y_test), test_dyn, threshold=best_t)
    metrics["threshold"] = best_t
    return metrics


def train_one_sampling_xgb(
    X_train_scaled: pd.DataFrame,
    y_train: np.ndarray,
    X_val_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    sampler: ImbalanceSampler,
    sampling_strategy: str,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train one XGB model under sampling strategy and return (val_proba, test_proba)."""
    X_r, y_r = sampler.apply_sampling(X_train_scaled, y_train, strategy=sampling_strategy)
    model = XGBoostWrapper(name=model_name)
    model.fit(X_r, y_r)
    val_proba = model.predict_proba(X_val_scaled)
    test_proba = model.predict_proba(X_test_scaled)
    return val_proba, test_proba


def ensemble_metrics_with_threshold(
    y_val: np.ndarray,
    val_probas: List[np.ndarray],
    y_test: np.ndarray,
    test_probas: List[np.ndarray],
) -> Dict[str, float]:
    """Average probabilities, pick best threshold on val, evaluate on test."""
    val_avg = np.mean(val_probas, axis=0)
    best_t = select_threshold_from_validation(y_val, val_avg)

    test_avg = np.mean(test_probas, axis=0)
    metrics = compute_metrics(y_test, test_avg, threshold=best_t)
    metrics["threshold"] = best_t
    return metrics


def _subsets_of_size_k(n: int, k: int, require_old_new: bool, n_old: int = 3) -> List[Tuple[int, ...]]:
    """All subsets of range(n) of size k. If require_old_new, only those with >=1 from [0..n_old-1] and >=1 from [n_old..n-1]."""
    out = []
    for idx in itertools.combinations(range(n), k):
        if require_old_new:
            has_old = any(i < n_old for i in idx)
            has_new = any(i >= n_old for i in idx)
            if not (has_old and has_new):
                continue
        out.append(idx)
    return out


def combination_metrics_6_models(
    y_val: np.ndarray,
    val_probas_6: List[np.ndarray],
    y_test: np.ndarray,
    test_probas_6: List[np.ndarray],
    k: int,
) -> Dict[str, float]:
    """
    6 個模型 = [Old_under, Old_over, Old_hybrid, New_under, New_over, New_hybrid]。
    對 `combination_metrics_6_models_details` 各子集指標取平均（macro over subsets）；長表另輸出
    type=k_subsets_mean 一列與 type=k_subset 多列明細。
    """
    details = combination_metrics_6_models_details(
        y_val, val_probas_6, y_test, test_probas_6, k
    )
    if not details:
        return {"AUC": np.nan, "F1": np.nan, "Recall": np.nan}
    return {
        "AUC": float(np.mean([d["AUC"] for d in details])),
        "F1": float(np.mean([d["F1"] for d in details])),
        "Recall": float(np.mean([d["Recall"] for d in details])),
    }

