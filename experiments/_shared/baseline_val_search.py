"""
Phase 1 baseline：在固定 validation 上搜尋超參數（不做內層 CV）。

與年份切割腳本一致：訓練集經採樣後為 X_fit，validation 為同一 fold、同一 method
下已切出的 X_val（未採樣）。搜尋目標為 validation ROC-AUC；選參後僅 fit 一次。

網格對齊常見教學／實驗設定（樹數、學習率、深度、min_child_weight；RF 多維隨機搜尋；
SVM 的 C 與 poly/rbf kernel）。
"""
from __future__ import annotations

import json
from itertools import product
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------------------------
# Grids（與使用者提供的 tuning 表一致）
# ---------------------------------------------------------------------------
XGB_GRID: Mapping[str, Sequence[Any]] = {
    "n_estimators": [100, 300, 500, 800, 1000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "min_child_weight": [1, 3, 5, 20, 50],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "gamma": [0.0, 0.1, 0.2],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 3.0, 10.0],
}

LGB_GRID: Mapping[str, Sequence[Any]] = {
    "num_leaves": [31, 50, 100, 150],
    "max_depth": [5, 7, 10, 15, -1],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 500, 800, 1000],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 0.1, 1.0, 3.0],
}

# RF：與舊版僅掃 n_estimators∈{50..200} 不同，納入深度／葉子／分裂／max_features，隨機採樣 n_iter 組。
RF_GRID: Mapping[str, Sequence[Any]] = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [None, 8, 12, 16, 20, 24, 32],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "min_samples_split": [2, 4, 8, 16, 32],
    "max_features": ["sqrt", "log2", 0.2, 0.35, 0.5, 0.65],
}

SVM_GRID: Mapping[str, Sequence[Any]] = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "kernel": ["poly", "rbf"],
}

LR_GRID: Mapping[str, Sequence[Any]] = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "solver": ["lbfgs", "saga"],
    "max_iter": [500, 1000, 2000],
}

TABM_GRID: Mapping[str, Sequence[Any]] = {
    "k": [16, 24, 32],
    "n_blocks": [2, 3, 4],
    "d_block": [128, 256, 384, 512],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "lr": [5e-4, 1e-3, 2e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4, 3e-4, 1e-3],
    "use_num_embeddings": [False, True],
    "d_embedding": [8, 16],
}

TABR_GRID: Mapping[str, Sequence[Any]] = {
    "d_main": [64, 96, 128, 160],
    "d_multiplier": [1.5, 2.0, 2.5],
    "encoder_n_blocks": [1, 2, 3],
    "predictor_n_blocks": [1, 2, 3],
    "context_size": [32, 64, 96, 128],
    "context_dropout": [0.0, 0.1, 0.2],
    "dropout0": [0.0, 0.1, 0.2, 0.3],
    "dropout1": [0.0, 0.1, 0.2, 0.3],
    "lr": [1e-4, 2e-4, 5e-4, 1e-3],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
}


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int).ravel()
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(y_score).ravel()))


def _dict_product(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


def _xgb_random_candidates(grid: Mapping[str, Sequence[Any]], n_iter: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    if n_iter <= 0:
        n_iter = 32
    keys = list(grid.keys())
    seen = set()
    out: List[Dict[str, Any]] = []
    guard = max(n_iter * 50, n_iter + 10)
    steps = 0
    while len(out) < n_iter and steps < guard:
        steps += 1
        cand = tuple(rng.choice(list(grid[k])) for k in keys)
        if cand in seen:
            continue
        seen.add(cand)
        out.append(dict(zip(keys, cand)))
    return out


def _rf_random_candidates(grid: Mapping[str, Sequence[Any]], n_iter: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """與 XGB 相同：由網格隨機抽樣不重複候選。"""
    if n_iter <= 0:
        n_iter = 32
    keys = list(grid.keys())
    seen = set()
    out: List[Dict[str, Any]] = []
    guard = max(n_iter * 80, n_iter + 10)
    steps = 0
    while len(out) < n_iter and steps < guard:
        steps += 1
        cand = tuple(rng.choice(list(grid[k])) for k in keys)
        if cand in seen:
            continue
        seen.add(cand)
        out.append(dict(zip(keys, cand)))
    return out


def _random_candidates_from_grid(
    grid: Mapping[str, Sequence[Any]],
    n_iter: int,
    rng: np.random.Generator,
    *,
    guard_factor: int = 80,
) -> List[Dict[str, Any]]:
    if n_iter <= 0:
        n_iter = 32
    keys = list(grid.keys())
    seen = set()
    out: List[Dict[str, Any]] = []
    guard = max(n_iter * guard_factor, n_iter + 10)
    steps = 0
    while len(out) < n_iter and steps < guard:
        steps += 1
        cand = tuple(rng.choice(list(grid[k])) for k in keys)
        if cand in seen:
            continue
        seen.add(cand)
        out.append(dict(zip(keys, cand)))
    return out


def rf_wrapper_kwargs_from_best(best: Mapping[str, Any]) -> Dict[str, Any]:
    """將 search_rf_on_val 回傳的 best 轉成 RandomForestWrapper 可接受的 kwargs（不含 name）。"""
    if not best:
        return {}
    out: Dict[str, Any] = {}
    if "n_estimators" in best:
        out["n_estimators"] = int(best["n_estimators"])
    if "max_depth" in best:
        v = best["max_depth"]
        out["max_depth"] = None if v is None else int(v)
    if "min_samples_leaf" in best:
        out["min_samples_leaf"] = int(best["min_samples_leaf"])
    if "min_samples_split" in best:
        out["min_samples_split"] = int(best["min_samples_split"])
    if "max_features" in best:
        v = best["max_features"]
        if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
            out["max_features"] = int(v)
        elif isinstance(v, (np.floating, float)):
            out["max_features"] = float(v)
        else:
            out["max_features"] = str(v)
    return out


def search_xgb_on_val(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_params: MutableMapping[str, Any],
    scale_pos_weight: float,
    *,
    n_iter: int,
    seed: int,
) -> Tuple[Dict[str, Any], float]:
    """
    以 validation AUC 選擇 XGB 超參數（隨機採樣 n_iter 組）。

    Returns:
        (best_candidate_over_grid, best_val_auc)；若訓練標籤無兩類則 ({}, nan)。
    """
    y_fit = np.asarray(y_fit).ravel()
    if len(np.unique(y_fit)) < 2:
        return {}, float("nan")

    base = dict(base_params)
    base.pop("scale_pos_weight", None)

    rng = np.random.default_rng(int(seed))
    candidates = _xgb_random_candidates(XGB_GRID, n_iter, rng)
    best_auc = -1.0
    best_cand: Dict[str, Any] = {}

    for cand in candidates:
        params = {**base, **cand, "scale_pos_weight": float(scale_pos_weight)}
        clf = xgb.XGBClassifier(**params)
        try:
            clf.fit(X_fit, y_fit, verbose=False)
        except Exception:
            continue
        try:
            proba = clf.predict_proba(X_val)[:, 1]
        except Exception:
            continue
        auc = _safe_roc_auc(np.asarray(y_val), proba)
        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_cand = dict(cand)

    if not best_cand:
        return {}, float("nan")
    return best_cand, float(best_auc)


def search_lgb_on_val(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_params: MutableMapping[str, Any],
    scale_pos_weight: float,
    *,
    n_iter: int,
    seed: int,
) -> Tuple[Dict[str, Any], float]:
    """以 validation AUC 選擇 LightGBM 超參數（隨機採樣 n_iter 組）。

    Notes:
      - 使用 sklearn API `lgb.LGBMClassifier` 以支援 `n_estimators`。
      - 一律套用 `scale_pos_weight`（由呼叫端計算）。

    Returns:
        (best_candidate_over_grid, best_val_auc)；若訓練標籤無兩類則 ({}, nan)。
    """
    y_fit = np.asarray(y_fit).astype(int).ravel()
    if len(np.unique(y_fit)) < 2:
        return {}, float("nan")

    base = dict(base_params)
    # 避免與 scale_pos_weight 衝突
    base.pop("is_unbalance", None)
    base.pop("scale_pos_weight", None)

    rng = np.random.default_rng(int(seed))
    candidates = _random_candidates_from_grid(LGB_GRID, int(n_iter), rng, guard_factor=120)

    best_auc = -1.0
    best_cand: Dict[str, Any] = {}

    for idx, cand in enumerate(candidates):
        params = {
            **base,
            **cand,
            "scale_pos_weight": float(scale_pos_weight),
            "n_jobs": -1,
            "random_state": int(seed) + idx,
            "verbosity": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        try:
            clf.fit(X_fit, y_fit)
            proba = clf.predict_proba(X_val)[:, 1]
        except Exception:
            continue
        auc = _safe_roc_auc(np.asarray(y_val), proba)
        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_cand = dict(cand)

    if not best_cand:
        return {}, float("nan")
    return best_cand, float(best_auc)


def search_lr_on_val(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_params: MutableMapping[str, Any],
    *,
    n_iter: int,
    seed: int,
) -> Tuple[Dict[str, Any], float]:
    """以 validation AUC 選擇 LogisticRegression 超參數。

    Notes:
      - 只在 validation 上選參，不做內層 CV（與 Phase1 baseline 其他模型一致）。
      - LR 網格通常較小：若 n_iter >= 全組合數，會改為掃全網格以提高穩定性。

    Returns:
        (best_candidate_over_grid, best_val_auc)；若訓練標籤無兩類則 ({}, nan)。
    """
    y_fit = np.asarray(y_fit).astype(int).ravel()
    if len(np.unique(y_fit)) < 2:
        return {}, float("nan")

    base = dict(base_params)

    # 候選：小網格時掃全網格；否則隨機抽樣 n_iter 組
    total = 1
    for v in LR_GRID.values():
        total *= max(len(list(v)), 1)

    if n_iter is None or n_iter <= 0:
        n_iter = 32
    rng = np.random.default_rng(int(seed))
    if n_iter >= total:
        candidates = _dict_product(LR_GRID)
    else:
        candidates = _random_candidates_from_grid(LR_GRID, int(n_iter), rng, guard_factor=120)

    best_auc = -1.0
    best_cand: Dict[str, Any] = {}

    for cand in candidates:
        params = {**base, **cand}
        # 與 wrapper 對齊：預設 class_weight=balanced
        params.setdefault("class_weight", "balanced")
        try:
            clf = LogisticRegression(**params)
            clf.fit(X_fit, y_fit)
            proba = clf.predict_proba(X_val)[:, 1]
        except Exception:
            continue
        auc = _safe_roc_auc(np.asarray(y_val), proba)
        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_cand = dict(cand)

    if not best_cand:
        return {}, float("nan")
    return best_cand, float(best_auc)


def search_rf_on_val(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    n_iter: int = 48,
    seed: int = 42,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], float]:
    """RF：在多維網格上隨機採樣 n_iter 組，以 validation ROC-AUC 選最佳。"""
    y_fit = np.asarray(y_fit).ravel()
    if len(np.unique(y_fit)) < 2:
        return {}, float("nan")

    rng = np.random.default_rng(int(seed))
    ni = int(n_iter) if int(n_iter) > 0 else 32
    candidates = _rf_random_candidates(RF_GRID, ni, rng)

    best_auc = -1.0
    best_cand: Dict[str, Any] = {}
    for cand in candidates:
        params = {
            **cand,
            "class_weight": "balanced",
            "random_state": int(seed),
            "n_jobs": int(n_jobs),
        }
        clf = RandomForestClassifier(**params)
        try:
            clf.fit(X_fit, y_fit)
            proba = clf.predict_proba(X_val)[:, 1]
        except Exception:
            continue
        auc = _safe_roc_auc(np.asarray(y_val), proba)
        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_cand = dict(cand)
    if not best_cand:
        return {}, float("nan")
    return best_cand, float(best_auc)


def search_svm_on_val(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float]:
    """SVM：C × kernel 全掃；probability=True（與 SVMWrapper 一致）。"""
    y_fit = np.asarray(y_fit).ravel()
    if len(np.unique(y_fit)) < 2:
        return {}, float("nan")

    best_auc = -1.0
    best_cand: Dict[str, Any] = {}
    for cand in _dict_product(SVM_GRID):
        clf = SVC(
            C=cand["C"],
            kernel=cand["kernel"],
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=seed,
            max_iter=-1,
            cache_size=500,
        )
        try:
            clf.fit(X_fit, y_fit)
            proba = clf.predict_proba(X_val)[:, 1]
        except Exception:
            continue
        auc = _safe_roc_auc(np.asarray(y_val), proba)
        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_cand = dict(cand)
    if not best_cand:
        return {}, float("nan")
    return best_cand, float(best_auc)


def search_tabm_on_val(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_params: MutableMapping[str, Any] | None = None,
    *,
    n_iter: int = 24,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float]:
    """TabM：在隨機候選上以 validation ROC-AUC 選最佳。"""
    from src.models import TabMWrapper

    y_fit = np.asarray(y_fit).ravel()
    if len(np.unique(y_fit)) < 2:
        return {}, float("nan")

    base = dict(base_params or {})
    rng = np.random.default_rng(int(seed))
    ni = int(n_iter) if int(n_iter) > 0 else 24
    candidates = _random_candidates_from_grid(TABM_GRID, ni, rng, guard_factor=120)

    best_auc = -1.0
    best_cand: Dict[str, Any] = {}
    for idx, cand in enumerate(candidates):
        # 若不使用數值 embedding，d_embedding 對結果無意義，避免記錄噪音。
        if not bool(cand.get("use_num_embeddings", False)):
            cand = {k: v for k, v in cand.items() if k != "d_embedding"}

        params = {
            **base,
            **cand,
            "seed": int(seed) + idx,
        }
        model = TabMWrapper(name="tabm_tune", **params)
        try:
            model.fit(X_fit, y_fit)
            proba = model.predict_proba(X_val)
        except Exception:
            continue
        auc = _safe_roc_auc(np.asarray(y_val), proba)
        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_cand = dict(cand)
    if not best_cand:
        return {}, float("nan")
    return best_cand, float(best_auc)


def search_tabr_on_val(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_params: MutableMapping[str, Any] | None = None,
    *,
    n_iter: int = 24,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float]:
    """TabR：在隨機候選上以 validation ROC-AUC 選最佳。"""
    from src.models import TabRWrapper

    y_fit = np.asarray(y_fit).ravel()
    if len(np.unique(y_fit)) < 2:
        return {}, float("nan")

    base = dict(base_params or {})
    rng = np.random.default_rng(int(seed))
    ni = int(n_iter) if int(n_iter) > 0 else 24
    candidates = _random_candidates_from_grid(TABR_GRID, ni, rng, guard_factor=120)

    best_auc = -1.0
    best_cand: Dict[str, Any] = {}
    for idx, cand in enumerate(candidates):
        params = {
            **base,
            **cand,
            "seed": int(seed) + idx,
        }
        model = TabRWrapper(name="tabr_tune", **params)
        try:
            model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
            proba = model.predict_proba(X_val)
        except Exception:
            continue
        auc = _safe_roc_auc(np.asarray(y_val), proba)
        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_cand = dict(cand)
    if not best_cand:
        return {}, float("nan")
    return best_cand, float(best_auc)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_value(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _json_safe_params(d: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: _json_safe_value(v) for k, v in d.items()}


def tuning_meta(best: Mapping[str, Any], val_auc: float) -> Dict[str, Any]:
    """附加到 metrics row。"""
    return {
        "tune_val_auc": val_auc,
        "tune_best_params": json.dumps(_json_safe_params(best), sort_keys=True) if best else "",
    }


def export_tuning_log(df_raw: pd.DataFrame, path, logger=None) -> None:
    """若 raw 含 tuning 欄位，輸出窄表方便閱讀。"""
    if "tune_best_params" not in df_raw.columns:
        return
    sub = df_raw[["split", "method", "sampling", "tune_val_auc", "tune_best_params"]].copy()
    nonempty = sub["tune_best_params"].astype(str).str.strip().str.len() > 0
    sub = sub.loc[nonempty]
    if sub.empty:
        return
    sub.to_csv(path, index=False, float_format="%.6f")
    if logger is not None:
        logger.info(f"  Saved -> {path.name}")
