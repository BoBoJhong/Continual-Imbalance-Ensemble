"""
Phase 1 — Diabetes 130 原始檔（diabetic_data.csv）五折交叉驗證基準

資料：data/raw/medical/diabetes130/diabetic_data.csv（完整 UCI 欄位）
標籤：readmitted == '<30' 為正類（1），其餘為 0。

模型：Logistic Regression、Random Forest、XGBoost、LightGBM。

不平衡採樣（與 medical_year_splits_* 一致）：
  - 在 ColumnTransformer 數值化後，對**訓練矩陣**套用 `ImbalanceSampler`：
    `none` / `undersampling`（Tomek）/ `oversampling`（ADASYN）/ `hybrid`（SMOTEENN）。
  - 驗證與測試折不採樣。預設四種策略皆跑；`--sampling none` 或 `--sampling none oversampling` 可指定子集。

交叉驗證（單純五折）：
  - 預設 `stratified_row`：StratifiedKFold；可選 `stratified_group`：StratifiedGroupKFold（依 patient_nbr）。
  - **超參數（預設開啟）**：每折將外層訓練集切約 85%/15% 內層（group 模式用 GroupShuffleSplit），
    在內層上以 **validation ROC-AUC** 隨機搜尋（`experiments._shared.baseline_val_search` 網格）；
    選參後以**整段外層訓練折**重 fit 前處理與模型，測試折 **threshold=0.5**。
  - `--no-tune`：關閉搜尋，使用腳本內固定預設超參數。

輸出：results/phase1_baseline/medical_diabetic130_kfold/<tag>/

用法：
  python experiments/phase1_baseline/medical_diabetic130_kfold_baselines.py
  python experiments/phase1_baseline/medical_diabetic130_kfold_baselines.py --no-tune
  python experiments/phase1_baseline/medical_diabetic130_kfold_baselines.py --tune-n-iter 48 --split stratified_group
  python experiments/phase1_baseline/medical_diabetic130_kfold_baselines.py --sampling none --no-tune
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.sampler import ImbalanceSampler
from src.evaluation import compute_metrics
from src.utils import get_config_loader, get_logger, set_seed

from experiments._shared.baseline_val_search import (
    search_lgb_on_val,
    search_lr_on_val,
    search_rf_on_val,
    search_xgb_on_val,
    tuning_meta,
)
from experiments._shared.diabetic130_raw import DEFAULT_DIABETIC_CSV, load_diabetic130_raw

METRICS = ("AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error")

# 與 experiments/phase1_baseline/medical_year_splits_lgb.py 等一致
SAMPLING_STRATEGIES = ("none", "undersampling", "oversampling", "hybrid")


def _infer_num_cat_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
            continue
        coerced = pd.to_numeric(s, errors="coerce")
        if float(coerced.notna().mean()) >= 0.85:
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def _coerce_numeric_inplace(X: pd.DataFrame, num_cols: list[str]) -> None:
    for c in num_cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")


def build_feature_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "ord",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        # 避免 num/cat 兩路 joblib 並行時尖峰記憶體（長跑 + 多折易碎片化）
        n_jobs=1,
    )


def _scale_pos_weight(y: np.ndarray) -> float:
    yv = np.asarray(y).ravel()
    u, c = np.unique(yv, return_counts=True)
    if len(u) != 2:
        return 1.0
    neg = int(c[u == 0][0]) if 0 in u else int(c[0])
    pos = int(c[u == 1][0]) if 1 in u else int(c[-1])
    return float(neg / max(pos, 1))


def _inner_tune_indices(
    y_train: np.ndarray,
    groups_train: np.ndarray | None,
    split_mode: str,
    *,
    random_state: int,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """外層訓練折內再切約 85%/15%，供 validation 上選超參數。"""
    n = len(y_train)
    yv = np.asarray(y_train).ravel()
    rs = int(random_state) + int(fold_id) * 17
    if split_mode == "stratified_group" and groups_train is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=rs)
        idx = np.arange(n)
        tr_i, va_i = next(gss.split(idx, yv, groups=groups_train))
        return tr_i, va_i
    idx = np.arange(n).reshape(-1, 1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=rs)
    tr_i, va_i = next(sss.split(idx, yv))
    return tr_i, va_i


def _lr_frame(arr: np.ndarray) -> pd.DataFrame:
    n = arr.shape[1]
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(n)])


def _dense_float_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """ColumnTransformer 輸出為 ndarray；採樣後為 DataFrame。統一轉成 float64 2D。"""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def _apply_sampling_safe(
    sampler: ImbalanceSampler,
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    strategy: str,
    logger,
) -> tuple[pd.DataFrame, np.ndarray]:
    """與 year-splits 一致：在數值化特徵上採樣；失敗時退回 none。"""
    yv = np.asarray(y_arr).ravel()
    if strategy == "none":
        return X_df.reset_index(drop=True), yv
    try:
        Xo, yo = sampler.apply_sampling(X_df.copy(), yv, strategy=strategy)
        if not isinstance(Xo, pd.DataFrame):
            Xo = pd.DataFrame(np.asarray(Xo), columns=list(X_df.columns))
        return Xo.reset_index(drop=True), np.asarray(yo).ravel()
    except Exception as e:
        logger.warning(f"sampling={strategy} 失敗，改用 none：{e}")
        return X_df.reset_index(drop=True), yv


def _make_models(spw: float, random_state: int) -> dict[str, object]:
    return {
        "lr": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=spw,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        ),
        "lgbm": LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=spw,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def run(
    *,
    csv_path: Path,
    n_splits: int,
    random_state: int,
    split_mode: str,
    drop_icd: bool,
    max_rows: int,
    smoke: bool,
    output_tag: str | None,
    tune: bool,
    tune_n_iter: int,
    sampling_strategies: tuple[str, ...] | list[str],
    logger,
) -> Path:
    set_seed(random_state)
    strategies = tuple(sampling_strategies)
    for s in strategies:
        if s not in SAMPLING_STRATEGIES:
            raise ValueError(f"不支援的 sampling={s}，請用 {SAMPLING_STRATEGIES}")
    X, y, groups = load_diabetic130_raw(csv_path)
    if drop_icd:
        for c in ("diag_1", "diag_2", "diag_3"):
            if c in X.columns:
                X = X.drop(columns=[c])
    if smoke:
        max_rows = min(max_rows or 8000, 8000)
    if max_rows > 0 and len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        groups = groups[idx]

    num_cols, cat_cols = _infer_num_cat_columns(X)
    _coerce_numeric_inplace(X, num_cols)
    logger.info(
        f"樣本數={len(X)}, 正類率={y.mean()*100:.2f}%, "
        f"數值欄={len(num_cols)}, 類別欄={len(cat_cols)}, split={split_mode}"
    )

    if split_mode == "stratified_group":
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(X, y, groups)
    elif split_mode == "stratified_row":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(X, y)
    else:
        raise ValueError(split_mode)

    fold_splits = list(split_iter)

    rows: list[dict] = []

    for strategy in strategies:
        sampler = ImbalanceSampler(random_state=random_state)
        for fold_id, (tr_idx, te_idx) in enumerate(fold_splits):
            X_train = X.iloc[tr_idx].reset_index(drop=True)
            y_train = y[tr_idx]
            X_te = X.iloc[te_idx].reset_index(drop=True)
            y_te = y[te_idx]
            g_train = groups[tr_idx]

            tune_seed = int(random_state) + fold_id * 7919 + (zlib.adler32(strategy.encode()) % 997)

            if tune:
                tr_i, va_i = _inner_tune_indices(
                    y_train,
                    g_train,
                    split_mode,
                    random_state=random_state,
                    fold_id=fold_id,
                )
                X_fit_raw = X_train.iloc[tr_i].reset_index(drop=True)
                X_val_tune_raw = X_train.iloc[va_i].reset_index(drop=True)
                y_fit = y_train[tr_i]
                y_val_tune = y_train[va_i]

                pre_tune = build_feature_preprocessor(num_cols, cat_cols)
                Xm_fit = pre_tune.fit_transform(X_fit_raw)
                Xm_val = pre_tune.transform(X_val_tune_raw)
                fn_tune = list(pre_tune.get_feature_names_out())
                X_fit_df = pd.DataFrame(Xm_fit, columns=fn_tune)
                X_val_df = pd.DataFrame(Xm_val, columns=fn_tune)

                X_fit_s, y_fit_s = _apply_sampling_safe(sampler, X_fit_df, y_fit, strategy, logger)
                sc_tune = StandardScaler()
                Xm_fit_lr = sc_tune.fit_transform(_dense_float_matrix(X_fit_s))
                Xm_val_lr = sc_tune.transform(_dense_float_matrix(Xm_val))

                cfg = get_config_loader()
                xgb_base = dict(cfg.get("model_config", "xgboost.base_params", {}))
                lgb_base = dict(cfg.get("model_config", "lightgbm.base_params", {}))
                lr_base = dict(class_weight="balanced", random_state=random_state, max_iter=5000)

                spw_tune = _scale_pos_weight(y_fit_s)
                best_lr, auc_lr = search_lr_on_val(
                    _lr_frame(Xm_fit_lr),
                    y_fit_s,
                    _lr_frame(Xm_val_lr),
                    y_val_tune,
                    lr_base,
                    n_iter=int(tune_n_iter),
                    seed=tune_seed,
                )
                best_rf, auc_rf = search_rf_on_val(
                    X_fit_s,
                    y_fit_s,
                    X_val_df,
                    y_val_tune,
                    n_iter=int(tune_n_iter),
                    seed=tune_seed,
                )
                best_xgb, auc_xgb = search_xgb_on_val(
                    X_fit_s,
                    y_fit_s,
                    X_val_df,
                    y_val_tune,
                    xgb_base,
                    spw_tune,
                    n_iter=int(tune_n_iter),
                    seed=tune_seed,
                )
                best_lgb, auc_lgb = search_lgb_on_val(
                    X_fit_s,
                    y_fit_s,
                    X_val_df,
                    y_val_tune,
                    lgb_base,
                    spw_tune,
                    n_iter=int(tune_n_iter),
                    seed=tune_seed,
                )

                del pre_tune, Xm_fit, Xm_val, X_fit_df, X_val_df, X_fit_s, sc_tune, Xm_fit_lr, Xm_val_lr
                gc.collect()

                pre = build_feature_preprocessor(num_cols, cat_cols)
                Xtr_full = pre.fit_transform(X_train)
                Xte_m = pre.transform(X_te)
                fn_full = list(pre.get_feature_names_out())
                Xtr_df = pd.DataFrame(Xtr_full, columns=fn_full)
                Xte_df = pd.DataFrame(Xte_m, columns=fn_full)

                sampler_final = ImbalanceSampler(random_state=random_state)
                Xtr_s, y_train_s = _apply_sampling_safe(sampler_final, Xtr_df, y_train, strategy, logger)
                spw_full = _scale_pos_weight(y_train_s)

                lr_scaler_final = StandardScaler()
                Xtr_lr = lr_scaler_final.fit_transform(_dense_float_matrix(Xtr_s))
                Xte_lr = lr_scaler_final.transform(np.asarray(Xte_m, dtype=np.float64))

                tune_specs: list[tuple[str, dict, float]] = [
                    ("lr", best_lr, auc_lr),
                    ("rf", best_rf, auc_rf),
                    ("xgb", best_xgb, auc_xgb),
                    ("lgbm", best_lgb, auc_lgb),
                ]
                for mname, best, val_auc in tune_specs:
                    try:
                        if mname == "lr":
                            params = {**lr_base, **(best or {})}
                            params["max_iter"] = max(int(params.get("max_iter", 5000)), 5000)
                            params.setdefault("solver", "lbfgs")
                            model = LogisticRegression(**params)
                            model.fit(Xtr_lr, y_train_s)
                            p_te = model.predict_proba(Xte_lr)[:, 1]
                        elif mname == "rf":
                            rf_params = {
                                **(best or {"n_estimators": 200}),
                                "class_weight": "balanced",
                                "random_state": random_state,
                                "n_jobs": -1,
                            }
                            model = RandomForestClassifier(**rf_params)
                            model.fit(Xtr_s, y_train_s)
                            p_te = model.predict_proba(Xte_df)[:, 1]
                        elif mname == "xgb":
                            cand = dict(best or {})
                            params = {
                                **xgb_base,
                                **cand,
                                "scale_pos_weight": float(spw_full),
                                "random_state": random_state,
                                "n_jobs": -1,
                                "eval_metric": "logloss",
                                "tree_method": "hist",
                            }
                            model = XGBClassifier(**params)
                            model.fit(Xtr_s, y_train_s, verbose=False)
                            p_te = model.predict_proba(Xte_df)[:, 1]
                        else:
                            cand = dict(best or {})
                            cand.pop("verbosity", None)
                            params = {
                                **lgb_base,
                                **cand,
                                "scale_pos_weight": float(spw_full),
                                "random_state": random_state,
                                "n_jobs": -1,
                                "verbose": -1,
                            }
                            model = LGBMClassifier(**params)
                            model.fit(Xtr_s, y_train_s)
                            p_te = model.predict_proba(Xte_df)[:, 1]
                        m = compute_metrics(y_te, p_te, threshold=0.5)
                        row = {
                            "sampling": strategy,
                            "fold": fold_id,
                            "model": mname,
                            "n_train": len(X_train),
                            "n_train_sampled": int(len(Xtr_s)),
                            "n_val_tune": int(len(va_i)),
                            "n_test": len(X_te),
                            "threshold": 0.5,
                        }
                        row.update(m)
                        row.update(tuning_meta(best or {}, float(val_auc)))
                        rows.append(row)
                        tauc = float(row.get("tune_val_auc", float("nan")))
                        tauc_s = "nan" if math.isnan(tauc) else f"{tauc:.4f}"
                        logger.info(
                            f"{strategy} fold={fold_id} {mname} AUC={row['AUC']:.4f} F1={row['F1']:.4f} "
                            f"n_test={len(X_te)} tune_val_auc={tauc_s}"
                        )
                    except Exception as e:
                        logger.error(f"fold={fold_id} model={mname} 失敗: {e}")
                        raise
                del pre, Xtr_full, Xte_m, Xtr_df, Xte_df, Xtr_s, lr_scaler_final
                gc.collect()
            else:
                pre = build_feature_preprocessor(num_cols, cat_cols)
                Xtr = pre.fit_transform(X_train)
                Xte_m = pre.transform(X_te)
                feat_names = list(pre.get_feature_names_out())
                Xtr_df = pd.DataFrame(Xtr, columns=feat_names)
                Xte_df = pd.DataFrame(Xte_m, columns=feat_names)
                Xtr_s, y_train_s = _apply_sampling_safe(sampler, Xtr_df, y_train, strategy, logger)
                spw_full = _scale_pos_weight(y_train_s)
                models = _make_models(spw_full, random_state)
                lr_scaler = StandardScaler()
                Xtr_lr = lr_scaler.fit_transform(_dense_float_matrix(Xtr_s))
                Xte_lr = lr_scaler.transform(np.asarray(Xte_m, dtype=np.float64))

                for mname, model in models.items():
                    try:
                        if mname == "lr":
                            model.fit(Xtr_lr, y_train_s)
                            p_te = model.predict_proba(Xte_lr)[:, 1]
                        else:
                            model.fit(Xtr_s, y_train_s)
                            p_te = model.predict_proba(Xte_df)[:, 1]
                        m = compute_metrics(y_te, p_te, threshold=0.5)
                        row = {
                            "sampling": strategy,
                            "fold": fold_id,
                            "model": mname,
                            "n_train": len(X_train),
                            "n_train_sampled": int(len(Xtr_s)),
                            "n_val_tune": 0,
                            "n_test": len(X_te),
                            "threshold": 0.5,
                        }
                        row.update(m)
                        row["tune_val_auc"] = float("nan")
                        row["tune_best_params"] = ""
                        rows.append(row)
                        logger.info(
                            f"{strategy} fold={fold_id} {mname} AUC={row['AUC']:.4f} "
                            f"F1={row['F1']:.4f} n_test={len(X_te)}"
                        )
                    except Exception as e:
                        logger.error(f"fold={fold_id} model={mname} 失敗: {e}")
                        raise
                del pre, Xtr, Xte_m, Xtr_df, Xte_df, Xtr_s, lr_scaler
                gc.collect()

    df_fold = pd.DataFrame(rows)
    tag = output_tag or f"kfold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = project_root / "results" / "phase1_baseline" / "medical_diabetic130_kfold" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    df_fold.to_csv(out_dir / "kfold_metrics_per_fold.csv", index=False)

    agg = df_fold.groupby(["sampling", "model"])[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    summary = agg.reset_index()
    summary.to_csv(out_dir / "kfold_summary_mean_std.csv", index=False)

    meta = {
        "csv_path": str(csv_path),
        "n_splits": n_splits,
        "random_state": random_state,
        "split_mode": split_mode,
        "cv_style": "stratified_kfold_val_auc_tune_refit_full_train_threshold_0.5"
        if tune
        else "stratified_kfold_default_hparams_threshold_0.5",
        "tune": bool(tune),
        "tune_n_iter": int(tune_n_iter),
        "drop_icd": drop_icd,
        "sampling_strategies": ",".join(strategies),
        "max_rows": max_rows,
        "smoke": smoke,
        "n_samples": int(len(X)),
        "positive_rate": float(y.mean()),
    }
    pd.Series(meta).to_csv(out_dir / "run_meta.csv", header=["value"])

    logger.info(f"已寫入 {out_dir}")
    return out_dir


def main() -> None:
    logger = get_logger("medical_diabetic130_kfold", console=True, file=False)
    p = argparse.ArgumentParser(description="Diabetes130 raw CSV — 5-fold CV baselines (LR/RF/XGB/LGBM)")
    p.add_argument("--csv", type=Path, default=DEFAULT_DIABETIC_CSV, help="diabetic_data.csv 路徑")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--split",
        choices=("stratified_group", "stratified_row"),
        default="stratified_row",
        help="stratified_row=單純列層級分層五折（預設）；stratified_group=依病人分組+分層五折",
    )
    p.add_argument("--drop-icd", action="store_true", help="略過 diag_1/2/3（較快、維度較低）")
    p.add_argument("--max-rows", type=int, default=0, help=">0 時隨機子抽樣（除錯用）")
    p.add_argument("--smoke", action="store_true", help="小規模快速跑（≈8000 列）")
    p.add_argument(
        "--no-tune",
        action="store_true",
        help="關閉超參數搜尋（否則每折在內層 validation 上以 AUC 隨機搜尋後，再於整段訓練折重 fit）",
    )
    p.add_argument("--tune-n-iter", type=int, default=48, help="每模型每折隨機搜尋組數（與 baseline_val_search 一致）")
    p.add_argument(
        "--sampling",
        nargs="+",
        choices=list(SAMPLING_STRATEGIES),
        default=list(SAMPLING_STRATEGIES),
        help="與 medical_year_splits_* 一致：none / undersampling / oversampling / hybrid（可多個；預設四種全跑）",
    )
    p.add_argument("--output-tag", type=str, default=None, help="結果子目錄名稱")
    args = p.parse_args()

    run(
        csv_path=args.csv,
        n_splits=args.n_splits,
        random_state=args.random_state,
        split_mode=args.split,
        drop_icd=args.drop_icd,
        max_rows=args.max_rows,
        smoke=args.smoke,
        output_tag=args.output_tag,
        tune=not args.no_tune,
        tune_n_iter=args.tune_n_iter,
        sampling_strategies=tuple(args.sampling),
        logger=logger,
    )


if __name__ == "__main__":
    main()
