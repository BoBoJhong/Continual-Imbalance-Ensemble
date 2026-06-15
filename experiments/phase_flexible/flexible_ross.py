"""
Flexible ROSS + DAWCE Framework
================================
這是 ROSS/DAWCE 的通用版本，時間切割粒度不限於「年」。

支援的 granularity：
    'year'          → 以整數年份切（例如 fyear 欄位）
    'quarter'       → 以季度切（Q1/Q2/Q3/Q4）
    'month'         → 以月份切
    'sample'        → 以樣本數量切（每隔 N 筆為一個單位）

與現有 phase4/phase5 程式碼完全獨立，不影響既有實驗。

使用方式：
    from experiments.phase_flexible.flexible_ross import FlexibleROSS

    ross = FlexibleROSS(
        time_col="fyear",
        granularity="year",
        min_old_units=3,
        min_new_units=1,
    )
    results = ross.run(
        df_train=df_train,
        target_col="bankruptcy",
        df_val=df_val,
        df_test=df_test,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_logger

# ── 常數 ────────────────────────────────────────────────────────────────────
POOL_STRATEGIES = ["undersampling", "oversampling", "hybrid"]
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


# ── 時間單位工具 ──────────────────────────────────────────────────────────────

def _extract_time_units(
    df: pd.DataFrame,
    time_col: str,
    granularity: Literal["year", "quarter", "month", "sample"],
    sample_step: int = 500,
) -> list:
    """
    從 DataFrame 的 time_col 提取排序好的時間單位清單。

    granularity='sample' 時，sample_step 決定每幾筆為一個單位。
    回傳的值可直接用來做 mask（year/quarter/month 回傳對應的 period label；
    sample 回傳 (start_idx, end_idx) tuple）。
    """
    if granularity == "year":
        return sorted(df[time_col].astype(int).unique().tolist())

    elif granularity == "quarter":
        periods = pd.PeriodIndex(pd.to_datetime(df[time_col]), freq="Q")
        return sorted(periods.unique().tolist(), key=lambda p: p.ordinal)

    elif granularity == "month":
        periods = pd.PeriodIndex(pd.to_datetime(df[time_col]), freq="M")
        return sorted(periods.unique().tolist(), key=lambda p: p.ordinal)

    elif granularity == "sample":
        n = len(df)
        boundaries = list(range(0, n, sample_step))
        if boundaries[-1] != n:
            boundaries.append(n)
        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    else:
        raise ValueError(f"Unknown granularity: {granularity!r}. "
                         f"Choose from 'year', 'quarter', 'month', 'sample'.")


def _split_by_unit(
    df: pd.DataFrame,
    time_col: str,
    granularity: str,
    old_units: list,
    new_units: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    依照 old_units / new_units 切割 DataFrame，回傳 (df_old, df_new)。
    """
    if granularity == "year":
        old_mask = df[time_col].astype(int).isin(old_units)
        new_mask = df[time_col].astype(int).isin(new_units)
        return df.loc[old_mask].reset_index(drop=True), df.loc[new_mask].reset_index(drop=True)

    elif granularity in ("quarter", "month"):
        freq = "Q" if granularity == "quarter" else "M"
        col_period = pd.PeriodIndex(pd.to_datetime(df[time_col]), freq=freq)
        old_mask = col_period.isin(old_units)
        new_mask = col_period.isin(new_units)
        return df.loc[old_mask].reset_index(drop=True), df.loc[new_mask].reset_index(drop=True)

    elif granularity == "sample":
        old_start = old_units[0][0]
        old_end   = old_units[-1][1]
        new_start = new_units[0][0]
        new_end   = new_units[-1][1]
        return df.iloc[old_start:old_end].reset_index(drop=True), df.iloc[new_start:new_end].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown granularity: {granularity!r}")


# ── 核心模組 ──────────────────────────────────────────────────────────────────

class FlexibleROSS:
    """
    通用 ROSS + DAWCE 框架。

    Parameters
    ----------
    time_col : str
        資料中代表時間的欄位名稱（整數年份、datetime、或任何可排序的欄位）。
    granularity : str
        時間切割粒度：'year' / 'quarter' / 'month' / 'sample'。
    min_old_units : int
        Old period 最少需要幾個時間單位（防止 Old 資料太少）。
    min_new_units : int
        New period 最少需要幾個時間單位（防止 New 資料太少）。
    boundary_metric : str
        ROSS 邊界選擇依據：'AUC'（預設）或 'F1'。
    weight_metric : str
        加權搜尋依據：'F1'（預設）或 'AUC'。
    sample_step : int
        granularity='sample' 時，每幾筆為一個時間單位。
    random_state : int
        隨機種子。
    verbose : bool
        是否顯示搜尋過程 log。
    """

    def __init__(
        self,
        time_col: str,
        granularity: Literal["year", "quarter", "month", "sample"] = "year",
        min_old_units: int = 3,
        min_new_units: int = 1,
        boundary_metric: Literal["AUC", "F1"] = "AUC",
        weight_metric: Literal["F1", "AUC"] = "F1",
        sample_step: int = 500,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.time_col = time_col
        self.granularity = granularity
        self.min_old_units = min_old_units
        self.min_new_units = min_new_units
        self.boundary_metric = boundary_metric
        self.weight_metric = weight_metric
        self.sample_step = sample_step
        self.random_state = random_state
        self.logger = get_logger("FlexibleROSS", console=verbose, file=False)

        # 結果儲存
        self.candidates_: pd.DataFrame | None = None
        self.best_boundary_idx_: int | None = None
        self.best_w_new_: float | None = None
        self.weight_sweep_: pd.DataFrame | None = None

    # ── 內部工具 ──────────────────────────────────────────────────────────────

    def _preprocess(
        self,
        df_old: pd.DataFrame,
        df_new: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame | None,
        feature_cols: list[str],
    ) -> tuple:
        """StandardScaler fitted on Old only，套用至所有集合。"""
        scaler = StandardScaler()
        X_old = scaler.fit_transform(df_old[feature_cols])
        X_new = scaler.transform(df_new[feature_cols])
        X_val = scaler.transform(df_val[feature_cols])
        X_test_out = scaler.transform(df_test[feature_cols]) if df_test is not None else None
        return (
            pd.DataFrame(X_old, columns=feature_cols),
            pd.DataFrame(X_new, columns=feature_cols),
            pd.DataFrame(X_val, columns=feature_cols),
            pd.DataFrame(X_test_out, columns=feature_cols) if X_test_out is not None else None,
        )

    def _train_pool(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        tag: str,
    ) -> tuple[list[XGBoostWrapper], list[np.ndarray]]:
        """用三種取樣策略訓練模型池，回傳 (models, val_probas_placeholder)。"""
        sampler = ImbalanceSampler(random_state=self.random_state)
        models = []
        for strategy in POOL_STRATEGIES:
            try:
                X_r, y_r = sampler.apply_sampling(X, y, strategy=strategy)
            except Exception:
                X_r, y_r = X, y
            m = XGBoostWrapper(name=f"{tag}_{strategy}", use_imbalance=False)
            m.fit(X_r, y_r)
            models.append(m)
        return models

    def _pool_mean_proba(
        self,
        models: list[XGBoostWrapper],
        X: pd.DataFrame,
    ) -> np.ndarray:
        """回傳模型池的平均機率。"""
        return np.mean([m.predict_proba(X) for m in models], axis=0)

    def _best_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """以 F1 最大化在驗證集選最佳 threshold。"""
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    # ── 主流程 ────────────────────────────────────────────────────────────────

    def run(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
    ) -> dict:
        """
        執行完整 ROSS 邊界搜尋 + DAWCE 加權搜尋。

        Parameters
        ----------
        df_train : DataFrame
            訓練資料（含 time_col 與 target_col）。
        target_col : str
            目標欄位名稱（0/1 二元標籤）。
        df_val : DataFrame
            驗證資料（用於邊界選擇與加權搜尋）。
        df_test : DataFrame, optional
            測試資料（不參與任何選擇，最後評估用）。

        Returns
        -------
        dict
            {
              'candidates': DataFrame,        ← 每個候選邊界的驗證成績
              'best_boundary_idx': int,       ← 選出的邊界索引
              'best_boundary_label': any,     ← 選出的邊界時間標籤
              'best_w_new': float,            ← 最佳 w_new
              'weight_sweep': DataFrame,      ← 各 w_new 的測試成績
              'test_metrics': dict,           ← 最終測試集成績（若提供 df_test）
            }
        """
        feature_cols = [c for c in df_train.columns if c not in [self.time_col, target_col]]
        y_val = np.asarray(df_val[target_col])
        y_test = np.asarray(df_test[target_col]) if df_test is not None else None

        # Step 1：生成時間單位清單
        all_units = _extract_time_units(
            df_train, self.time_col, self.granularity, self.sample_step
        )
        n = len(all_units)
        self.logger.info(
            f"[ROSS] granularity={self.granularity!r}, "
            f"total units={n}, "
            f"min_old={self.min_old_units}, min_new={self.min_new_units}"
        )

        if n < self.min_old_units + self.min_new_units:
            raise ValueError(
                f"只有 {n} 個時間單位，但 min_old+min_new={self.min_old_units + self.min_new_units}。"
                f"請減少 min_old_units / min_new_units，或提供更多資料。"
            )

        # Step 2：枚舉所有合法邊界（split_at = New 的第一個時間單位索引）
        candidate_rows = []
        for split_at in range(self.min_old_units, n - self.min_new_units + 1):
            old_units = all_units[:split_at]
            new_units = all_units[split_at:]

            df_old, df_new = _split_by_unit(
                df_train, self.time_col, self.granularity, old_units, new_units
            )
            y_old = np.asarray(df_old[target_col])
            y_new = np.asarray(df_new[target_col])

            if len(y_old) < 50 or len(y_new) < 50:
                self.logger.warning(
                    f"  skip split_at={split_at}: old={len(y_old)}, new={len(y_new)} (太少樣本)"
                )
                continue

            X_old_s, X_new_s, X_val_s, _ = self._preprocess(
                df_old, df_new, df_val, None, feature_cols
            )

            old_models = self._train_pool(X_old_s, y_old, tag=f"old_{split_at}")
            new_models = self._train_pool(X_new_s, y_new, tag=f"new_{split_at}")

            old_val_proba = self._pool_mean_proba(old_models, X_val_s)
            new_val_proba = self._pool_mean_proba(new_models, X_val_s)

            old_val_auc = roc_auc_score(y_val, old_val_proba)
            new_val_auc = roc_auc_score(y_val, new_val_proba)
            old_val_f1  = f1_score(y_val, (old_val_proba >= 0.5).astype(int), zero_division=0)
            new_val_f1  = f1_score(y_val, (new_val_proba >= 0.5).astype(int), zero_division=0)

            boundary_label = all_units[split_at]
            self.logger.info(
                f"  split_at={split_at} boundary={boundary_label} | "
                f"Old val AUC={old_val_auc:.4f} F1={old_val_f1:.4f} | "
                f"New val AUC={new_val_auc:.4f} F1={new_val_f1:.4f}"
            )

            candidate_rows.append({
                "split_at": split_at,
                "boundary_label": str(boundary_label),
                "old_n": len(y_old),
                "new_n": len(y_new),
                "old_val_AUC": old_val_auc,
                "new_val_AUC": new_val_auc,
                "old_val_F1": old_val_f1,
                "new_val_F1": new_val_f1,
                "gap_AUC": new_val_auc - old_val_auc,
                "gap_F1": new_val_f1 - old_val_f1,
                # 暫存模型供後續加權搜尋用
                "_old_models": old_models,
                "_new_models": new_models,
            })

        if not candidate_rows:
            raise RuntimeError("沒有任何合法候選邊界，請檢查資料與參數設定。")

        candidates = pd.DataFrame(candidate_rows)
        candidates = candidates.sort_values(
            f"new_val_{self.boundary_metric}", ascending=False
        ).reset_index(drop=True)

        self.candidates_ = candidates.drop(columns=["_old_models", "_new_models"])
        best_row = candidates.iloc[0]
        self.best_boundary_idx_ = int(best_row["split_at"])

        self.logger.info(
            f"\n[ROSS] 選出最佳邊界: split_at={self.best_boundary_idx_} "
            f"label={best_row['boundary_label']} "
            f"(new_val_{self.boundary_metric}={best_row[f'new_val_{self.boundary_metric}']:.4f})"
        )

        # Step 3：用選出邊界重訓完整模型池（含 val 資料）
        old_units_best = all_units[: self.best_boundary_idx_]
        new_units_best = all_units[self.best_boundary_idx_ :]
        df_old_best, df_new_best = _split_by_unit(
            df_train, self.time_col, self.granularity, old_units_best, new_units_best
        )
        y_old_best = np.asarray(df_old_best[target_col])
        y_new_best = np.asarray(df_new_best[target_col])

        X_old_f, X_new_f, X_val_f, X_test_f = self._preprocess(
            df_old_best, df_new_best, df_val, df_test, feature_cols
        )

        old_models_f = self._train_pool(X_old_f, y_old_best, tag="final_old")
        new_models_f = self._train_pool(X_new_f, y_new_best, tag="final_new")

        old_val_p  = self._pool_mean_proba(old_models_f, X_val_f)
        new_val_p  = self._pool_mean_proba(new_models_f, X_val_f)
        old_test_p = self._pool_mean_proba(old_models_f, X_test_f) if X_test_f is not None else None
        new_test_p = self._pool_mean_proba(new_models_f, X_test_f) if X_test_f is not None else None

        # Step 4：網格搜尋最佳 w_new（在驗證集上）
        weight_rows = []
        best_w, best_score = 0.5, -1.0

        for w_new in WEIGHT_GRID:
            w_old = float(1.0 - w_new)
            val_proba = w_old * old_val_p + float(w_new) * new_val_p
            threshold = self._best_threshold(y_val, val_proba)
            val_score = (
                roc_auc_score(y_val, val_proba)
                if self.weight_metric == "AUC"
                else f1_score(y_val, (val_proba >= threshold).astype(int), zero_division=0)
            )

            row: dict = {
                "w_old": w_old,
                "w_new": float(w_new),
                "val_score": val_score,
                "threshold": threshold,
            }

            # 若有測試集，順便記錄測試成績
            if old_test_p is not None:
                test_proba = w_old * old_test_p + float(w_new) * new_test_p
                test_metrics = compute_metrics(y_test, test_proba, threshold=threshold)
                row.update({f"test_{k}": v for k, v in test_metrics.items()})

            weight_rows.append(row)

            if val_score > best_score:
                best_score = val_score
                best_w = float(w_new)

        self.best_w_new_ = best_w
        self.weight_sweep_ = pd.DataFrame(weight_rows)

        self.logger.info(
            f"[DAWCE] 最佳 w_new={best_w:.2f} "
            f"(val_{self.weight_metric}={best_score:.4f})"
        )

        # Step 5：整理最終結果
        test_metrics_out = {}
        if old_test_p is not None:
            best_threshold = weight_rows[
                [r["w_new"] for r in weight_rows].index(best_w)
            ]["threshold"]
            test_proba_best = (1 - best_w) * old_test_p + best_w * new_test_p
            test_metrics_out = compute_metrics(y_test, test_proba_best, threshold=best_threshold)
            self.logger.info(
                f"[Test] AUC={test_metrics_out['AUC']:.4f} "
                f"F1={test_metrics_out['F1']:.4f} "
                f"Recall={test_metrics_out['Recall']:.4f} "
                f"Precision={test_metrics_out['Precision']:.4f}"
            )

        return {
            "candidates": self.candidates_,
            "best_boundary_idx": self.best_boundary_idx_,
            "best_boundary_label": best_row["boundary_label"],
            "best_w_new": self.best_w_new_,
            "weight_sweep": self.weight_sweep_,
            "test_metrics": test_metrics_out,
        }
