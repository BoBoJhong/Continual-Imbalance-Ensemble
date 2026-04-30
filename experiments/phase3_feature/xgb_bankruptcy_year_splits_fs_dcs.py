"""
Phase 3 — Study II: 特徵選擇 + DCS 動態選擇器（破產資料集）
=============================================================================
Pipeline 與 Phase 2 DCS 腳本完全一致（單一 scaler, Old+New concat），
差異僅在 scale 前插入 FeatureSelector。

FS 配置與靜態 / DES 實驗完全對齊（5 種），確保跨方法可比較性：
  no_fs / mutual_info_r50 / mutual_info_r80 / shap_r50 / shap_r80

輸出：results/phase3_feature/xgb_bankruptcy_fs_full_dcs.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import DataPreprocessor, ImbalanceSampler
from src.models import XGBoostWrapper
from src.features import FeatureSelector
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split
from experiments._shared.common_dcs import _Wrapper, run_dcs_all_variants_from_pool

# ---------------------------------------------------------------------------
# 常數（與 DES FS 腳本完全對齊）
# ---------------------------------------------------------------------------

SAMPLING_STRATEGIES = ["undersampling", "oversampling", "hybrid"]

FS_CONFIGS = [
    ("no_fs",           None,          None),
    ("mutual_info_r50", "mutual_info", 0.5),
    ("mutual_info_r80", "mutual_info", 0.8),
    ("shap_r50",        "shap",        0.5),
    ("shap_r80",        "shap",        0.8),
]

METRICS = ["AUC", "F1", "Recall"]
OUTPUT_DIR = project_root / "results" / "phase3_feature"


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _apply_fs(fs_tag, fs_method, fs_ratio, X_old, y_old, X_new, X_test, logger):
    """在 X_old 上 fit FeatureSelector，transform 全部三個分割。"""
    if fs_method is None:
        return X_old, X_new, X_test
    k = max(1, int(X_old.shape[1] * fs_ratio))
    fs = FeatureSelector(method=fs_method, k=k)
    X_old_fs = fs.fit_transform(X_old, y_old)
    X_new_fs  = fs.transform(X_new)
    X_test_fs = fs.transform(X_test)
    logger.info(f"    FS [{fs_tag}]: {X_old.shape[1]} → {X_old_fs.shape[1]} features")
    return X_old_fs, X_new_fs, X_test_fs


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main():
    logger = get_logger("Phase3_FS_DCS_Bankruptcy", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for split_idx, (label, old_end_year) in enumerate(YEAR_SPLITS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{split_idx}/{len(YEAR_SPLITS)}] {label}  (old_end={old_end_year})")
        logger.info("=" * 60)

        # 載入原始資料（不含年份 array，DCS 不需要）
        X_old_raw, y_old, X_new_raw, y_new, X_test_raw, y_test = \
            get_bankruptcy_year_split(logger, old_end_year=old_end_year)

        y_old     = np.asarray(y_old)
        y_new     = np.asarray(y_new)
        y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)

        for fs_tag, fs_method, fs_ratio in FS_CONFIGS:
            logger.info(f"  FS: {fs_tag}")

            # 1. Feature Selection（在 Old 上 fit）
            X_old_fs, X_new_fs, X_test_fs = _apply_fs(
                fs_tag, fs_method, fs_ratio,
                X_old_raw, y_old, X_new_raw, X_test_raw, logger
            )

            # 2. 單一 scaler（Old + New concat fit，與 Phase 2 DCS 完全一致）
            pre = DataPreprocessor()
            X_all = pd.concat([X_old_fs, X_new_fs], axis=0).reset_index(drop=True)
            X_all_s = pre.scale_features(X_all, fit=True)[0]
            n_old = len(X_old_fs)
            X_h = X_all_s.iloc[:n_old]   # Old scaled
            X_n = X_all_s.iloc[n_old:]   # New scaled
            X_t = pre.scale_features(X_test_fs, fit=False)[0]  # Test scaled

            # 3. 訓練 6 個 XGBoost（Old×3 + New×3，與 DCS Phase 2 相同）
            sampler = ImbalanceSampler()
            pool_models = []
            for s in SAMPLING_STRATEGIES:
                # Old model
                X_r, y_r = sampler.apply_sampling(X_h, y_old, strategy=s)
                m_old = XGBoostWrapper(name=f"old_{label}_{fs_tag}_{s}")
                m_old.fit(X_r, y_r)
                pool_models.append(_Wrapper(m_old))
            for s in SAMPLING_STRATEGIES:
                # New model
                X_r, y_r = sampler.apply_sampling(X_n, y_new, strategy=s)
                m_new = XGBoostWrapper(name=f"new_{label}_{fs_tag}_{s}")
                m_new.fit(X_r, y_r)
                pool_models.append(_Wrapper(m_new))

            # 4. 執行 DCS 全變體（OLA / LCA / OLA_TW / LCA_TW）
            results = run_dcs_all_variants_from_pool(
                pool_models,
                X_h, y_old,
                X_n, y_new,
                X_t, y_test_arr,
                logger, k=7,
            )

            for variant_name, metrics in results.items():
                all_rows.append({
                    "split":    label,
                    "fs":       fs_tag,
                    "ensemble": variant_name,
                    **{k: metrics[k] for k in METRICS if k in metrics},
                })

    # 儲存結果
    df = pd.DataFrame(all_rows)
    out_path = OUTPUT_DIR / "xgb_bankruptcy_fs_full_dcs.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    logger.info(f"\nDCS results saved: {out_path}  ({len(df)} rows)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
