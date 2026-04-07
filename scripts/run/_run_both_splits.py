"""
scripts/run/_run_both_splits.py
================================
完整跑所有資料集 × 兩種切割方式 (chronological / block_cv)：
  - P1: Retrain
  - P2: Named Ensemble Combos（LightGBM ModelPool，舊版對照）
  - P3: DCS Comparison + DES Advanced（同上）

論文主線 Phase2 集成（XGB Old/New 年份切割）請用 experiments/phase2_ensemble 內
`static/xgb_oldnew_*_static.py`、`dynamic/des/xgb_oldnew_*_des.py`、`dynamic/dcs/xgb_oldnew_*_dcs.py`，輸出於
results/phase2_ensemble/static/、…/dynamic/des/、…/dynamic/dcs/。

輸出檔名均附帶 split_mode suffix，例如：
  results/phase1_baseline/bankruptcy_retrain_chronological.csv
  results/phase2_ensemble/medical_ensemble_results_block_cv.csv
  results/phase2_ensemble/dynamic/stock_spx_dcs_comparison_block_cv.csv
  
使用方式：
    python scripts/run/_run_both_splits.py
    python scripts/run/_run_both_splits.py --datasets bankruptcy medical
    python scripts/run/_run_both_splits.py --splits chronological
    python scripts/run/_run_both_splits.py --phases p1 p3
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from src.evaluation import compute_metrics, results_to_dataframe
from experiments._shared.common_bankruptcy import get_bankruptcy_splits
from experiments._shared.common_dataset import get_splits
from experiments._shared.common_dcs import run_dcs_all_variants
from experiments._shared.common_des import run_des
from experiments._shared.common_des_advanced import run_des_advanced

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

ALL_DATASETS = ["bankruptcy", "medical", "stock_spx", "stock_dji", "stock_ndx"]
ALL_SPLITS   = ["chronological", "block_cv"]

OUT_P1 = project_root / "results" / "phase1_baseline"
OUT_P2 = project_root / "results" / "phase2_ensemble"
OUT_P3 = project_root / "results" / "phase2_ensemble" / "dynamic"

# Confirmed via reverse-engineering from existing results (2025-03):
NAMED_COMBOS = {
    "ensemble_old_3":               ["old_under", "old_over", "old_hybrid"],
    "ensemble_new_3":               ["new_under", "new_over", "new_hybrid"],
    "ensemble_all_6":               ["old_under", "old_over", "old_hybrid",
                                     "new_under", "new_over", "new_hybrid"],
    "ensemble_2_old_hybrid_new_hybrid": ["old_hybrid", "new_hybrid"],
    "ensemble_3_type_a":            ["old_under", "old_over", "new_hybrid"],
    "ensemble_3_type_b":            ["old_hybrid", "new_over", "new_hybrid"],
    "ensemble_4":                   ["old_under", "old_over", "new_over", "new_hybrid"],
    "ensemble_5":                   ["old_under", "old_over", "old_hybrid",
                                     "new_over", "new_hybrid"],
}

# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────

def get_data(ds_name: str, logger, split_mode: str):
    """統一取得切割後資料 (Bankruptcy 或其他)。"""
    if ds_name == "bankruptcy":
        return get_bankruptcy_splits(logger, split_mode=split_mode, dataset="us_1999_2018")
    else:
        return get_splits(ds_name, logger, split_mode=split_mode)


# ─────────────────────────────────────────────────────────────────────────
# P1: RETRAIN
# ─────────────────────────────────────────────────────────────────────────

def run_p1(ds_name: str, split_mode: str, logger):
    """執行 P1 Retrain，各 3 種採樣策略，儲存帶 split_mode 的 CSV。"""
    X_h, y_h, X_n, y_n, X_t, y_t = get_data(ds_name, logger, split_mode)
    yt = np.asarray(y_t.values if hasattr(y_t, "values") else y_t)
    yh = y_h.values if hasattr(y_h, "values") else y_h
    yn = y_n.values if hasattr(y_n, "values") else y_n

    sm = ImbalanceSampler()
    X_c = pd.concat([X_h, X_n])
    y_c = np.concatenate([yh, yn])

    retrain_res  = {}

    for strat in ["none", "undersampling", "hybrid"]:
        # Retrain
        Xr, yr = sm.apply_sampling(X_c, y_c, strategy=strat)
        m_re = LightGBMWrapper(name=f"retrain_{strat}"); m_re.fit(Xr, yr)
        retrain_res[f"retrain_{strat}"] = compute_metrics(yt, m_re.predict_proba(X_t))
        _auc = retrain_res[f"retrain_{strat}"]["AUC"]
        logger.info(f"  [P1 Retrain {strat}] AUC={_auc:.4f}")

    OUT_P1.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(retrain_res).T.to_csv(
        OUT_P1 / f"{ds_name}_retrain_{split_mode}.csv")
    logger.info(f"  P1 saved: {ds_name} / {split_mode}")


# ─────────────────────────────────────────────────────────────────────────
# P2: NAMED ENSEMBLE COMBOS
# ─────────────────────────────────────────────────────────────────────────

def run_p2(ds_name: str, split_mode: str, logger):
    """執行 P2 Named Ensemble，儲存帶 split_mode 的 CSV。"""
    X_h, y_h, X_n, y_n, X_t, y_t = get_data(ds_name, logger, split_mode)
    yt = np.asarray(y_t.values if hasattr(y_t, "values") else y_t)
    yh = y_h.values if hasattr(y_h, "values") else y_h
    yn = y_n.values if hasattr(y_n, "values") else y_n

    old_pool = ModelPool(pool_name="old"); old_pool.create_pool(X_h, yh, prefix="old")
    new_pool = ModelPool(pool_name="new"); new_pool.create_pool(X_n, yn, prefix="new")
    ap = {**old_pool.predict_proba(X_t), **new_pool.predict_proba(X_t)}

    results = {}
    for name, keys in NAMED_COMBOS.items():
        proba = np.mean([ap[k] for k in keys if k in ap], axis=0)
        results[name] = compute_metrics(yt, proba)
        logger.info(f"  [{name}] AUC={results[name]['AUC']:.4f}")

    OUT_P2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).T.to_csv(
        OUT_P2 / f"{ds_name}_ensemble_results_{split_mode}.csv")
    logger.info(f"  P2 saved: {ds_name} / {split_mode}")


# ─────────────────────────────────────────────────────────────────────────
# P3: DCS COMPARISON
# ─────────────────────────────────────────────────────────────────────────

def run_p3_dcs(ds_name: str, split_mode: str, logger):
    """執行 P3 DCS，儲存帶 split_mode 的 CSV。"""
    X_h, y_h, X_n, y_n, X_t, y_t = get_data(ds_name, logger, split_mode)

    results = run_dcs_all_variants(X_h, y_h, X_n, y_n, X_t, y_t, logger, k=7)
    for name, m in results.items():
        logger.info(f"  [{name}] AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

    OUT_P3.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(results).to_csv(
        OUT_P3 / f"{ds_name}_dcs_comparison_{split_mode}.csv")
    logger.info(f"  P3-DCS saved: {ds_name} / {split_mode}")


# ─────────────────────────────────────────────────────────────────────────
# P3: DES ADVANCED
# ─────────────────────────────────────────────────────────────────────────

def run_p3_des(ds_name: str, split_mode: str, logger):
    """執行 P3 DES Advanced，儲存帶 split_mode 的 CSV。"""
    X_h, y_h, X_n, y_n, X_t, y_t = get_data(ds_name, logger, split_mode)

    results = {
        "DES_baseline":
            run_des(X_h, y_h, X_n, y_n, X_t, y_t, logger, k=7),
        "DES_time_weighted":
            run_des_advanced(X_h, y_h, X_n, y_n, X_t, y_t, logger,
                             k=7, time_weight_new=2.0, minority_weight=1.0),
        "DES_minority_weighted":
            run_des_advanced(X_h, y_h, X_n, y_n, X_t, y_t, logger,
                             k=7, time_weight_new=1.0, minority_weight=2.0),
        "DES_combined":
            run_des_advanced(X_h, y_h, X_n, y_n, X_t, y_t, logger,
                             k=7, time_weight_new=2.0, minority_weight=2.0),
    }
    for name, m in results.items():
        logger.info(f"  [{name}] AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

    OUT_P3.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(results).to_csv(
        OUT_P3 / f"{ds_name}_des_advanced_{split_mode}.csv")
    logger.info(f"  P3-DES saved: {ds_name} / {split_mode}")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

PHASE_MAP = {
    "p1": run_p1,
    "p2": run_p2,
    "p3_dcs": run_p3_dcs,
    "p3_des": run_p3_des,
}


def main():
    parser = argparse.ArgumentParser(description="Run all experiments for both split modes")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        choices=ALL_DATASETS, help="Which datasets to process")
    parser.add_argument("--splits", nargs="+", default=ALL_SPLITS,
                        choices=ALL_SPLITS, help="Which split modes to run")
    parser.add_argument("--phases", nargs="+", default=list(PHASE_MAP.keys()),
                        choices=list(PHASE_MAP.keys()), help="Which phases to run")
    args = parser.parse_args()

    logger = get_logger("RunBothSplits", console=True, file=True)
    set_seed(42)

    total = len(args.datasets) * len(args.splits) * len(args.phases)
    done  = 0
    errors = []

    logger.info("=" * 70)
    logger.info(f"Run Both Splits: {len(args.datasets)} datasets x {len(args.splits)} splits x {len(args.phases)} phases = {total} jobs")
    logger.info(f"Datasets : {args.datasets}")
    logger.info(f"Splits   : {args.splits}")
    logger.info(f"Phases   : {args.phases}")
    logger.info("=" * 70)

    for ds in args.datasets:
        for split in args.splits:
            for phase in args.phases:
                tag = f"{ds} / {split} / {phase}"
                logger.info(f"\n>>> {tag}")
                try:
                    fn = PHASE_MAP[phase]
                    fn(ds, split, logger)
                    done += 1
                    logger.info(f"    OK  ({done}/{total})")
                except Exception as exc:
                    logger.error(f"    FAILED {tag}: {exc}")
                    errors.append((tag, str(exc)))

    logger.info("\n" + "=" * 70)
    logger.info(f"完成 {done}/{total} 個任務")
    if errors:
        logger.info(f"失敗 {len(errors)} 個：")
        for tag, err in errors:
            logger.info(f"  {tag}: {err}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

