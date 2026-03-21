"""
一鍵執行所有實驗（依研究階段順序）。

Phase 1  Baseline      : 基準方法 (Re-training / Fine-tuning)，全資料集
Phase 2  Ensemble      : XGB Old/New 年份切割（靜態／DES／DCS 分腳本；結果分 static、dynamic/des、dcs）
Phase 3  Feature       : 特徵選擇研究 & 掃描（Study II）
Phase 4  Analysis      : 比例研究、Split 比較、Base Learner、閾值分析
"""
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

# (相對於 experiments/ 的子路徑, 腳本名稱不含 .py, timeout 秒)
# 注意：dynamic 下的 des/、dcs/ 是第二層子目錄
EXPERIMENTS = [
    # ── Phase 1: Baseline ──────────────────────────────────────
    ("phase1_baseline",         "retrain",                  300),
    ("phase1_baseline",         "finetune",                 300),
    # ── Phase 2: XGB ensemble（static／dynamic/des／dynamic/dcs 分開；檔名 xgb_oldnew_*）──
    ("phase2_ensemble/static", "xgb_oldnew_bankruptcy_year_splits_static", 1200),
    ("phase2_ensemble/static", "xgb_oldnew_stock_year_splits_static", 1200),
    ("phase2_ensemble/static", "xgb_oldnew_medical_year_splits_static", 1200),
    ("phase2_ensemble/dynamic/des", "xgb_oldnew_bankruptcy_year_splits_des", 1800),
    ("phase2_ensemble/dynamic/des", "xgb_oldnew_stock_year_splits_des", 1800),
    ("phase2_ensemble/dynamic/des", "xgb_oldnew_medical_year_splits_des", 1800),
    ("phase2_ensemble/dynamic/dcs", "xgb_oldnew_bankruptcy_year_splits_dcs", 1800),
    ("phase2_ensemble/dynamic/dcs", "xgb_oldnew_stock_year_splits_dcs", 1800),
    ("phase2_ensemble/dynamic/dcs", "xgb_oldnew_medical_year_splits_dcs", 1800),
    # ── Phase 3: Feature Selection (Study II) ──────────────────
    ("phase3_feature",          "fs_study",                 300),
    ("phase3_feature",          "fs_sweep",                 2700),
    # ── Phase 4: Supplementary Analysis ────────────────────────
    ("phase4_analysis",         "split_comparison",         300),
    ("phase4_analysis",         "proportion_study",         600),
    ("phase4_analysis",         "base_learner_comparison",  600),
    ("phase4_analysis",         "stock_threshold_cost",     300),
]


def main():
    ok, fail, skip = 0, 0, 0
    for subdir, name, timeout in EXPERIMENTS:
        path = project_root / "experiments" / subdir / f"{name}.py"
        if not path.exists():
            print(f"[SKIP] {subdir}/{name}.py 不存在")
            skip += 1
            continue
        print(f"\n{'='*60}\n[{subdir}] {name}\n{'='*60}")
        r = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(project_root),
            timeout=timeout,
        )
        if r.returncode != 0:
            print(f"[FAIL] {subdir}/{name} 結束碼 {r.returncode}")
            fail += 1
        else:
            print(f"[OK]   {subdir}/{name}")
            ok += 1
    print(f"\n=== 完成：OK={ok}  FAIL={fail}  SKIP={skip} ===")
    print("可再執行: python scripts\\analysis\\compare_all_results.py")


if __name__ == "__main__":
    main()
