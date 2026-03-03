"""
一鍵執行所有實驗（依研究階段順序）。

Phase 1  Baseline      : 基準方法 (Re-training / Fine-tuning)，全資料集
Phase 2  Ensemble      : 靜態集成 (Under / Over / Hybrid / 全組合)
Phase 3  Dynamic       : DES standard / advanced、DCS comparison
Phase 4  Feature       : 特徵選擇研究 & 掃描
Phase 5  Analysis      : 比例研究、Split 比較、Base Learner、閾值分析
"""
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent

# (相對於 experiments/ 的子路徑, 腳本名稱不含 .py, timeout 秒)
# 注意：phase3 下的 des/ dcs/ 是第二層子目錄
EXPERIMENTS = [
    # ── Phase 1: Baseline ──────────────────────────────────────
    ("phase1_baseline",         "retrain",                  300),
    ("phase1_baseline",         "finetune",                 300),
    # ── Phase 2: Static Ensemble ───────────────────────────────
    ("phase2_ensemble",         "undersampling",            300),
    ("phase2_ensemble",         "oversampling",             300),
    ("phase2_ensemble",         "hybrid",                   300),
    ("phase2_ensemble",         "all_combinations",         600),
    # ── Phase 3: Dynamic Selection ─────────────────────────────
    ("phase3_dynamic/des",      "standard",                 300),
    ("phase3_dynamic/des",      "advanced",                 600),
    ("phase3_dynamic/dcs",      "comparison",               600),
    # ── Phase 4: Feature Selection (Study II) ──────────────────
    ("phase4_feature",          "fs_study",                 300),
    ("phase4_feature",          "fs_sweep",                 2700),
    # ── Phase 5: Supplementary Analysis ────────────────────────
    ("phase5_analysis",         "split_comparison",         300),
    ("phase5_analysis",         "proportion_study",         600),
    ("phase5_analysis",         "base_learner_comparison",  600),
    ("phase5_analysis",         "stock_threshold_cost",     300),
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
    print("可再執行: python scripts\\compare_all_results.py")


if __name__ == "__main__":
    main()
