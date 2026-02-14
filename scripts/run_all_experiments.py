"""
一鍵執行所有實驗（01~08）。
依序：Bankruptcy baseline -> ensemble -> DES -> Study II -> Stock -> Medical -> Stock DES -> Medical DES
"""
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
experiments = [
    "01_bankruptcy_baseline",
    "02_bankruptcy_ensemble",
    "03_bankruptcy_des",
    "04_bankruptcy_feature_selection_study",
    "05_stock_baseline_ensemble",
    "06_medical_baseline_ensemble",
    "07_stock_des",
    "08_medical_des",
]


def main():
    for name in experiments:
        path = project_root / "experiments" / f"{name}.py"
        if not path.exists():
            print(f"[SKIP] {path} 不存在")
            continue
        print(f"\n{'='*60}\n執行: python experiments\\{name}.py\n{'='*60}")
        r = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(project_root),
            timeout=300,
        )
        if r.returncode != 0:
            print(f"[FAIL] {name} 結束碼 {r.returncode}")
        else:
            print(f"[OK] {name}")
    print("\n全部執行完畢。可再執行: python scripts\\compare_all_results.py")


if __name__ == "__main__":
    main()
