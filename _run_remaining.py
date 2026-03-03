import sys, subprocess
from pathlib import Path
project_root = Path(__file__).parent

REMAINING = [
    ("phase4_feature",   "fs_sweep",                2700),
    ("phase5_analysis",  "split_comparison",          300),
    ("phase5_analysis",  "proportion_study",          900),
    ("phase5_analysis",  "base_learner_comparison",   600),
    ("phase5_analysis",  "stock_threshold_cost",      300),
]

ok, fail, skip = 0, 0, 0
for subdir, name, timeout in REMAINING:
    path = project_root / "experiments" / subdir / f"{name}.py"
    if not path.exists():
        print(f"[SKIP] {subdir}/{name}.py"); skip += 1; continue
    sep = "=" * 60
    print(f"\n{sep}\n[{subdir}] {name}\n{sep}")
    r = subprocess.run([sys.executable, str(path)], cwd=str(project_root), timeout=timeout)
    if r.returncode != 0:
        print(f"[FAIL] {name}  code={r.returncode}"); fail += 1
    else:
        print(f"[OK]   {name}"); ok += 1

print(f"\n=== OK={ok}  FAIL={fail}  SKIP={skip} ===")
