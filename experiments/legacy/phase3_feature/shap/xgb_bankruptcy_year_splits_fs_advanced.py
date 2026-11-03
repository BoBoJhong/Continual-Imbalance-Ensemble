"""Run advanced bankruptcy FS focused on shap method."""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from experiments.phase3_feature._core.advanced_bankruptcy import run_advanced


def main() -> None:
    project_root = Path(__file__).resolve().parents[4]
    run_advanced(project_root, "shap", ["shap"])


if __name__ == "__main__":
    main()
