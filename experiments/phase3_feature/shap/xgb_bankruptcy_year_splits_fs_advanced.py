"""Run advanced bankruptcy FS focused on shap method."""
from __future__ import annotations

from pathlib import Path

from experiments.phase3_feature._core.advanced_bankruptcy import run_advanced


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    run_advanced(project_root, "shap", ["shap"])


if __name__ == "__main__":
    main()
