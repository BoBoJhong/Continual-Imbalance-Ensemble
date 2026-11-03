"""Study II entry: training-only MI/CART/SHAP/RFE paired with no-FS."""
# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.phase2_ensemble.dynamic.des.bankruptcy_standard_dynamic import FS_METHODS, main

if __name__ == "__main__":
    main(default_fs_methods=FS_METHODS)
