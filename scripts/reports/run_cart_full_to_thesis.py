"""
Run bankruptcy year-split full pipeline for CART only, then export thesis files.

Output:
  results/phase3_feature/thesis/xgb_bankruptcy_CART_static.csv
  results/phase3_feature/thesis/xgb_bankruptcy_CART_dynamic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import experiments.phase3_feature.xgb_bankruptcy_year_splits_fs_full as full_mod

TMP_OUT = ROOT / "results" / "phase3_feature" / "cart"
THESIS_OUT = ROOT / "results" / "phase3_feature" / "thesis"


def main() -> None:
    TMP_OUT.mkdir(parents=True, exist_ok=True)
    THESIS_OUT.mkdir(parents=True, exist_ok=True)

    # Run the existing full pipeline, but only CART r80 config.
    full_mod.FS_CONFIGS = [("cart_r80", "cart", 0.8)]
    full_mod.OUTPUT_DIR = TMP_OUT
    full_mod.main()

    p_static = TMP_OUT / "xgb_bankruptcy_fs_full_static.csv"
    p_des = TMP_OUT / "xgb_bankruptcy_fs_full_des.csv"
    if not p_static.exists() or not p_des.exists():
        raise FileNotFoundError("CART run did not generate expected full static/des files.")

    st = pd.read_csv(p_static)
    dy = pd.read_csv(p_des)

    st_cart = st.loc[st["fs"].astype(str) == "cart_r80", ["split", "ensemble", "AUC", "F1", "Recall"]].copy()
    dy_cart = dy.loc[dy["fs"].astype(str) == "cart_r80", ["split", "ensemble", "AUC", "F1", "Recall"]].copy()

    out_static = THESIS_OUT / "xgb_bankruptcy_CART_static.csv"
    out_dynamic = THESIS_OUT / "xgb_bankruptcy_CART_dynamic.csv"
    st_cart.to_csv(out_static, index=False, float_format="%.6f")
    dy_cart.to_csv(out_dynamic, index=False, float_format="%.6f")

    print(f"Wrote {out_static} ({len(st_cart)} rows)")
    print(f"Wrote {out_dynamic} ({len(dy_cart)} rows)")


if __name__ == "__main__":
    main()

