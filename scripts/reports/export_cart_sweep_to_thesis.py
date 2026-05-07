"""
Export CART results to thesis-friendly CSV files.

Priority source (split-level, many rows):
  results/phase3_feature/combined/xgb_bankruptcy_fs_full_static.csv
  results/phase3_feature/combined/xgb_bankruptcy_fs_full_des.csv
  (filter fs == cart_r80)

Fallback source (3-row sweep summary):
  results/phase3_feature/mutual_info/*_fs_sweep.csv

Output:
  results/phase3_feature/thesis/xgb_<dataset>_CART_static.csv
  results/phase3_feature/thesis/xgb_<dataset>_CART_dynamic.csv   (bankruptcy only, if full DES exists)
  results/phase3_feature/thesis/xgb_<dataset>_cart_sweep.csv     (legacy compatibility)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = ROOT / "results" / "phase3_feature" / "mutual_info"
COMBINED_DIR = ROOT / "results" / "phase3_feature" / "combined"
THESIS_DIR = ROOT / "results" / "phase3_feature" / "thesis"

METRICS = ["AUC", "F1", "Recall"]
CART_TAGS = ["cart_r20", "cart_r50", "cart_r80"]


def _collect_cart_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # fs_sweep layout: index is fs tag, columns are metrics
    rows = []
    for tag in CART_TAGS:
        if tag in df.index:
            row = {"fs": tag}
            for m in METRICS:
                row[m] = float(df.loc[tag, m]) if m in df.columns else None
            rows.append(row)
    return pd.DataFrame(rows, columns=["fs", *METRICS])


def _write_from_full_bankruptcy() -> bool:
    """Write split-level CART static/dynamic if full files exist."""
    p_static = COMBINED_DIR / "xgb_bankruptcy_fs_full_static.csv"
    p_des = COMBINED_DIR / "xgb_bankruptcy_fs_full_des.csv"
    if not p_static.exists() or not p_des.exists():
        return False

    st = pd.read_csv(p_static)
    dy = pd.read_csv(p_des)
    if "fs" not in st.columns or "fs" not in dy.columns:
        return False

    st_cart = st.loc[st["fs"].astype(str) == "cart_r80", ["split", "ensemble", "AUC", "F1", "Recall"]].copy()
    dy_cart = dy.loc[dy["fs"].astype(str) == "cart_r80", ["split", "ensemble", "AUC", "F1", "Recall"]].copy()
    if st_cart.empty:
        return False

    out_st = THESIS_DIR / "xgb_bankruptcy_CART_static.csv"
    st_cart.to_csv(out_st, index=False, float_format="%.6f")
    print(f"Wrote {out_st}")

    if not dy_cart.empty:
        out_dy = THESIS_DIR / "xgb_bankruptcy_CART_dynamic.csv"
        dy_cart.to_csv(out_dy, index=False, float_format="%.6f")
        print(f"Wrote {out_dy}")
    return True


def main() -> None:
    THESIS_DIR.mkdir(parents=True, exist_ok=True)

    wrote_full = _write_from_full_bankruptcy()
    if wrote_full:
        print("Bankruptcy CART exported from full split-level results.")

    files = sorted(SWEEP_DIR.glob("*_fs_sweep.csv"))
    if not files:
        raise FileNotFoundError(f"No *_fs_sweep.csv in {SWEEP_DIR}")

    for src in files:
        ds = src.stem.replace("_fs_sweep", "")
        out_unified = THESIS_DIR / f"xgb_{ds}_CART_static.csv"
        out_legacy = THESIS_DIR / f"xgb_{ds}_cart_sweep.csv"
        cart = _collect_cart_rows(src)
        if cart.empty:
            continue
        cart.to_csv(out_unified, index=False, float_format="%.6f")
        cart.to_csv(out_legacy, index=False, float_format="%.6f")
        print(f"Wrote {out_unified}")
        print(f"Wrote {out_legacy}")


if __name__ == "__main__":
    main()

