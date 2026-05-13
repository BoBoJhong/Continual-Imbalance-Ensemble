"""
Extract GA (genetic algorithm) rows from Phase 3 FS sweep CSVs into results/phase3_feature/ga/.

Source of truth today: mutual_info/*_fs_sweep.csv (multi-method tables; GA is not a separate triad dir).
Full fs_full static/DES/DCS for ga_r80 is not run in the main bankruptcy/medical pipelines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_sweep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tag_col = df.columns[0]
    df = df.rename(columns={tag_col: "fs_tag"})
    return df


def _ga_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["fs_tag"].astype(str).str.startswith("ga_")].copy()


def main() -> None:
    p3 = PROJECT_ROOT / "results" / "phase3_feature"
    mi = p3 / "mutual_info"
    out = p3 / "ga"
    out.mkdir(parents=True, exist_ok=True)

    pairs = [
        (mi / "bankruptcy_fs_sweep.csv", out / "bankruptcy_ga_fs_sweep.csv"),
        (mi / "medical_fs_sweep.csv", out / "medical_ga_fs_sweep.csv"),
        (mi / "stock_spx_fs_sweep.csv", out / "stock_spx_ga_fs_sweep.csv"),
        (mi / "stock_ndx_fs_sweep.csv", out / "stock_ndx_ga_fs_sweep.csv"),
        (mi / "stock_dji_fs_sweep.csv", out / "stock_dji_ga_fs_sweep.csv"),
    ]

    lines = [
        "Phase 3 — GA 結果整理",
        "",
        "此資料夾僅收錄從 FS sweep 表抽出的 ga_r* 列（FeatureSelector method=ga）。",
        "破產／醫療主線 Phase 3（no_fs + mi/shap/rfe + static/DES/DCS）未含 GA。",
        "",
        "來源檔：",
    ]

    for src, dst in pairs:
        if not src.exists():
            continue
        df = _load_sweep(src)
        ga = _ga_only(df)
        ga.to_csv(dst, index=False, float_format="%.10f")
        lines.append(f"  {src.relative_to(PROJECT_ROOT)} -> {dst.relative_to(PROJECT_ROOT)} ({len(ga)} rows)")

    manifest = out / "_SOURCES.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
