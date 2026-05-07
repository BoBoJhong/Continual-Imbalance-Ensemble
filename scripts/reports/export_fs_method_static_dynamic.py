"""
彙整「三種 FS × 靜態集成 × 動態集成」結果，供論文貼表。

資料來源（遷移後）：
  results/phase3_feature/<mutual_info|shap|rfe>/xgb_bankruptcy_fs_full_static.csv
  results/phase3_feature/<...>/xgb_bankruptcy_fs_full_des.csv

靜態：該方法對應 fs 列，保留 Old_3 / New_3 / All_6 全部結果。
動態：該方法對應 fs 列 + DES 池（Dynamic_*）；另附「依 AUC 該 split 最佳 DES」一列。

另會輸出六個獨立檔（三種 FS × 靜態／動態）：
  xgb_bankruptcy_<MI|SHAP|RFE>_static.csv
  xgb_bankruptcy_<MI|SHAP|RFE>_dynamic.csv

執行（專案根目錄）：
  python scripts/reports/export_fs_method_static_dynamic.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _split_sort_key(series: pd.Series) -> pd.Series:
    def key_one(s: str) -> int:
        m = re.match(r"split_(\d+)\+", str(s))
        return int(m.group(1)) if m else 0

    return series.map(key_one)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "phase3_feature" / "thesis"

# 方法資料夾名、結果檔內 fs 欄位值、論文用簡稱
METHODS: tuple[tuple[str, str, str], ...] = (
    ("mutual_info", "mi_r80", "MI"),
    ("shap", "shap_r80", "SHAP"),
    ("rfe", "rfe_r80", "RFE"),
)
METHOD_ORDER = {m[2]: i for i, m in enumerate(METHODS)}

STATIC_ENSEMBLE = "New_3"
STATIC_NAME = "靜態集成"
DYNAMIC_NAME = "動態集成"
INT_ORDER = {STATIC_NAME: 0, DYNAMIC_NAME: 1}


def _pick_static(df: pd.DataFrame, fs_tag: str) -> pd.DataFrame:
    m = (df["fs"].astype(str) == fs_tag) & (df["ensemble"].astype(str) == STATIC_ENSEMBLE)
    return df.loc[m, ["split", "AUC", "F1", "Recall"]].copy()


def _static_full(df: pd.DataFrame, fs_tag: str) -> pd.DataFrame:
    m = df["fs"].astype(str) == fs_tag
    return df.loc[m, ["split", "ensemble", "AUC", "F1", "Recall"]].copy()


def _dynamic_long(df: pd.DataFrame, fs_tag: str) -> pd.DataFrame:
    m = df["fs"].astype(str) == fs_tag
    out = df.loc[m, ["split", "ensemble", "AUC", "F1", "Recall"]].copy()
    return out


def _best_dynamic_per_split(dyn: pd.DataFrame) -> pd.DataFrame:
    """每個 split 取 AUC 最高的 DES 列。"""
    if dyn.empty:
        return dyn
    idx = dyn.groupby("split")["AUC"].idxmax()
    return dyn.loc[idx].reset_index(drop=True)


def _safe_to_csv(df: pd.DataFrame, path: Path, *, float_format: str = "%.6f") -> Path:
    """Write CSV; if target is locked on Windows, fallback to *_full.csv."""
    try:
        df.to_csv(path, index=False, float_format=float_format)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_full{path.suffix}")
        df.to_csv(alt, index=False, float_format=float_format)
        return alt


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    long_rows: list[pd.DataFrame] = []
    wide_merged: pd.DataFrame | None = None

    base = ROOT / "results" / "phase3_feature"

    for subdir, fs_tag, label in METHODS:
        p_static = base / subdir / "xgb_bankruptcy_fs_full_static.csv"
        p_des = base / subdir / "xgb_bankruptcy_fs_full_des.csv"
        if not p_static.exists() or not p_des.exists():
            raise FileNotFoundError(f"缺少 {p_static} 或 {p_des}")

        st = pd.read_csv(p_static)
        dy = pd.read_csv(p_des)

        s = _pick_static(st, fs_tag)

        # 六檔之一：該方法靜態完整（Old_3 / New_3 / All_6）
        static_only = _static_full(st, fs_tag)
        static_only["_sk"] = _split_sort_key(static_only["split"])
        static_only = static_only.sort_values(["_sk", "ensemble"], kind="stable").drop(columns=["_sk"])
        _safe_to_csv(static_only, OUT_DIR / f"xgb_bankruptcy_{label}_static.csv")

        s.insert(0, "fs_method", label)
        s.insert(1, "integration", STATIC_NAME)
        s.insert(2, "ensemble", STATIC_ENSEMBLE)
        long_rows.append(s)

        d_long = _dynamic_long(dy, fs_tag)

        # 六檔之一：該方法動態（三種 DES 各列）
        dyn_only = d_long.copy()
        dyn_only["_sk"] = _split_sort_key(dyn_only["split"])
        dyn_only = dyn_only.sort_values(["_sk", "ensemble"], kind="stable").drop(columns=["_sk"])
        _safe_to_csv(dyn_only, OUT_DIR / f"xgb_bankruptcy_{label}_dynamic.csv")
        for _, row in d_long.iterrows():
            long_rows.append(
                pd.DataFrame(
                    [
                        {
                            "split": row["split"],
                            "fs_method": label,
                            "integration": DYNAMIC_NAME,
                            "ensemble": row["ensemble"],
                            "AUC": row["AUC"],
                            "F1": row["F1"],
                            "Recall": row["Recall"],
                        }
                    ]
                )
            )

        best = _best_dynamic_per_split(d_long)
        best = best.rename(
            columns={
                "AUC": f"{label}_dyn_best_AUC",
                "F1": f"{label}_dyn_best_F1",
                "Recall": f"{label}_dyn_best_Recall",
                "ensemble": f"{label}_dyn_best_DES",
            }
        )

        w_st = s.rename(
            columns={
                "AUC": f"{label}_static_AUC",
                "F1": f"{label}_static_F1",
                "Recall": f"{label}_static_Recall",
            }
        ).drop(columns=["fs_method", "integration", "ensemble"])

        w = w_st.merge(
            best[
                [
                    "split",
                    f"{label}_dyn_best_DES",
                    f"{label}_dyn_best_AUC",
                    f"{label}_dyn_best_F1",
                    f"{label}_dyn_best_Recall",
                ]
            ],
            on="split",
            how="outer",
        )
        if wide_merged is None:
            wide_merged = w
        else:
            wide_merged = wide_merged.merge(w, on="split", how="outer")

    long_df = pd.concat(long_rows, ignore_index=True)
    if not long_df.empty and "split" in long_df.columns:
        long_df["_sk"] = _split_sort_key(long_df["split"])
        long_df["_mo"] = long_df["fs_method"].map(METHOD_ORDER)
        long_df["_io"] = long_df["integration"].map(INT_ORDER)
        long_df = long_df.sort_values(["_mo", "_sk", "_io", "ensemble"], kind="stable").drop(columns=["_sk", "_mo", "_io"])

    long_path = OUT_DIR / "xgb_bankruptcy_fs_mi_shap_rfe_static_dynamic_long.csv"
    _safe_to_csv(long_df, long_path)

    wide_df = wide_merged if wide_merged is not None else pd.DataFrame()
    if not wide_df.empty and "split" in wide_df.columns:
        wide_df["_sk"] = _split_sort_key(wide_df["split"])
        wide_df = wide_df.sort_values("_sk", kind="stable").drop(columns=["_sk"]).reset_index(drop=True)
    wide_path = OUT_DIR / "xgb_bankruptcy_fs_mi_shap_rfe_static_dynamic_wide.csv"
    _safe_to_csv(wide_df, wide_path)

    six = [OUT_DIR / f"xgb_bankruptcy_{m[2]}_{kind}.csv" for m in METHODS for kind in ("static", "dynamic")]
    print(f"Wrote {long_path}")
    print(f"Wrote {wide_path}")
    for p in six:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
