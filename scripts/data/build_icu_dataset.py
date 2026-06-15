"""
從 MIMIC-IV（或相容欄位）CSV 整合 ICU stay 層級特徵表，輸出 icu_stay_features.csv。

預設整合邏輯（研究單位：一次 ICU stay = icustays.stay_id）：
  1) icustays 為主表
  2) 併 admissions → hospital_expire_flag 等
  3) 併 patients → gender, anchor_age, anchor_year_group 等
  4) labevents：僅選定 itemid，且 charttime（可回填 storetime）落在該 stay 的 intime 之後
     0～24 小時內，且不晚於該 stay 的 outtime
  5) 依 stay_id × lab_name 聚合 valuenum：mean / min / max / last / count
  6) 併回 cohort；y = hospital_expire_flag
  7) 寫出寬表；保留 intime / outtime 與 anchor_year_group，供依時間順序做 Old/New/Test（例如同年群內以前後段切分），而非隨機切分。
     數值型 lab 欄位可能有缺失（臨床上常見），建模階段建議以 SimpleImputer(strategy="median") 等方式處理。

目錄配置（擇一）：
  --mimic-root path     → path/hosp/*.csv 與 path/icu/icustays.csv
  --hosp-dir / --icu-dir 明確指定

大型 labevents：使用 --chunksize 分塊讀取，避免一次載入記憶體。

用法：
  python scripts/data/build_icu_dataset.py --mimic-root "D:/mimiciv/3.0"
  python scripts/data/build_icu_dataset.py --hosp-dir ./hosp --icu-dir ./icu -o data/processed/icu_stay_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 預設檢驗項目（itemid 依 MIMIC 常用對照；若你的版本缺部分 id，請改對照或從 d_labitems 查）
# ---------------------------------------------------------------------------

SELECTED_LABS: dict[str, list[int]] = {
    "glucose": [50931, 50809, 52569],
    "creatinine": [50912, 52546],
    "urea_nitrogen": [51006, 52647],
    "white_blood_cells": [51301, 51755, 51756],
    "hemoglobin": [51222, 50811, 51640],
    "platelet_count": [51265, 51704],
    "sodium": [50983, 52623, 50824],
    "potassium": [50971, 52610, 50822],
    "chloride": [50902, 52535, 50806],
    "bicarbonate": [50882, 50803],
    "lactate": [50813, 52442, 53154],
}


def _resolve_paths(
    mimic_root: Path | None,
    hosp_dir: Path | None,
    icu_dir: Path | None,
) -> tuple[Path, Path]:
    if mimic_root is not None:
        root = mimic_root.resolve()
        return root / "hosp", root / "icu"
    if hosp_dir is None or icu_dir is None:
        raise ValueError("請提供 --mimic-root，或同時提供 --hosp-dir 與 --icu-dir")
    return hosp_dir.resolve(), icu_dir.resolve()


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"找不到檔案: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def _build_itemid_to_lab() -> dict[int, str]:
    m: dict[int, str] = {}
    for lab_name, itemids in SELECTED_LABS.items():
        for i in itemids:
            m[int(i)] = lab_name
    return m


def _optional_warn_itemids(d_labitems: pd.DataFrame, selected: set[int]) -> None:
    if "itemid" not in d_labitems.columns:
        return
    known = set(pd.to_numeric(d_labitems["itemid"], errors="coerce").dropna().astype(int))
    missing = sorted(selected - known)
    if missing:
        print(
            f"[WARN] 下列 itemid 未出現在 d_labitems 中（可能版本不同）: {missing[:20]}"
            + (" ..." if len(missing) > 20 else ""),
            file=sys.stderr,
        )


def _load_labevents_chunked(
    labevents_path: Path,
    *,
    subject_ids: set[int],
    itemids: set[int],
    chunksize: int,
) -> pd.DataFrame:
    usecols_candidates = ["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum"]
    # 只讀存在的欄位（不同匯出版本可能略異）
    head = pd.read_csv(labevents_path, nrows=0)
    usecols = [c for c in usecols_candidates if c in head.columns]
    if "subject_id" not in usecols or "itemid" not in usecols:
        raise ValueError(f"labevents 缺少必要欄位，目前欄位: {head.columns.tolist()}")

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        labevents_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce")
        chunk = chunk[chunk["subject_id"].notna()]
        chunk["subject_id"] = chunk["subject_id"].astype(np.int64)
        chunk = chunk[chunk["subject_id"].isin(subject_ids)]
        if chunk.empty:
            continue
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue
        if "hadm_id" in chunk.columns:
            chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        parts.append(chunk)

    if not parts:
        return pd.DataFrame(columns=usecols)

    labs = pd.concat(parts, ignore_index=True)
    # 時間：charttime 優先，缺則 storetime（若存在）
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    if "storetime" in labs.columns:
        labs["storetime"] = pd.to_datetime(labs["storetime"], errors="coerce")
        labs["charttime"] = labs["charttime"].fillna(labs["storetime"])
    if "valuenum" in labs.columns:
        labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    return labs


def build_dataset(
    hosp_dir: Path,
    icu_dir: Path,
    *,
    chunksize: int = 5_000_000,
) -> pd.DataFrame:
    patients_path = hosp_dir / "patients.csv"
    admissions_path = hosp_dir / "admissions.csv"
    labevents_path = hosp_dir / "labevents.csv"
    d_labitems_path = hosp_dir / "d_labitems.csv"
    icustays_path = icu_dir / "icustays.csv"

    patients = _read_csv(patients_path)
    admissions = _read_csv(admissions_path)
    icustays = _read_csv(icustays_path)
    d_labitems = _read_csv(d_labitems_path)

    print("patients:", patients.shape)
    print("admissions:", admissions.shape)
    print("icustays:", icustays.shape)
    print("d_labitems:", d_labitems.shape)

    itemid_to_lab = _build_itemid_to_lab()
    selected_itemids = set(itemid_to_lab.keys())
    _optional_warn_itemids(d_labitems, selected_itemids)

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"], errors="coerce")

    icustays["intime"] = pd.to_datetime(icustays["intime"], errors="coerce")
    icustays["outtime"] = pd.to_datetime(icustays["outtime"], errors="coerce")

    adm_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "hospital_expire_flag",
    ]
    adm_cols = [c for c in adm_cols if c in admissions.columns]
    cohort = icustays.merge(
        admissions[adm_cols],
        on=["subject_id", "hadm_id"],
        how="left",
    )

    pt_cols = [
        "subject_id",
        "gender",
        "anchor_age",
        "anchor_year",
        "anchor_year_group",
        "dod",
    ]
    pt_cols = [c for c in pt_cols if c in patients.columns]
    cohort = cohort.merge(patients[pt_cols], on="subject_id", how="left")

    print("cohort:", cohort.shape)
    if "hospital_expire_flag" in cohort.columns:
        print(cohort["hospital_expire_flag"].value_counts(dropna=False))
    if "anchor_year_group" in cohort.columns:
        print(cohort["anchor_year_group"].value_counts(dropna=False))

    cohort_subjects = set(cohort["subject_id"].astype(np.int64).unique())
    print("Reading labevents (chunked)...")
    labs = _load_labevents_chunked(
        labevents_path,
        subject_ids=cohort_subjects,
        itemids=selected_itemids,
        chunksize=chunksize,
    )
    print("selected labevents rows:", len(labs))

    labs_24h = pd.DataFrame()
    if not labs.empty:
        labs = labs.dropna(subset=["valuenum", "charttime"])
        labs["_itemid"] = pd.to_numeric(labs["itemid"], errors="coerce")
        labs = labs[labs["_itemid"].notna()]
        labs["lab_name"] = labs["_itemid"].astype(np.int64).map(itemid_to_lab)
        labs = labs.dropna(subset=["lab_name"]).drop(columns=["_itemid"])

        stay_keys = cohort[
            ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
        ].copy()
        stay_keys["subject_id"] = stay_keys["subject_id"].astype(np.int64)
        stay_keys["hadm_id"] = stay_keys["hadm_id"].astype(np.int64)

        labs_icu = labs.merge(stay_keys, on=["subject_id", "hadm_id"], how="inner")
        labs_icu["hours_from_icu_admit"] = (
            labs_icu["charttime"] - labs_icu["intime"]
        ).dt.total_seconds() / 3600.0

        labs_24h = labs_icu[
            (labs_icu["hours_from_icu_admit"] >= 0)
            & (labs_icu["hours_from_icu_admit"] <= 24)
            & (labs_icu["charttime"] <= labs_icu["outtime"])
        ].copy()

        print("labs within first 24h (per stay, capped by outtime):", labs_24h.shape)
        print("ICU stays with >=1 lab row:", labs_24h["stay_id"].nunique())

    if labs_24h.empty:
        lab_features = cohort[["stay_id"]].drop_duplicates().copy()
    else:
        print(labs_24h["lab_name"].value_counts())
        labs_24h = labs_24h.sort_values(["stay_id", "lab_name", "charttime"])
        lab_agg = labs_24h.groupby(["stay_id", "lab_name"])["valuenum"].agg(
            ["mean", "min", "max", "last", "count"]
        )
        lab_features = lab_agg.unstack("lab_name")
        lab_features.columns = [f"{lab}_{stat}" for stat, lab in lab_features.columns]
        lab_features = lab_features.reset_index()

    print("lab_features:", lab_features.shape)

    dataset = cohort.merge(lab_features, on="stay_id", how="left")
    print("dataset before cleaning:", dataset.shape)

    if "gender" in dataset.columns:
        dataset["gender_male"] = dataset["gender"].map({"M": 1, "F": 0})
    else:
        dataset["gender_male"] = np.nan

    dataset["icu_admit_shifted_year"] = dataset["intime"].dt.year

    if "hospital_expire_flag" in dataset.columns:
        dataset["y"] = dataset["hospital_expire_flag"]
    else:
        dataset["y"] = np.nan

    print("Target distribution:")
    print(dataset["y"].value_counts(dropna=False))
    if "anchor_year_group" in dataset.columns:
        print("Year group vs target:")
        print(pd.crosstab(dataset["anchor_year_group"], dataset["y"]))

    basic_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "anchor_year_group",
        "anchor_age",
        "gender",
        "gender_male",
        "race",
        "insurance",
        "admission_type",
        "first_careunit",
        "last_careunit",
        "los",
        "hospital_expire_flag",
        "y",
    ]
    basic_cols = [c for c in basic_cols if c in dataset.columns]

    prefixes = tuple(
        f"{name}_" for name in [
            "glucose",
            "creatinine",
            "urea_nitrogen",
            "white_blood_cells",
            "hemoglobin",
            "platelet_count",
            "sodium",
            "potassium",
            "chloride",
            "bicarbonate",
            "lactate",
        ]
    )
    lab_cols = [c for c in dataset.columns if c.startswith(prefixes)]
    final_cols = basic_cols + lab_cols
    final_dataset = dataset[final_cols].copy()

    print("final_dataset:", final_dataset.shape)
    return final_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="整合 MIMIC ICU stay + 入院 24h 內 lab 特徵")
    p.add_argument(
        "--mimic-root",
        type=Path,
        default=None,
        help="MIMIC 根目錄（其下需有 hosp/ 與 icu/）",
    )
    p.add_argument("--hosp-dir", type=Path, default=None, help="hosp CSV 目錄")
    p.add_argument("--icu-dir", type=Path, default=None, help="icu CSV 目錄（含 icustays.csv）")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/icu_stay_features.csv"),
        help="輸出 CSV 路徑（預設 data/processed/icu_stay_features.csv）",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=5_000_000,
        help="讀取 labevents 每批列數（降低可減少記憶體）",
    )
    args = p.parse_args()

    hosp_dir, icu_dir = _resolve_paths(args.mimic_root, args.hosp_dir, args.icu_dir)
    print("hosp_dir:", hosp_dir)
    print("icu_dir:", icu_dir)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final_df = build_dataset(hosp_dir, icu_dir, chunksize=int(args.chunksize))
    final_df.to_csv(out_path, index=False)
    print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()
