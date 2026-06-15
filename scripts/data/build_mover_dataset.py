"""
MOVER 三層資料管線：Raw（不修改）→ Feature（各表清理／彙總）→ Model（每 LOG_ID 一列）。

Raw：data/raw/MOVER/*.csv
Feature：data/processed/MOVER/feature/
Model：data/processed/MOVER/model/mover_surgery_model_table.csv

重複 LOG_ID 處理（兩類）：
  - 完全相同行（資料重複匯出）：直接 drop_duplicates，保留第一列。
  - 欄位有衝突的 LOG_ID（10 筆）：保留 raw CSV 中出現較早的那列（keep='first'），
    以 pi_dedup_flag 欄位標記：
      'ok'           – 不重複
      'exact_copy'   – 完全相同，刪除多餘列
      'conflict_keep'– 欄位有衝突，保留本列（較早）
      'conflict_drop'– 欄位有衝突，已被刪除（不出現在輸出）

標籤語意（三態）：
  any_complication = 1  → 確認有術後併發症
  any_complication = 0  → 有記錄，且全部值均為 None/空 → 確認無
  label_status = 'unlabeled' → complications 表內找不到此 LOG_ID → 未知，不可視為 0

特徵：僅保留手術前／手術當下可知欄位（排除 LOS、出院、ICU 等術後洩漏欄位）。
PRIMARY_PROCEDURE_NM：保留 top-N（預設 50）最高頻手術，其餘合併為 "Other_procedure"。
時間切分：2017–2019 train、2020 val、2021 test、2022 drift、其餘 other。

用法：
  python scripts/data/build_mover_dataset.py
  python scripts/data/build_mover_dataset.py --raw-dir data/raw/MOVER --out-dir data/processed/MOVER --top-n-procedure 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 術後／結果相關欄位：第一版不進特徵（避免洩漏）
PI_EXCLUDED_COLS = frozenset({
    "DISCH_DISP_C",
    "DISCH_DISP",
    "HOSP_ADMSN_TIME",
    "HOSP_DISCH_TIME",
    "LOS",
    "ICU_ADMIN_FLAG",
})

# 主表保留欄位（含 key 與時間戳，供算 duration / age）
PI_KEEP_COLS = [
    "LOG_ID",
    "MRN",
    "SURGERY_DATE",
    "BIRTH_DATE",
    "SEX",
    "PRIMARY_ANES_TYPE_NM",
    "ASA_RATING_C",
    "ASA_RATING",
    "PATIENT_CLASS_GROUP",
    "PATIENT_CLASS_NM",
    "PRIMARY_PROCEDURE_NM",
    "IN_OR_DTTM",
    "OUT_OR_DTTM",
    "AN_START_DATETIME",
    "AN_STOP_DATETIME",
]

# 特定 procedure event → 0/1 flag 欄位
EVENT_FLAGS: dict[str, str] = {
    "Sign In": "evt_sign_in",
    "Intubation": "evt_intubation",
    "Extubation": "evt_extubation",
    "Transported to PACU/ICU with O2, vital signs stable": "evt_transported_pacu_icu",
    "Case Delayed": "evt_case_delayed",
    "IV Antibiotics": "evt_iv_antibiotics",
    "Two Anti-Emetics Administered": "evt_two_anti_emetics",
    "Emergence": "evt_emergence",
    "Induction": "evt_induction",
    "Tourniquet Inflated": "evt_tourniquet_inflated",
    "TEE Echo Placed": "evt_tee_echo_placed",
    "LMA  Placed": "evt_lma_placed",
}

TEMPORAL_SPLIT_RULES: list[tuple[str, range]] = [
    ("train", range(2017, 2020)),
    ("val", range(2020, 2021)),
    ("test", range(2021, 2022)),
    ("drift", range(2022, 2023)),
]

TOP_N_PROCEDURE_DEFAULT = 50


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"找不到檔案: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def _parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _duration_hours(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / 3600.0


def _is_complication_value(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip()
    return bool(s) and s.lower() not in {"none", "nan"}


def _dedup_patient_info(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    壓成每 LOG_ID 一列，回傳清洗後的 DataFrame 與重複統計。

    策略：
      1. 先找完全相同的多餘列（exact duplicate rows）→ 刪除
      2. 剩餘仍有重複 LOG_ID 者（欄位有衝突）→ keep='first'，標記 flag

    回傳 df 含 pi_dedup_flag 欄。
    """
    n_raw = len(df)

    # Step 1: 刪除完全相同的行
    df = df.drop_duplicates(keep="first").copy()
    n_after_exact = len(df)
    n_exact_removed = n_raw - n_after_exact

    # Step 2: 標記衝突型重複（同 LOG_ID 但至少一欄不同）
    dup_mask = df.duplicated(subset="LOG_ID", keep=False)
    conflict_ids = set(df.loc[dup_mask, "LOG_ID"].unique())

    df["pi_dedup_flag"] = "ok"
    n_conflict_dropped = 0
    if conflict_ids:
        keep_mask = ~df.duplicated(subset="LOG_ID", keep="first")
        df.loc[df["LOG_ID"].isin(conflict_ids) & keep_mask, "pi_dedup_flag"] = "conflict_keep"

        drop_mask = df.duplicated(subset="LOG_ID", keep="first") & df["LOG_ID"].isin(conflict_ids)
        n_conflict_dropped = int(drop_mask.sum())
        df.loc[drop_mask, "pi_dedup_flag"] = "conflict_drop"
        df = df[df["pi_dedup_flag"] != "conflict_drop"].copy()

    stats = {
        "n_raw": n_raw,
        "n_exact_rows_removed": n_exact_removed,
        "n_conflict_logids": len(conflict_ids),
        "n_conflict_rows_dropped": n_conflict_dropped,
        "n_final": len(df),
    }
    return df.reset_index(drop=True), stats


def build_patient_information_features(raw_dir: Path, top_n_procedure: int = TOP_N_PROCEDURE_DEFAULT) -> tuple[pd.DataFrame, dict]:
    path = raw_dir / "patient_information.csv"
    df = _read_csv(path)

    missing = [c for c in PI_KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"patient_information 缺少欄位: {missing}")

    leak_in_keep = PI_EXCLUDED_COLS & set(PI_KEEP_COLS)
    if leak_in_keep:
        raise ValueError(f"不應納入特徵的欄位仍在 PI_KEEP_COLS: {leak_in_keep}")

    out = df[PI_KEEP_COLS].copy()
    out["SURGERY_DATE"] = _parse_dt(out["SURGERY_DATE"])
    out["BIRTH_DATE"] = _parse_dt(out["BIRTH_DATE"])
    out["IN_OR_DTTM"] = _parse_dt(out["IN_OR_DTTM"])
    out["OUT_OR_DTTM"] = _parse_dt(out["OUT_OR_DTTM"])
    out["AN_START_DATETIME"] = _parse_dt(out["AN_START_DATETIME"])
    out["AN_STOP_DATETIME"] = _parse_dt(out["AN_STOP_DATETIME"])

    # Dedup（壓成每 LOG_ID 一列）
    out, dedup_stats = _dedup_patient_info(out)

    # BIRTH_DATE 欄位儲存的是年齡（整數），不是出生日期；直接當數值讀取
    out["age_years"] = pd.to_numeric(out["BIRTH_DATE"], errors="coerce")
    out["surgery_year"] = out["SURGERY_DATE"].dt.year
    out["or_duration_hours"] = _duration_hours(out["IN_OR_DTTM"], out["OUT_OR_DTTM"])
    out["anesthesia_duration_hours"] = _duration_hours(
        out["AN_START_DATETIME"], out["AN_STOP_DATETIME"]
    )

    # Top-N procedure grouping
    top_procs = (
        out["PRIMARY_PROCEDURE_NM"]
        .value_counts()
        .head(top_n_procedure)
        .index.tolist()
    )
    out["procedure_group"] = out["PRIMARY_PROCEDURE_NM"].where(
        out["PRIMARY_PROCEDURE_NM"].isin(top_procs), other="Other_procedure"
    )

    return out, dedup_stats


def build_complication_labels(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "patient_post_op_complications.csv"
    df = _read_csv(path, usecols=["LOG_ID", "SMRTDTA_ELEM_VALUE"])

    # 刪除完全相同的重複列（complications 表有大量 exact dup）
    df = df.drop_duplicates()

    df["has_complication_value"] = df["SMRTDTA_ELEM_VALUE"].map(_is_complication_value)
    agg = df.groupby("LOG_ID", as_index=False).agg(
        complication_row_count=("SMRTDTA_ELEM_VALUE", "size"),
        complication_value_count=("has_complication_value", "sum"),
        any_complication=("has_complication_value", "max"),
    )
    agg["any_complication"] = agg["any_complication"].astype(int)
    return agg


def build_procedure_event_features(
    raw_dir: Path,
    surgery_times: pd.DataFrame,
) -> pd.DataFrame:
    path = raw_dir / "patient_procedure events.csv"
    usecols = ["LOG_ID", "EVENT_DISPLAY_NAME", "EVENT_TIME"]
    df = _read_csv(path, usecols=usecols)
    df["EVENT_TIME"] = _parse_dt(df["EVENT_TIME"])

    times = surgery_times[["LOG_ID", "IN_OR_DTTM", "SURGERY_DATE"]].copy()
    df = df.merge(times, on="LOG_ID", how="left")
    ref = df["IN_OR_DTTM"].fillna(df["SURGERY_DATE"])
    df["event_minutes_from_or_in"] = (df["EVENT_TIME"] - ref).dt.total_seconds() / 60.0

    # 去重：同一 LOG_ID + 事件名 + 時間視為一筆
    dedup = df.drop_duplicates(subset=["LOG_ID", "EVENT_DISPLAY_NAME", "EVENT_TIME"])

    base = dedup.groupby("LOG_ID", as_index=False).agg(
        procedure_event_count=("EVENT_DISPLAY_NAME", "size"),
        procedure_event_n_unique=("EVENT_DISPLAY_NAME", "nunique"),
        evt_first_minutes_from_or_in=("event_minutes_from_or_in", "min"),
        evt_last_minutes_from_or_in=("event_minutes_from_or_in", "max"),
    )
    base["evt_span_minutes"] = (
        base["evt_last_minutes_from_or_in"] - base["evt_first_minutes_from_or_in"]
    )

    for event_name, col in EVENT_FLAGS.items():
        flagged = (
            dedup.loc[dedup["EVENT_DISPLAY_NAME"] == event_name, ["LOG_ID"]]
            .drop_duplicates()
            .assign(**{col: 1})
        )
        base = base.merge(flagged, on="LOG_ID", how="left")
        base[col] = base[col].fillna(0).astype(int)

    return base


def assign_temporal_split(surgery_year: pd.Series) -> pd.Series:
    def _split(year: float | int) -> str:
        if pd.isna(year):
            return "unknown"
        y = int(year)
        for name, years in TEMPORAL_SPLIT_RULES:
            if y in years:
                return name
        return "other"

    return surgery_year.map(_split)


def _assign_label_status(row: pd.Series) -> str:
    """三態標籤狀態。"""
    if pd.isna(row["complication_row_count"]):
        return "unlabeled"
    return "labeled_positive" if row["any_complication"] == 1 else "labeled_negative"


def build_model_table(
    pi: pd.DataFrame,
    labels: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    # 只以 LOG_ID 做 join，避免 MRN 欄位在兩表不一致造成大量缺失
    model = pi.merge(labels, on="LOG_ID", how="left")
    model = model.merge(events, on="LOG_ID", how="left")

    # 事件欄位：無 procedure event 記錄的一律填 0
    event_fill_zero = [c for c in model.columns if c.startswith(("evt_", "procedure_"))]
    model[event_fill_zero] = model[event_fill_zero].fillna(0)

    # 三態標籤
    model["label_status"] = model.apply(_assign_label_status, axis=1)

    model["temporal_split"] = assign_temporal_split(model["surgery_year"])

    # 欄位順序
    key_cols = ["LOG_ID", "MRN", "SURGERY_DATE", "surgery_year", "temporal_split", "pi_dedup_flag"]
    baseline_cols = [
        "age_years",
        "SEX",
        "ASA_RATING_C",
        "ASA_RATING",
        "PRIMARY_ANES_TYPE_NM",
        "PATIENT_CLASS_GROUP",
        "PATIENT_CLASS_NM",
        "PRIMARY_PROCEDURE_NM",
        "procedure_group",
    ]
    duration_cols = ["or_duration_hours", "anesthesia_duration_hours"]
    event_cols = sorted(c for c in model.columns if c.startswith("evt_") or c.startswith("procedure_event"))
    label_cols = [
        "label_status",
        "any_complication",
        "complication_row_count",
        "complication_value_count",
    ]
    ordered = key_cols + baseline_cols + duration_cols + event_cols + label_cols
    ordered = [c for c in ordered if c in model.columns]
    return model[ordered]


def build_dataset(raw_dir: Path, out_dir: Path, top_n_procedure: int = TOP_N_PROCEDURE_DEFAULT) -> dict:
    feature_dir = out_dir / "feature"
    model_dir = out_dir / "model"
    feature_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    pi, dedup_stats = build_patient_information_features(raw_dir, top_n_procedure=top_n_procedure)
    labels = build_complication_labels(raw_dir)
    events = build_procedure_event_features(raw_dir, pi)
    model = build_model_table(pi, labels, events)

    pi_path = feature_dir / "patient_information.csv"
    labels_path = feature_dir / "patient_post_op_complications.csv"
    events_path = feature_dir / "patient_procedure_events.csv"
    model_path = model_dir / "mover_surgery_model_table.csv"

    pi.to_csv(pi_path, index=False)
    labels.to_csv(labels_path, index=False)
    events.to_csv(events_path, index=False)
    model.to_csv(model_path, index=False)

    label_status_counts = model["label_status"].value_counts().to_dict()
    labeled = model[model["label_status"] != "unlabeled"]

    proc_group_counts = model["procedure_group"].value_counts()
    other_n = int(proc_group_counts.get("Other_procedure", 0))
    top_n_actual = int((proc_group_counts.index != "Other_procedure").sum())

    meta = {
        "raw_dir": str(raw_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "dedup": dedup_stats,
        "n_pi_features": int(len(pi)),
        "n_complication_labels": int(len(labels)),
        "n_procedure_event_features": int(len(events)),
        "n_model_rows": int(len(model)),
        "label_status_counts": label_status_counts,
        "any_complication_rate_labeled_only": float(labeled["any_complication"].mean(skipna=True)),
        "temporal_split_counts": model["temporal_split"].value_counts().to_dict(),
        "surgery_year_counts": model["surgery_year"].value_counts().sort_index().astype(int).to_dict(),
        "procedure_group": {
            "top_n_param": top_n_procedure,
            "top_n_actual": top_n_actual,
            "other_n_rows": other_n,
        },
        "label_rules": {
            "labeled_positive": "any SMRTDTA_ELEM_VALUE not null/empty/None per LOG_ID",
            "labeled_negative": "record exists but all SMRTDTA_ELEM_VALUE are None/empty",
            "unlabeled": "LOG_ID not found in complications table → unknown, not negative",
        },
        "excluded_patient_info_cols": sorted(PI_EXCLUDED_COLS),
        "temporal_splits": {name: list(r) for name, r in TEMPORAL_SPLIT_RULES},
    }
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pi": pi,
        "labels": labels,
        "events": events,
        "model": model,
        "paths": {
            "pi": pi_path,
            "labels": labels_path,
            "events": events_path,
            "model": model_path,
            "meta": meta_path,
        },
        "meta": meta,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="MOVER Raw → Feature → Model 三層資料管線")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/MOVER"),
        help="原始 CSV 目錄（唯讀）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/MOVER"),
        help="輸出根目錄（feature/ 與 model/）",
    )
    p.add_argument(
        "--top-n-procedure",
        type=int,
        default=TOP_N_PROCEDURE_DEFAULT,
        help=f"PRIMARY_PROCEDURE_NM 保留頻率最高的 N 個，其餘合併為 Other_procedure（預設 {TOP_N_PROCEDURE_DEFAULT}）",
    )
    args = p.parse_args()

    result = build_dataset(
        args.raw_dir.resolve(),
        args.out_dir.resolve(),
        top_n_procedure=args.top_n_procedure,
    )
    meta = result["meta"]
    print("Saved feature layer:")
    for k in ("pi", "labels", "events"):
        print(f"  {result['paths'][k]}")
    print("Saved model layer:")
    print(f"  {result['paths']['model']}")
    print(f"  {result['paths']['meta']}")
    print(f"Rows (unique LOG_ID): {meta['n_model_rows']}")
    print(f"Dedup: {meta['dedup']}")
    print(f"Label status: {meta['label_status_counts']}")
    print(f"any_complication rate (labeled only): {meta['any_complication_rate_labeled_only']:.4f}")
    print(f"procedure_group: {meta['procedure_group']}")
    print(f"temporal_split: {meta['temporal_split_counts']}")


if __name__ == "__main__":
    main()
