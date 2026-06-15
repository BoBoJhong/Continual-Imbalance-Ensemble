"""
診斷並修復 mover_labeled.csv 的四個品質問題：
  1. BIRTH_DATE 語意確認 → age_years 重算或直接用原值
  2. or_duration_hours 負值 → clip(0)
  3. evt_minutes 極端值 → clip
  4. 缺值填補摘要（記錄各欄缺值率，供後續建模參考）

輸出：data/processed/MOVER/model/mover_labeled_fixed.csv
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PI   = Path("data/raw/MOVER/patient_information.csv")
LABELED  = Path("data/processed/MOVER/model/mover_labeled.csv")
OUT      = Path("data/processed/MOVER/model/mover_labeled_fixed.csv")

SEP = "-" * 60

# ── 0. Load ────────────────────────────────────────────────────
df = pd.read_csv(LABELED)
print(f"Loaded {len(df):,} rows from {LABELED}")

# ── 1. BIRTH_DATE 語意確認 ────────────────────────────────────
print(SEP)
print("1. BIRTH_DATE investigation")

raw_pi = pd.read_csv(RAW_PI, usecols=["LOG_ID", "BIRTH_DATE"])
sample_vals = raw_pi["BIRTH_DATE"].dropna().unique()[:30]
print(f"   Sample BIRTH_DATE values: {list(sample_vals)}")

# 嘗試把 BIRTH_DATE 當純數值（age）
numeric_vals = pd.to_numeric(raw_pi["BIRTH_DATE"], errors="coerce")
n_numeric = numeric_vals.notna().sum()
print(f"   Values parseable as pure integer/float: {n_numeric:,} / {len(raw_pi):,}")
print(f"   Numeric range: {numeric_vals.min():.0f} – {numeric_vals.max():.0f}")

# 嘗試當 datetime
dt_vals = pd.to_datetime(raw_pi["BIRTH_DATE"], errors="coerce", format="mixed")
n_dt = dt_vals.notna().sum()
print(f"   Values parseable as datetime: {n_dt:,} / {len(raw_pi):,}")

# 結論
if n_numeric > n_dt * 0.9:
    birth_is_age = True
    print("   CONCLUSION: BIRTH_DATE appears to store AGE directly (integer)")
    print("   -> age_years = BIRTH_DATE (no surgery_date calculation needed)")
else:
    birth_is_age = False
    print("   CONCLUSION: BIRTH_DATE looks like actual datetime")
    print("   -> age_years = surgery_date - birth_date / 365.25")

# Merge raw BIRTH_DATE back and rebuild age_years (dedup raw_pi first)
raw_pi["birth_numeric"] = numeric_vals
raw_pi_dedup = raw_pi.drop_duplicates(subset="LOG_ID", keep="first")
df = df.merge(raw_pi_dedup[["LOG_ID","birth_numeric"]], on="LOG_ID", how="left")
if birth_is_age:
    df["age_years"] = df["birth_numeric"]
    print(f"   age_years after fix — min={df['age_years'].min():.1f}  "
          f"median={df['age_years'].median():.1f}  max={df['age_years'].max():.1f}")
df = df.drop(columns=["birth_numeric"])

# ── 2. or_duration_hours negative values ─────────────────────
print(SEP)
print("2. or_duration_hours negative values")
neg_mask = df["or_duration_hours"] < 0
print(f"   Negative rows before fix: {neg_mask.sum():,}")
df["or_duration_hours"] = df["or_duration_hours"].clip(lower=0)
print(f"   Negative rows after clip(0): {(df['or_duration_hours'] < 0).sum():,}")
print(f"   Range after fix: [{df['or_duration_hours'].min():.2f}, {df['or_duration_hours'].max():.2f}]")

# ── 3. evt_minutes extreme values ────────────────────────────
print(SEP)
print("3. evt_minutes extreme value clipping")
EVT_LOWER, EVT_UPPER = -300, 1440   # -5h to +24h from OR start
for col in ["evt_first_minutes_from_or_in", "evt_last_minutes_from_or_in", "evt_span_minutes"]:
    if col not in df.columns:
        continue
    before_min, before_max = df[col].min(), df[col].max()
    lower = 0 if col == "evt_span_minutes" else EVT_LOWER
    df[col] = df[col].clip(lower=lower, upper=EVT_UPPER)
    after_min, after_max = df[col].min(), df[col].max()
    clipped = ((df[col] == lower) | (df[col] == EVT_UPPER)).sum() if col != "evt_span_minutes" else 0
    print(f"   {col}")
    print(f"     before: [{before_min:.0f}, {before_max:.0f}]")
    print(f"     after:  [{after_min:.0f}, {after_max:.0f}]")

# ── 4. Missing value summary ──────────────────────────────────
print(SEP)
print("4. Missing value summary (post-fix)")
miss_cols = {
    col: df[col].isna().sum()
    for col in df.columns
    if df[col].isna().sum() > 0
}
for col, n in sorted(miss_cols.items(), key=lambda x: -x[1]):
    pct = n / len(df) * 100
    strategy = ""
    if col in ("ASA_RATING_C",):
        strategy = "→ median impute for LR; tree models handle natively"
    elif col in ("or_duration_hours","anesthesia_duration_hours"):
        strategy = "→ median impute for LR; tree models handle natively"
    elif col in ("PRIMARY_ANES_TYPE_NM","ASA_RATING"):
        strategy = "→ mode impute or 'Unknown' category"
    print(f"   {col:<42s} {n:>5,} ({pct:5.1f}%)  {strategy}")

# ── 5. Final sanity check ─────────────────────────────────────
print(SEP)
print("5. Final sanity check")
assert df["LOG_ID"].nunique() == len(df), "Duplicate LOG_IDs detected!"
assert df["any_complication"].isna().sum() == 0, "Label has NaN!"
assert (df["or_duration_hours"] < 0).sum() == 0, "Negative or_duration_hours!"
print(f"   LOG_ID unique:           {df['LOG_ID'].nunique():,} / {len(df):,}  OK")
print(f"   any_complication NaN:    {df['any_complication'].isna().sum()}  OK")
print(f"   or_duration_hours < 0:   {(df['or_duration_hours']<0).sum()}  OK")
print(f"   age_years range:         {df['age_years'].min():.1f} – {df['age_years'].max():.1f}")
print(f"   pos_rate (overall):      {df['any_complication'].mean():.4f}")

# ── 6. Save ────────────────────────────────────────────────────
df.to_csv(OUT, index=False)
print(SEP)
print(f"Saved fixed dataset: {OUT}  ({len(df):,} rows, {len(df.columns)} cols)")
print("READY FOR MODELING")
