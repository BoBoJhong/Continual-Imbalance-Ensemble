"""
mover_labeled.csv 完整品質稽核腳本。
包含：
  1. 唯一性 / dedup flag
  2. 特徵完整率（缺值比例）
  3. 標籤分布 / positive rate by split
  4. 數值特徵基本統計（範圍合理性）
  5. 類別特徵覆蓋（procedure_group / SEX / ASA / 麻醉型態）
  6. conflict_keep sensitivity：8 筆保留 vs 移除的 pos_rate 差異
  7. 最終 GO / NO-GO 裁定
"""
from __future__ import annotations
import pandas as pd
import numpy as np

PATH = "data/processed/MOVER/model/mover_labeled.csv"
df = pd.read_csv(PATH)

NUMERIC_FEATURES = [
    "age_years",
    "ASA_RATING_C",
    "or_duration_hours",
    "anesthesia_duration_hours",
    "procedure_event_count",
    "procedure_event_n_unique",
    "evt_first_minutes_from_or_in",
    "evt_last_minutes_from_or_in",
    "evt_span_minutes",
]
CAT_FEATURES = [
    "SEX",
    "ASA_RATING",
    "PRIMARY_ANES_TYPE_NM",
    "PATIENT_CLASS_GROUP",
    "PATIENT_CLASS_NM",
    "procedure_group",
]
EVENT_FLAGS = [c for c in df.columns if c.startswith("evt_") and c not in
               {"evt_first_minutes_from_or_in","evt_last_minutes_from_or_in","evt_span_minutes"}]

SEP = "-" * 60
issues: list[str] = []

# ── 1. Uniqueness ──────────────────────────────────────────────
print(SEP)
print("1. Uniqueness")
print(f"   Total rows:       {len(df):>7,}")
print(f"   Unique LOG_ID:    {df['LOG_ID'].nunique():>7,}")
dup = df.duplicated(subset="LOG_ID", keep=False).sum()
print(f"   Dup LOG_ID rows:  {dup:>7,}  {'OK' if dup==0 else 'FAIL'}")
if dup > 0:
    issues.append(f"Duplicate LOG_IDs: {dup}")

print(f"   pi_dedup_flag breakdown:")
for flag, cnt in df["pi_dedup_flag"].value_counts().items():
    print(f"     {flag}: {cnt:,}")

# ── 2. Missing values ──────────────────────────────────────────
print(SEP)
print("2. Missing values (feature columns)")
high_miss = []
for col in NUMERIC_FEATURES + CAT_FEATURES + EVENT_FLAGS:
    if col not in df.columns:
        continue
    n_miss = df[col].isna().sum()
    pct = n_miss / len(df) * 100
    flag = "!!" if pct > 20 else ("!" if pct > 5 else "")
    if pct > 0:
        print(f"   {col:<42s}  {n_miss:>5,} ({pct:5.1f}%) {flag}")
    if pct > 20:
        high_miss.append(col)
        issues.append(f"High missingness ({pct:.1f}%): {col}")
if not high_miss:
    print("   No column >20% missing — OK")

print(f"\n   Label (any_complication) missing: {df['any_complication'].isna().sum()}")

# ── 3. Label distribution ──────────────────────────────────────
print(SEP)
print("3. Label distribution by temporal_split")
grp = df.groupby("temporal_split")["any_complication"].agg(
    n="size", n_pos="sum", pos_rate="mean"
).reset_index()
for _, row in grp.iterrows():
    print(f"   {row['temporal_split']:<8}  n={int(row['n']):>6,}  n_pos={int(row['n_pos']):>4,}  pos_rate={row['pos_rate']:.4f}")
overall_pos = df["any_complication"].mean()
print(f"   {'TOTAL':<8}  n={len(df):>6,}  n_pos={int(df['any_complication'].sum()):>4,}  pos_rate={overall_pos:.4f}")

# Check pos_rate stability (max/min ratio across labeled splits)
split_rates = grp[grp["temporal_split"].isin(["train","val","test","drift"])]["pos_rate"]
ratio = split_rates.max() / split_rates.min()
print(f"\n   pos_rate max/min ratio across train/val/test/drift: {ratio:.2f}x  {'OK (< 2x)' if ratio < 2 else 'WARN: large drift in label rate'}")
if ratio >= 2:
    issues.append(f"pos_rate instability: max/min={ratio:.2f}x")

# ── 4. Numeric feature ranges ──────────────────────────────────
print(SEP)
print("4. Numeric feature statistics")
stat_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
stats = df[stat_cols].describe(percentiles=[.01,.25,.5,.75,.99]).T[["count","mean","std","1%","25%","50%","75%","99%","min","max"]]
for col, row in stats.iterrows():
    ok = True
    note = ""
    if col == "age_years" and (row["min"] < 0 or row["max"] > 120):
        ok = False; note = "!! out-of-range age"
    if col in ("or_duration_hours","anesthesia_duration_hours") and row["min"] < 0:
        ok = False; note = "!! negative duration"
    if col == "ASA_RATING_C" and (row["min"] < 1 or row["max"] > 6):
        ok = False; note = "!! invalid ASA code"
    print(f"   {col:<42s}  min={row['min']:>8.2f}  median={row['50%']:>7.2f}  max={row['max']:>9.2f}  {note}")
    if not ok:
        issues.append(f"Out-of-range values: {col}")

# ── 5. Categorical coverage ────────────────────────────────────
print(SEP)
print("5. Categorical feature coverage")
for col in CAT_FEATURES:
    if col not in df.columns:
        continue
    n_unique = df[col].nunique()
    top5 = df[col].value_counts(dropna=False).head(5)
    pct_top1 = top5.iloc[0] / len(df) * 100 if len(top5) else 0
    print(f"   {col:<30s}  {n_unique:>4} unique  top-1={top5.index[0]!r} ({pct_top1:.1f}%)")

print(f"\n   procedure_group 'Other_procedure': {(df['procedure_group']=='Other_procedure').sum():,} rows "
      f"({(df['procedure_group']=='Other_procedure').mean()*100:.1f}%)")

# ── 6. Conflict-keep sensitivity ──────────────────────────────
print(SEP)
print("6. conflict_keep sensitivity analysis")
n_conflict = (df["pi_dedup_flag"] == "conflict_keep").sum()
print(f"   conflict_keep rows: {n_conflict}")

df_keep = df.copy()
df_drop = df[df["pi_dedup_flag"] != "conflict_keep"].copy()

def summary_stats(d):
    return {
        "n": len(d),
        "n_pos": int(d["any_complication"].sum()),
        "pos_rate": round(float(d["any_complication"].mean()), 6),
    }

for split in ["train", "val", "test", "drift"]:
    k = summary_stats(df_keep[df_keep["temporal_split"]==split])
    d = summary_stats(df_drop[df_drop["temporal_split"]==split])
    diff = k["pos_rate"] - d["pos_rate"]
    print(f"   {split:<8}  keep pos_rate={k['pos_rate']:.4f}  drop pos_rate={d['pos_rate']:.4f}  diff={diff:+.4f}")

k_tot = summary_stats(df_keep[df_keep["temporal_split"].isin(["train","val","test","drift"])])
d_tot = summary_stats(df_drop[df_drop["temporal_split"].isin(["train","val","test","drift"])])
print(f"   {'TOTAL':<8}  keep pos_rate={k_tot['pos_rate']:.4f}  drop pos_rate={d_tot['pos_rate']:.4f}  diff={k_tot['pos_rate']-d_tot['pos_rate']:+.4f}")
print(f"   -> 8 rows impact: {'NEGLIGIBLE' if abs(k_tot['pos_rate']-d_tot['pos_rate'])<0.001 else 'NOTABLE'}")

# ── 7. GO / NO-GO ─────────────────────────────────────────────
print(SEP)
print("7. Final QA verdict")
if issues:
    print("   ISSUES FOUND:")
    for iss in issues:
        print(f"     [!] {iss}")
else:
    print("   No critical issues found.")

print()
if not issues:
    print("  *** GO: Dataset is ready for baseline experiments ***")
else:
    print("  *** CONDITIONAL GO: Review flagged issues before training ***")
print(SEP)
