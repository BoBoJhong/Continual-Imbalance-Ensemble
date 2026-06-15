"""Quick integrity audit for mover_surgery_model_table.csv"""
import pandas as pd

model = pd.read_csv("data/processed/MOVER/model/mover_surgery_model_table.csv")

print("=== 1. Row count and uniqueness ===")
print("Total rows in model table:", len(model))
print("Unique LOG_ID:", model["LOG_ID"].nunique())
dup_count = model.duplicated(subset="LOG_ID", keep=False).sum()
print("Rows with duplicated LOG_ID:", dup_count, "(0 = fully deduped)")

print()
print("=== 2. pi_dedup_flag breakdown ===")
print(model["pi_dedup_flag"].value_counts())

print()
print("=== 3. label_status breakdown ===")
print(model["label_status"].value_counts())

print()
print("--- NaN audit per label column ---")
for col in ["any_complication", "complication_row_count", "complication_value_count"]:
    total_na = model[col].isna().sum()
    unlabeled_na = model.loc[model["label_status"] == "unlabeled", col].isna().sum()
    labeled_na = model.loc[model["label_status"] != "unlabeled", col].isna().sum()
    print(f"  {col}: total_NaN={total_na}  unlabeled_NaN={unlabeled_na}  labeled_NaN={labeled_na}")

print()
print("=== 4. unlabeled rows by temporal_split ===")
unlab = model[model["label_status"] == "unlabeled"]
print(unlab["temporal_split"].value_counts())

print()
print("=== 5. labeled-only positive rate by temporal_split ===")
labeled = model[model["label_status"] != "unlabeled"].copy()
grp = labeled.groupby("temporal_split")["any_complication"].agg(
    n="size", n_pos="sum", pos_rate="mean"
)
print(grp.round(4))

print()
print("=== 6. raw file LOG_ID uniqueness (source check) ===")
raw_pi = pd.read_csv("data/raw/MOVER/patient_information.csv", usecols=["LOG_ID"])
print("raw patient_information rows:", len(raw_pi))
print("raw unique LOG_ID:", raw_pi["LOG_ID"].nunique())
print("raw duplicate LOG_ID rows:", raw_pi.duplicated(subset="LOG_ID", keep=False).sum())
