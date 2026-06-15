"""
從 mover_surgery_model_table.csv 產出兩份建模用資料集。

主線（排除 unlabeled）：
  data/processed/MOVER/model/mover_labeled.csv

Sensitivity（unlabeled 視為 negative）：
  data/processed/MOVER/model/mover_pessimistic.csv

兩份都只含建模欄位（不含時間戳與中間欄位），
並輸出統計摘要 mover_modeling_summary.json。

用法：
  python scripts/data/prepare_mover_modeling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MODEL_TABLE = Path("data/processed/MOVER/model/mover_surgery_model_table.csv")
OUT_DIR = Path("data/processed/MOVER/model")

# 最終建模欄位（特徵 + key + label）
FEATURE_COLS = [
    # Keys
    "LOG_ID",
    "MRN",
    "SURGERY_DATE",
    "surgery_year",
    "temporal_split",
    "pi_dedup_flag",
    "label_status",
    # Baseline
    "age_years",
    "SEX",
    "ASA_RATING_C",
    "ASA_RATING",
    "PRIMARY_ANES_TYPE_NM",
    "PATIENT_CLASS_GROUP",
    "PATIENT_CLASS_NM",
    "PRIMARY_PROCEDURE_NM",
    "procedure_group",
    # Duration
    "or_duration_hours",
    "anesthesia_duration_hours",
    # Event summary
    "procedure_event_count",
    "procedure_event_n_unique",
    "evt_first_minutes_from_or_in",
    "evt_last_minutes_from_or_in",
    "evt_span_minutes",
    # Event flags
    "evt_case_delayed",
    "evt_emergence",
    "evt_extubation",
    "evt_induction",
    "evt_intubation",
    "evt_iv_antibiotics",
    "evt_lma_placed",
    "evt_sign_in",
    "evt_tee_echo_placed",
    "evt_tourniquet_inflated",
    "evt_transported_pacu_icu",
    "evt_two_anti_emetics",
    # Label
    "any_complication",
    "complication_row_count",
    "complication_value_count",
]


def _split_stats(df: pd.DataFrame, label_col: str = "any_complication") -> dict:
    rows = {}
    for split, grp in df.groupby("temporal_split", sort=False):
        rows[split] = {
            "n": int(len(grp)),
            "n_pos": int(grp[label_col].sum()),
            "pos_rate": round(float(grp[label_col].mean()), 4),
        }
    total = df[label_col]
    rows["_total"] = {
        "n": int(len(df)),
        "n_pos": int(total.sum()),
        "pos_rate": round(float(total.mean()), 4),
    }
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_TABLE} ...")
    model = pd.read_csv(MODEL_TABLE)

    # 只保留有定義的欄位（允許部分欄位不存在）
    keep_cols = [c for c in FEATURE_COLS if c in model.columns]
    missing_def = [c for c in FEATURE_COLS if c not in model.columns]
    if missing_def:
        print(f"  [warn] columns not found in model table (skipped): {missing_def}")

    model = model[keep_cols]

    # ── 主線：排除 unlabeled ──────────────────────────────────────────────────
    labeled = model[model["label_status"] != "unlabeled"].copy()
    labeled["any_complication"] = labeled["any_complication"].astype(int)

    labeled_path = OUT_DIR / "mover_labeled.csv"
    labeled.to_csv(labeled_path, index=False)
    print(f"Saved labeled:      {labeled_path}  ({len(labeled):,} rows)")

    # ── Sensitivity：unlabeled → negative ─────────────────────────────────────
    pessimistic = model.copy()
    pessimistic["any_complication"] = (
        pessimistic["any_complication"].fillna(0).astype(int)
    )
    pessimistic["label_status"] = pessimistic["label_status"].replace(
        "unlabeled", "unlabeled_as_neg"
    )

    pessimistic_path = OUT_DIR / "mover_pessimistic.csv"
    pessimistic.to_csv(pessimistic_path, index=False)
    print(f"Saved pessimistic:  {pessimistic_path}  ({len(pessimistic):,} rows)")

    # ── 摘要 ─────────────────────────────────────────────────────────────────
    summary = {
        "source": str(MODEL_TABLE.resolve()),
        "labeled": {
            "n_rows": int(len(labeled)),
            "n_unlabeled_excluded": int(model["label_status"].eq("unlabeled").sum()),
            "by_temporal_split": _split_stats(labeled),
        },
        "pessimistic": {
            "n_rows": int(len(pessimistic)),
            "n_unlabeled_treated_as_neg": int(model["label_status"].eq("unlabeled").sum()),
            "by_temporal_split": _split_stats(pessimistic),
        },
        "note": (
            "Use 'labeled' for primary experiments. "
            "Use 'pessimistic' for sensitivity analysis only. "
            "Difference in positive rate between the two reveals bias from unlabeled records."
        ),
    }

    summary_path = OUT_DIR / "mover_modeling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary:      {summary_path}")

    # ── 印出摘要表 ────────────────────────────────────────────────────────────
    print()
    print("=== labeled (primary) ===")
    for split, s in summary["labeled"]["by_temporal_split"].items():
        print(f"  {split:10s}  n={s['n']:>6,}  n_pos={s['n_pos']:>4,}  pos_rate={s['pos_rate']:.4f}")

    print()
    print("=== pessimistic (sensitivity) ===")
    for split, s in summary["pessimistic"]["by_temporal_split"].items():
        print(f"  {split:10s}  n={s['n']:>6,}  n_pos={s['n_pos']:>4,}  pos_rate={s['pos_rate']:.4f}")

    print()
    print("labeled pos_rate vs pessimistic pos_rate difference:")
    l_rate = summary["labeled"]["by_temporal_split"]["_total"]["pos_rate"]
    p_rate = summary["pessimistic"]["by_temporal_split"]["_total"]["pos_rate"]
    print(f"  labeled={l_rate:.4f}  pessimistic={p_rate:.4f}  diff={l_rate - p_rate:+.4f}")


if __name__ == "__main__":
    main()
