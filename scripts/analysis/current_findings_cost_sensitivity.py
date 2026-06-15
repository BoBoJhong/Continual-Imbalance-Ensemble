"""
Cost-sensitive analysis for current weighted bankruptcy ensembles.

The available Phase 5 sweep already contains Type1_Error and Type2_Error for
each boundary / FS / Old-New weight setting. This script evaluates which setting
minimizes a simple expected cost under several Type2:Type1 cost ratios.

Outputs:
    results/statistical_tests/current_findings/bankruptcy_weighted_cost_sensitivity_all.csv
    results/statistical_tests/current_findings/bankruptcy_weighted_cost_sensitivity_best.csv
    results/statistical_tests/current_findings/bankruptcy_weighted_cost_sensitivity_transitions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

INPUT_PATH = project_root / "results" / "phase5_weighted" / "bk_validation_ross_weight_sweep.csv"
OUT_DIR = project_root / "results" / "statistical_tests" / "current_findings"

# In bankruptcy prediction, Type2 is commonly interpreted as missing a failed
# firm. A denser grid makes the model-choice transition visible as that miss
# becomes more expensive than a false alarm.
TYPE2_TO_TYPE1_RATIOS = (
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    3,
    5,
    7.5,
    10,
    15,
    20,
    30,
    50,
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    rows: list[dict] = []

    for ratio in TYPE2_TO_TYPE1_RATIOS:
        scored = df.copy()
        scored["type1_cost"] = 1.0
        scored["type2_cost"] = float(ratio)
        scored["expected_cost"] = (
            scored["type1_cost"] * scored["Type1_Error"]
            + scored["type2_cost"] * scored["Type2_Error"]
        )
        scored["type2_to_type1_cost_ratio"] = ratio
        rows.extend(scored.to_dict("records"))

    all_df = pd.DataFrame(rows)
    sort_cols = [
        "type2_to_type1_cost_ratio",
        "expected_cost",
        "F1",
        "AUC",
        "Precision",
    ]
    best_df = (
        all_df.sort_values(sort_cols, ascending=[True, True, False, False, False])
        .groupby("type2_to_type1_cost_ratio", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_df["choice"] = (
        best_df["method"].astype(str)
        + " | "
        + best_df["fs_variant"].astype(str)
        + " | w_new="
        + best_df["w_new"].map(lambda value: f"{value:.2f}")
    )

    transition_rows: list[dict] = []
    previous_choice = None
    for _, row in best_df.iterrows():
        current_choice = row["choice"]
        if current_choice != previous_choice:
            transition_rows.append(
                {
                    "from_ratio": row["type2_to_type1_cost_ratio"],
                    "choice": current_choice,
                    "method": row["method"],
                    "fs_variant": row["fs_variant"],
                    "w_new": row["w_new"],
                    "threshold": row["threshold"],
                    "expected_cost": row["expected_cost"],
                    "Type1_Error": row["Type1_Error"],
                    "Type2_Error": row["Type2_Error"],
                    "F1": row["F1"],
                    "Recall": row["Recall"],
                    "Precision": row["Precision"],
                }
            )
            previous_choice = current_choice
    transitions_df = pd.DataFrame(transition_rows)

    all_path = OUT_DIR / "bankruptcy_weighted_cost_sensitivity_all.csv"
    best_path = OUT_DIR / "bankruptcy_weighted_cost_sensitivity_best.csv"
    transitions_path = OUT_DIR / "bankruptcy_weighted_cost_sensitivity_transitions.csv"
    all_df.to_csv(all_path, index=False, float_format="%.8f")
    best_df.to_csv(best_path, index=False, float_format="%.8f")
    transitions_df.to_csv(transitions_path, index=False, float_format="%.8f")

    print(f"Saved all cost scores: {all_path}")
    print(f"Saved best cost choices: {best_path}")
    print(f"Saved choice transitions: {transitions_path}")
    print(
        best_df[
            [
                "type2_to_type1_cost_ratio",
                "choice",
                "method",
                "fs_variant",
                "w_new",
                "threshold",
                "expected_cost",
                "Type1_Error",
                "Type2_Error",
                "AUC",
                "F1",
                "Recall",
                "Precision",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
