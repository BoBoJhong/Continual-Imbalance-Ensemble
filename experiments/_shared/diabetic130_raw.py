"""
UCI Diabetes 130 原始表（diabetic_data.csv）載入與二元標籤。

- 正類：readmitted == '<30'（30 天內再入院）
- 回傳特徵矩陣（不含 encounter_id / patient_nbr）、標籤、以及供 GroupKFold 用的 patient_nbr
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIABETIC_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "raw"
    / "medical"
    / "diabetes130"
    / "diabetic_data.csv"
)


def load_diabetic130_raw(
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Args:
        csv_path: 預設為 data/raw/medical/diabetes130/diabetic_data.csv

    Returns:
        X: 特徵（已移除 encounter_id, patient_nbr；'?' 已替換為 NaN）
        y: 0/1，1 = <30 再入院
        groups: patient_nbr（與 X 列對齊）
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_DIABETIC_CSV
    if not path.is_file():
        raise FileNotFoundError(f"找不到 diabetic 原始檔: {path}")

    df = pd.read_csv(path)
    if "readmitted" not in df.columns:
        raise ValueError(f"CSV 需含 readmitted 欄位: {path}")
    if "patient_nbr" not in df.columns:
        raise ValueError(f"CSV 需含 patient_nbr（供分組交叉驗證）: {path}")

    y = (df["readmitted"].astype(str).str.strip() == "<30").astype(np.int8).to_numpy()
    groups = df["patient_nbr"].to_numpy()

    drop_cols = [c for c in ("encounter_id", "patient_nbr", "readmitted") if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    X = X.replace("?", np.nan)
    return X, y, groups
