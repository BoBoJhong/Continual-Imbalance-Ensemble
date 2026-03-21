"""
Download Diabetes 130-US Hospitals (1999-2008) from UCI via ucimlrepo.
Converts to time-series format compatible with common_dataset.py (mortality target).

Dataset: https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
- 101,766 hospital stays, 47 features
- Target: readmitted (<30 days = positive/minority class, ~11% imbalance)
- Has 'admission_type_id' and numeric features suitable for block CV split

Run: python scripts/data/download_real_medical_data.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def download_and_process():
    print("=" * 60)
    print("Downloading Diabetes 130-US Hospitals (UCI dataset #296)")
    print("=" * 60)

    from ucimlrepo import fetch_ucirepo

    print("Fetching from UCI repository...")
    dataset = fetch_ucirepo(id=296)

    X_raw = dataset.data.features
    y_raw = dataset.data.targets

    print(f"Raw shape: X={X_raw.shape}, y={y_raw.shape}")
    print(f"Target column: {y_raw.columns.tolist()}")
    print(f"Target values:\n{y_raw.iloc[:, 0].value_counts()}")

    # === Build binary target: readmitted <30 days = 1 (minority/positive) ===
    target_col = y_raw.columns[0]
    y_binary = (y_raw[target_col] == "<30").astype(int)
    print(f"\nPositive rate (<30d readmit): {y_binary.mean()*100:.1f}%")

    # === Select numeric / ordinal features ===
    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
    ]
    # Add age bucket as ordinal (10-year bands encoded as int)
    X_work = X_raw[numeric_cols].copy()

    # Age: "[0-10)" -> 0, "[10-20)" -> 1, ...
    if "age" in X_raw.columns:
        age_map = {
            "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
            "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
            "[80-90)": 8, "[90-100)": 9,
        }
        X_work["age"] = X_raw["age"].map(age_map).fillna(4)

    # Insulin & diabetes meds: none/no/steady/up/down -> int
    for col in ["insulin", "metformin", "glipizide", "glyburide"]:
        if col in X_raw.columns:
            X_work[col] = X_raw[col].map(
                {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
            ).fillna(0)

    # gender
    if "gender" in X_raw.columns:
        X_work["gender"] = (X_raw["gender"] == "Male").astype(float)

    # change & diabetesMed
    for col in ["change", "diabetesMed"]:
        if col in X_raw.columns:
            X_work[col] = (X_raw[col] == "Ch").astype(float) if col == "change" else \
                          (X_raw[col] == "Yes").astype(float)

    X_work = X_work.fillna(X_work.median())
    X_work = X_work.astype(np.float64)

    # === Create pseudo-temporal order ===
    # Use encounter_id order as time proxy (records are roughly chronological)
    # Alternatively batch by number_inpatient + id to create temporal blocks
    # We just preserve the row order (already roughly 1999-2008 order in dataset)
    X_work = X_work.reset_index(drop=True)
    y_binary = y_binary.reset_index(drop=True)

    # === Add 'date' column (monthly buckets for 120 months = 10 years) ===
    n = len(X_work)
    # Spread evenly across 120 months (1999-01 to 2008-12)
    months = pd.date_range("1999-01", periods=120, freq="ME")
    month_idx = np.floor(np.arange(n) / n * 120).astype(int).clip(0, 119)
    X_work.insert(0, "date", months[month_idx])

    # === Add mortality (use readmission <30 as proxy) ===
    X_work["mortality"] = y_binary.values

    # === Save ===
    out_dir = project_root / "data" / "raw" / "medical" / "diabetes130"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "diabetes130_medical.csv"
    X_work.to_csv(out_csv, index=False)

    print(f"\nSaved to: {out_csv}")
    print(f"Shape: {X_work.shape}")
    print(f"Positive (readmit<30d) rate: {X_work['mortality'].mean()*100:.1f}%")
    print(f"Features: {[c for c in X_work.columns if c not in ['date','mortality']]}")
    return X_work


def update_common_dataset(dataset_path: Path):
    """
    Print instructions to update common_dataset.py to use Diabetes130.
    (We'll do the actual patch automatically.)
    """
    print("\n" + "=" * 60)
    print("Updating common_dataset.py to use Diabetes 130...")
    print("=" * 60)

    cd_path = project_root / "experiments" / "common_dataset.py"
    content = cd_path.read_text(encoding="utf-8")

    old_block = '''def _load_medical(logger):
    """載入 Medical 資料（UCI/synthetic，目標 mortality）。"""
    path = project_root / "data/raw/medical/synthetic/synthetic_medical_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Medical 資料不存在: {path}")
    df = pd.read_csv(path)
    y = df["mortality"]
    X = df.drop(columns=["date", "mortality"])
    X = X.astype(np.float64)
    logger.info(f"Medical 資料大小: {X.shape}, 死亡率: {y.mean()*100:.2f}%")
    return X, y'''

    new_block = f'''def _load_medical(logger):
    """載入 Medical 資料（UCI Diabetes 130-US Hospitals 1999-2008，目標: 30天再入院）。
    
    資料來源: UCI ML Repository #296
    https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
    正類（<30d readmission）比例約 11%（少數類）。
    """
    # Priority: real UCI data > synthetic fallback
    uci_path = project_root / "data/raw/medical/diabetes130/diabetes130_medical.csv"
    synthetic_path = project_root / "data/raw/medical/synthetic/synthetic_medical_data.csv"
    
    if uci_path.exists():
        path = uci_path
        logger.info("使用真實 UCI Diabetes 130-US Hospitals 資料")
    elif synthetic_path.exists():
        path = synthetic_path
        logger.info("[WARNING] 使用合成 Medical 資料（建議換成真實 UCI 資料）")
    else:
        raise FileNotFoundError(
            f"Medical 資料不存在，請執行:\\n"
            f"  python scripts/data/download_real_medical_data.py"
        )
    
    df = pd.read_csv(path)
    y = df["mortality"]
    X = df.drop(columns=["date", "mortality"])
    X = X.astype(np.float64)
    logger.info(f"Medical 資料大小: {{X.shape}}, 正類率: {{y.mean()*100:.2f}}%")
    return X, y'''

    if old_block in content:
        new_content = content.replace(old_block, new_block)
        cd_path.write_text(new_content, encoding="utf-8")
        print("common_dataset.py 已成功更新！")
    else:
        print("[WARNING] 未找到原始 _load_medical 函數，請手動確認 common_dataset.py")
        print("原始函數格式可能已更改，請手動將 _load_medical 的 path 改為:")
        print(f"  {dataset_path}")


if __name__ == "__main__":
    df = download_and_process()
    update_common_dataset(
        project_root / "data/raw/medical/diabetes130/diabetes130_medical.csv"
    )
    print("\n" + "=" * 60)
    print("Done! Now re-run experiments 06 and 08:")
    print("  python experiments/06_medical_baseline_ensemble.py")
    print("  python experiments/08_medical_des.py")
    print("=" * 60)
