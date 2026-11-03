"""
Generate synthetic Medical datasets for experiments 05-08.
Run: python scripts/data/generate_synthetic_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent




def generate_medical():
    """Generate synthetic medical time series data (500 monthly samples)."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2010-01', periods=n, freq='ME')

    d = {
        'date': dates,
        'age': np.random.randint(30, 90, n).astype(float),
        'heart_rate': np.random.normal(75, 12, n),
        'blood_pressure_sys': np.random.normal(120, 15, n),
        'blood_pressure_dia': np.random.normal(80, 10, n),
        'temperature': np.random.normal(37, 0.5, n),
        'oxygen_saturation': np.random.normal(97, 2, n),
        'glucose': np.random.normal(100, 20, n),
        'creatinine': np.random.normal(1.0, 0.3, n),
        'white_blood_cells': np.random.normal(7, 2, n)
    }
    df = pd.DataFrame(d)

    risk = (
        (df['age'] > 70).astype(float) * 0.3 +
        (df['heart_rate'] > 100).astype(float) * 0.2 +
        (df['blood_pressure_sys'] > 140).astype(float) * 0.2 +
        (df['glucose'] > 120).astype(float) * 0.15 +
        np.random.random(n) * 0.15
    )
    df['mortality'] = (risk > 0.5).astype(int)

    out = project_root / 'data' / 'raw' / 'medical' / 'synthetic'
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'synthetic_medical_data.csv'
    df.to_csv(csv_path, index=False)
    mort_rate = df['mortality'].mean() * 100
    print(f"Medical data generated: {len(df)} rows, mortality rate={mort_rate:.1f}%")
    print(f"Saved to: {csv_path}")
    return df


if __name__ == '__main__':
    print("=" * 60)
    print("Generating synthetic datasets for Medical experiments")
    print("=" * 60)
    print()
    generate_medical()
    print()
    print("Done! You can now run experiments 05-08.")
