"""
Generate synthetic Stock and Medical datasets for experiments 05-08.
Run: python scripts/generate_synthetic_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent


def generate_stock():
    """Generate synthetic stock market data (2000 samples, S&P500-style)."""
    np.random.seed(42)
    n = 2000
    dates = pd.date_range('2000-01-03', periods=n, freq='B')
    price = 1000.0
    prices = [price]
    for _ in range(n - 1):
        price = price * (1 + np.random.normal(0.0003, 0.012))
        prices.append(price)

    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)

    returns = pd.Series(close).pct_change().fillna(0).values
    log_ret = np.log(pd.Series(close) / pd.Series(close).shift(1)).fillna(0).values

    def sma(x, w): return pd.Series(x).rolling(w, min_periods=1).mean().values
    def vol(x, w): return pd.Series(x).rolling(w, min_periods=1).std().fillna(0.01).values
    def rsi(x, w=14):
        d = pd.Series(x).diff()
        g = d.clip(lower=0).rolling(w, min_periods=1).mean()
        l_loss = (-d.clip(upper=0)).rolling(w, min_periods=1).mean()
        rs = g / (l_loss + 1e-9)
        return (100 - 100 / (1 + rs)).values

    future_ret = pd.Series(close).pct_change(20).shift(-20).fillna(0).values
    crash = (future_ret < -0.05).astype(int)
    # Concept drift: later period has higher crash rate
    crash[1500:] = (future_ret[1500:] < -0.03).astype(int)

    df = pd.DataFrame({
        'Date': dates,
        'Close': close, 'High': high, 'Low': low, 'Open': open_, 'Volume': volume,
        'Returns': returns, 'Log_Returns': log_ret,
        'SMA_5': sma(close, 5), 'SMA_20': sma(close, 20), 'SMA_60': sma(close, 60),
        'Volatility_20': vol(returns, 20),
        'RSI': rsi(returns),
        'Future_Returns_20': future_ret,
        'Crash_Event': crash
    })

    out = project_root / 'data' / 'raw' / 'stock'
    out.mkdir(parents=True, exist_ok=True)
    # Write 2-line header then data (skiprows=2 format)
    csv_path = out / 'stock_data.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('# Synthetic SP500-style stock data\n')
        f.write('# Generated for research purposes\n')
    df.to_csv(csv_path, mode='a', index=False, header=True)
    crash_rate = crash.mean() * 100
    print(f"Stock data generated: {len(df)} rows, crash rate={crash_rate:.1f}%")
    print(f"Saved to: {csv_path}")
    return df


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
    print("Generating synthetic datasets for Stock and Medical experiments")
    print("=" * 60)
    generate_stock()
    print()
    generate_medical()
    print()
    print("Done! You can now run experiments 05-08.")
