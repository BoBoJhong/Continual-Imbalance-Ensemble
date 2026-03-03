from pathlib import Path
p = Path("experiments/phase5_analysis/stock_threshold_cost.py")
raw = p.read_bytes()
# Replace \"\"\" (5c 22 5c 22 5c 22) with """ (22 22 22)
fixed = raw.replace(b'\\"\\"\\"', b'"""')
p.write_bytes(fixed)
print("Fixed:", sum(1 for i in range(len(raw)-2) if raw[i:i+3]==b'\\"\\"\\"'), "occurrences")
print("First 30 bytes now:", list(p.read_bytes()[:30]))
