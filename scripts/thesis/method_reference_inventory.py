"""Audit thesis methods against saved experiment sources; export missing inventories.

Reads the raw data and completed run, never trains models or alters saved results.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

from export_fixed_statistics import longtable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis/drafts/method_reference_revision_20261027"
GENERATED = ROOT / "thesis/latex/generated/method-inventory-20261027"
RUN = ROOT / "results/phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8-sig")).body if isinstance(n, ast.FunctionDef)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    proof_path = OUT / "inventory.json"
    if args.check:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        for name, value in proof["source_sha256"].items():
            assert sha(ROOT / name) == value, f"Method evidence changed: {name}"
        for name, value in proof["generated_sha256"].items():
            assert sha(ROOT / name) == value, f"Method table changed: {name}"
        print("PASS: dataset counts, FS settings, archived/current method bodies and thesis inventories match")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    sources = {}

    def remember(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)

    manifest = json.loads((RUN / "manifest.json").read_text())
    raw_path = ROOT / "data/raw/bankruptcy/american_bankruptcy_dataset.csv"
    assert sha(raw_path) == manifest["provenance"]["raw_sha256"]
    data = pd.read_csv(raw_path)
    features = [f"X{i}" for i in range(1, 19)]
    y = data.status_label.eq("failed")
    records = dict(rows=len(data), companies=data.company_name.nunique(), positive_rows=int(y.sum()), negative_rows=int((~y).sum()),
                   positive_percent=100 * y.mean(), imbalance_ratio=(~y).sum() / y.sum(),
                   first_year=int(data.fyear.min()), last_year=int(data.fyear.max()), n_features=18,
                   missing_features=int(data[features].isna().sum().sum()), nonfinite_features=int((~np.isfinite(data[features].to_numpy())).sum()),
                   duplicate_company_years=int(data.duplicated(["company_name", "fyear"]).sum()))
    remember(raw_path)
    remember(RUN / "manifest.json")
    remember(Path(__file__).resolve())
    remember(ROOT / "scripts/thesis/export_fixed_statistics.py")
    snapshot_dir = ROOT / "experiments/legacy/source_snapshots/20261012"
    old_fixed = snapshot_dir / "bankruptcy_standard_dynamic.py.txt"
    old_model = snapshot_dir / "study3b_full_pipeline.py.txt"
    assert sha(old_fixed) == manifest["provenance"]["sources"][r"experiments\phase2_ensemble\dynamic\des\bankruptcy_standard_dynamic.py"]
    assert sha(old_model) == manifest["provenance"]["sources"][r"experiments\phase_flexible\study3b_full_pipeline.py"]
    current_fixed = ROOT / "experiments/phase2_ensemble/dynamic/des/bankruptcy_standard_dynamic.py"
    current_model = ROOT / "experiments/_shared/common_modeling.py"
    fixed_names = ("load_data", "partition_data", "fit_representation", "select_threshold", "static_candidates", "evaluate_predictions")
    same = {name: functions(old_fixed)[name] == functions(current_fixed)[name] for name in fixed_names}
    same.update({name: functions(old_model)[name] == functions(current_model)[name] for name in ("make_model", "feature_columns")})
    assert all(same.values()), same
    for p in (old_fixed, old_model, current_fixed, current_model):
        remember(p)
    annual = data.assign(positive=y.astype(int)).groupby("fyear").agg(rows=("positive", "size"), positive=("positive", "sum"))
    annual["positive_percent"] = annual.positive / annual.rows * 100
    annual.to_csv(OUT / "annual_counts.csv")
    rows = [[str(int(year)), f"{int(r.rows):,}", f"{int(r.positive):,}", f"{r.positive_percent:.2f}\\%"] for year, r in annual.iterrows()]
    (GENERATED / "annual-data.tex").write_text(longtable(
        "各年度公司財務紀錄與破產標籤分布", "tab:method-annual-data", ["年度", "紀錄數", "破產標籤紀錄數", "破產標籤比例"],
        ["0.18", "0.23", "0.25", "0.27"], rows,
        "資料來源：本研究依實際使用資料整理。比例以公司年度紀錄計算，並非該年度新增破產事件率。"), encoding="utf-8")
    fs_rows, frequency_csv = [], []
    for fs_method in ("mi", "cart", "shap", "rfe"):
        for ratio in (50, 80):
            counts = Counter()
            for seed in manifest["config"]["seeds"]:
                path = RUN / f"seed_{seed}" / f"{fs_method}_r{ratio}" / "audit.json"
                audit = json.loads(path.read_text())
                assert audit["selector_fit_scope"] == "Old_fit_only"
                assert len(audit["selected_features"]) == (9 if ratio == 50 else 14)
                counts.update(audit["selected_features"])
                remember(path)
            method = fs_method.upper() + f" {ratio}\\%"
            fs_rows.append([method] + [str(counts[x]) for x in features])
            frequency_csv += [dict(method=fs_method, ratio=ratio, feature=x, selected_seeds=counts[x], n_seeds=10) for x in features]
    pd.DataFrame(frequency_csv).to_csv(OUT / "feature_selection_frequency.csv", index=False)
    # Split 18 feature columns into two readable tables rather than shrinking text.
    for block, start in ((1, 1), (2, 10)):
        subset = [[r[0]] + r[start:start + 9] for r in fs_rows]
        (GENERATED / f"feature-frequency-{block}.tex").write_text(longtable(
            f"實驗二特徵入選種子數（X{start}--X{start + 8}）", f"tab:method-fs-frequency-{block}",
            ["FS 設定"] + [f"X{i}" for i in range(start, start + 9)], ["0.20"] + ["0.075"] * 9,
            subset, "資料來源：本研究各種子特徵選取結果。數值為十個種子中入選的次數；此表僅描述已選特徵，不據此重新挑選 Test 最佳子集。"), encoding="utf-8")
    defaults = dict(mi_signature=str(inspect.signature(mutual_info_classif)),
                    cart=DecisionTreeClassifier(random_state=42, class_weight="balanced").get_params(),
                    shap=LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, class_weight="balanced", n_jobs=4).get_params())
    proof = dict(status="passed", run=RUN.relative_to(ROOT).as_posix(), records=records, exact_method_ast_matches=same,
                 selector_defaults=defaults, version_scope=manifest["provenance"]["versions"],
                 original_paper_company_count=8262, actual_local_company_count=8971,
                 company_count_discrepancy="author description differs from both public releases (8971 IDs); local file equals current public release; no author mapping to 8262 available",
                 source_sha256=sources, generated_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in GENERATED.iterdir() if p.is_file()})
    proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "records": records, "method_matches": same}, ensure_ascii=False, default=float, indent=2))


if __name__ == "__main__":
    main()
