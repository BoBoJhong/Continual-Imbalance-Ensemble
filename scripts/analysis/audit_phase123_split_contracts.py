"""Bankruptcy-only split audit; does not train models or rewrite experiment outputs."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase2_ensemble.dynamic.des import bankruptcy_standard_dynamic as study

RUN = ROOT / "results/phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z"
def default_audit_output(at=None):
    at = at or datetime.now(timezone.utc)
    return ROOT / "docs/audits" / at.strftime("%Y-%m") / (
        "phase123_split_audit_" + at.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
MODELS = {"xgb": "xgb", "lgbm": "lgb", "random_forest": "random_forest",
          "logistic_regression": "logistic_regression", "tabm": "tabm", "tabicl": "tabicl"}
RAW_STEMS = {"random_forest": "rf", "logistic_regression": "lr"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_functions(path):
    """Run only the existing split helpers, excluding model imports/training."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_split_fit_val", "_split_fit_val_by_year"}]
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(ast.Module(body=future + selected, type_ignores=[]))
    namespace = {"np": np, "pd": pd, "train_test_split": train_test_split}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_split_fit_val_by_year"]


def main(output=None):
    output = output or default_audit_output()
    data, features = study.load_data()
    parts = study.partition_data(data)
    saved = pd.read_csv(RUN / "partitions.csv").set_index("source_row_id", drop=False)
    assert saved.index.is_unique and set(saved.index) == set(data.index)
    for column in ("company_name", "fyear", "target"):
        pd.testing.assert_series_equal(saved.loc[data.index, column], data[column], check_names=False)
    summary = []
    for name, frame in parts.items():
        assert set(saved.index[saved.partition.eq(name)]) == set(frame.index)
        summary.append({"partition": name, "rows": len(frame), "positive": int(frame.target.sum()),
                        "first_year": int(frame.fyear.min()), "last_year": int(frame.fyear.max())})
    assert not set(parts["DSEL"].company_name) & set(parts["Calibration"].company_name)
    manifest = json.loads((RUN / "manifest.json").read_text())
    source_matches = {name: digest(ROOT / name) == wanted
                      for name, wanted in manifest["provenance"]["sources"].items()}
    raw_matches = digest(study.US_CSV) == manifest["provenance"]["raw_sha256"]
    assert raw_matches and all(source_matches.values())
    split_checks = []
    for folder, script_name in MODELS.items():
        helper = split_functions(ROOT / f"experiments/phase1_baseline/bankruptcy_year_splits_{script_name}.py")
        cases, shared_year_cases = 0, 0
        for old_end in range(1999, 2014):
            masks = [data.fyear.le(old_end), data.fyear.gt(old_end) & data.fyear.le(2014),
                     data.fyear.between(2015, 2018)]
            ids = [set(data.index[mask]) for mask in masks]
            assert not any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3))
            assert set.union(*ids) == set(data.index)
            for mask in masks[:2]:
                group = data.loc[mask]
                X = pd.DataFrame({"row_id": group.index.to_numpy()})
                fit, fit_y, val, val_y = helper(X, group.target.to_numpy(), group.fyear.to_numpy())
                fit_ids, val_ids = set(fit.row_id), set(val.row_id)
                assert not fit_ids & val_ids
                assert fit_ids | val_ids == set(group.index)
                np.testing.assert_array_equal(fit_y, data.loc[fit.row_id, "target"].to_numpy())
                np.testing.assert_array_equal(val_y, data.loc[val.row_id, "target"].to_numpy())
                fit_years, val_years = set(data.loc[list(fit_ids), "fyear"]), set(data.loc[list(val_ids), "fyear"])
                shared_year_cases += bool(fit_years & val_years)
                cases += 1
        split_checks.append({"model": folder, "checked_fit_validation_cases": cases,
                             "fit_validation_row_overlap": False, "shared_year_cases": shared_year_cases,
                             "scope": "current split helper on actual raw rows; historical execution not reconstructed"})
    baseline_files = []
    for folder, script_name in MODELS.items():
        stem = RAW_STEMS.get(script_name, script_name)
        for path in sorted((ROOT / "results/phase1_baseline" / folder).rglob(f"bankruptcy_year_splits_{stem}_raw.csv")):
            rows = pd.read_csv(path)
            values = rows[["AUC", "F1"]].apply(pd.to_numeric, errors="coerce").to_numpy()
            baseline_files.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                                   "rows": len(rows), "splits": int(rows.split.nunique()),
                                   "methods": sorted(rows.method.unique().tolist()),
                                   "finite_AUC_F1_rows": int(np.isfinite(values).all(axis=1).sum()),
                                   "duplicate_keys": int(rows.duplicated(["split", "method", "sampling"]).sum())})
    company_labels = data.groupby("company_name").target.nunique()
    audit = {"at": datetime.now(timezone.utc).isoformat(), "status": "share_with_caveats",
             "scope": "bankruptcy Phase 1-3 only",
             "raw_hash_matches_manifest": raw_matches, "source_hash_matches_manifest": source_matches,
             "standard_partitions": summary, "saved_membership_reproduced": True,
             "DSEL_calibration_company_overlap": False, "baseline_split_checks": split_checks,
             "baseline_result_inventory": baseline_files,
             "bankruptcy_all_companies_have_constant_final_status": bool(company_labels.eq(1).all()),
             "limits": ["does not independently retrain baseline models or certify historical execution",
                        "does not verify real company event dates or financial-data publication dates",
                        "split correctness alone does not establish full experimental validity"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k not in
                      {"baseline_result_inventory", "source_hash_matches_manifest", "limits"}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args().output)
