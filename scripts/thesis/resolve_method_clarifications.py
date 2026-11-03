"""Bind official data verification, validation-only summaries and local hardware.

Reads downloaded official evidence and existing results; never retrains or selects
an algorithm on Test. Tables are descriptive evidence for a fixed XGBoost design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from export_fixed_statistics import longtable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis/drafts/final_method_clarifications_20261027"
GENERATED = ROOT / "thesis/latex/generated/method-clarifications-20261027"
BASELINES = [
    ("XGBoost", "results/phase1_baseline/xgb/rerun_20260406_gridplus_n24/tuned/bankruptcy_year_splits_xgb_raw.csv"),
    ("LightGBM", "results/phase1_baseline/lgbm/bk_lgb_tuned48_20260421/tuned/bankruptcy_year_splits_lgb_raw.csv"),
    ("Random Forest", "results/phase1_baseline/random_forest/tuned/bankruptcy_year_splits_rf_raw.csv"),
    ("Logistic Regression", "results/phase1_baseline/logistic_regression/bk_lr_tuned4_20260421/tuned/bankruptcy_year_splits_lr_raw.csv"),
    ("TabM", "results/phase1_baseline/tabm/tuned_light_full_20260410/tuned/bankruptcy_year_splits_tabm_raw.csv"),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    proof_path = OUT / "clarification-proof.json"
    if args.check:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        for mapping in (proof["source_sha256"], proof["generated_sha256"]):
            for name, expected in mapping.items():
                assert sha(ROOT / name) == expected, f"Clarification evidence changed: {name}"
        print("PASS: official versions, fixed status target, validation summaries and hardware evidence agree")
        return
    GENERATED.mkdir(parents=True, exist_ok=True)
    sources = {}

    def remember(p):
        sources[p.relative_to(ROOT).as_posix()] = sha(p)

    local = ROOT / "data/raw/bankruptcy/american_bankruptcy_dataset.csv"
    public = OUT / "official-public-data.csv"
    original = OUT / "official-2022-data.csv"
    assert sha(local) == sha(public)
    current = pd.read_csv(local)
    old = pd.read_csv(original).rename(columns={"year": "fyear"})
    assert current.company_name.nunique() == old.company_name.nunique() == 8971
    assert len(current) == len(old) == 78682
    fields = ["company_name", "fyear", "status_label"]
    assert current[fields].sort_values(fields[:2]).reset_index(drop=True).equals(old[fields].sort_values(fields[:2]).reset_index(drop=True))
    assert current.groupby("company_name").status_label.nunique().eq(1).all()
    comparison = json.loads((OUT / "official-data-comparison.json").read_text(encoding="utf-8"))
    assert comparison["official_sha256"] == sha(local)
    for name in ["official-public-data.csv", "official-2022-data.csv", "official-data-comparison.json", "official-data-history.json",
                 "official-2022-comparison.json", "official-paper-archive.zip", "hardware.json"]:
        remember(OUT / name)
    remember(local)
    remember(Path(__file__).resolve())
    remember(ROOT / "scripts/thesis/export_fixed_statistics.py")
    with zipfile.ZipFile(OUT / "official-paper-archive.zip") as z:
        header = z.read("dataset_paper/financial_train.csv").decode("utf-8").splitlines()[0].split(",")
    names = [c.removeprefix("3_") for c in header if c.startswith("3_")]
    assert names == ["current_assets", "total_assets", "cost_of_goods_sold", "total_long_term_debt", "depreciation_and_amortization", "ebit", "ebitda", "gross_profit", "inventory", "total_current_liabilities", "net_income", "retained_earnings", "total_receivables", "total_revenue", "market_value", "total_liabilities", "net_sales", "total_operating_expenses"]
    rows, summaries, parameters = [], [], []
    expected_splits = {f"split_{i}+{16-i}" for i in range(1, 16)}
    strategies = {"none", "undersampling", "oversampling", "hybrid"}
    for label, relative in BASELINES:
        path = ROOT / relative
        remember(path)
        d = pd.read_csv(path)
        d = d[d.method.isin(["Old", "New", "Retrain"])].copy()
        assert not d.duplicated(["split", "method", "sampling"]).any()
        assert set(d.loc[d.method.eq("Old"), "split"]) == expected_splits
        assert set(d.loc[d.method.eq("New"), "split"]) == expected_splits
        assert len(d) == 124
        parts = {}
        valid_count = 0
        for method in ("Old", "New", "Retrain"):
            group = d[d.method.eq(method)]
            assert set(group.sampling) == strategies
            if "tune_val_auc" not in group or group.tune_val_auc.isna().any():
                parts[method] = None
            else:
                valid_count += len(group)
                # Same weight for each boundary, then each sampling condition.
                parts[method] = float(group.groupby("split").tune_val_auc.mean().mean())
            summaries.append(dict(candidate=label, method=method, rows=len(group), validation_auc=parts[method], source=relative))
        if "tune_best_params" in d:
            for r in d.itertuples():
                parameters.append(dict(candidate=label, split=r.split, method=r.method, sampling=r.sampling,
                                       validation_auc=r.tune_val_auc, selected_parameters=r.tune_best_params, source=relative))
        rows.append([label, str(valid_count)] + ["未保存" if parts[m] is None else f"{parts[m]:.4f}" for m in ("Old", "New", "Retrain")])
    pd.DataFrame(summaries).to_csv(OUT / "baseline-validation-summary.csv", index=False)
    pd.DataFrame(parameters).to_csv(OUT / "baseline-selected-parameters.csv", index=False)
    (GENERATED / "baseline-validation.tex").write_text(longtable(
        "Baseline 已保存之驗證 AUC 紀錄彙整", "tab:baseline-validation-evidence",
        ["分類器", "有效驗證筆數", "Old 平均", "New 平均", "Retrain 平均"], ["0.29", "0.16", "0.16", "0.16", "0.16"], rows,
        "資料來源：本研究既有 Baseline 驗證紀錄整理；未讀取 Test 指標作選型。Old／New 各 60 筆（15 邊界乘四採樣），Retrain 僅四筆，不複製為 15 次。不同版本搜尋預算、前處理與驗證來源未完全對齊；本表為描述性核對，不作公平演算法排名。未保存不表示指標為零。"), encoding="utf-8")
    hardware = json.loads((OUT / "hardware.json").read_text(encoding="utf-8-sig"))
    assert int(hardware["installed_ram_bytes"]) == 34359738368
    cpu = hardware["cpu"][0]
    hardware_rows = [
        ["CPU", "Intel Core i7-10870H，2.20 GHz；8 核心、16 邏輯處理器"],
        ["記憶體", "32 GB（安裝容量）"],
        ["作業系統", "Windows 11 家用版，64 位元；版本 10.0.26200"],
        ["獨立顯示卡", "NVIDIA GeForce GTX 1650 Ti；4096 MiB 顯示記憶體"],
        ["內建顯示晶片", "Intel UHD Graphics"],
        ["本機 NVIDIA 驅動", "592.82（本次查核值）"],
        ["固定年份 XGBoost", "CPU；hist 樹建構，未設定 CUDA device"],
        ["FS 執行", "scikit-learn 決策樹／MI 與 LightGBM CPU 設定；最多四執行緒"],
        ["查核日期", "2026 年 10 月 27 日"],
    ]
    (GENERATED / "hardware.tex").write_text(longtable(
        "本研究本機設備與固定年份執行設定", "tab:method-hardware", ["項目", "規格／設定"], ["0.25", "0.69"], hardware_rows,
        "資料來源：依使用者指定，採本機系統查核與實驗程式設定整理。設備具有 GPU 不代表所有方法都使用 GPU；本次查得的作業系統及驅動版本不回推為每次歷史執行的原始版本。"), encoding="utf-8")
    proof = dict(status="passed", official_commit=comparison["official_commit"], local_matches_official_byte_for_byte=True,
                 original_and_updated_public_company_count=8971, reported_paper_company_count=8262,
                 company_count_resolution="inconsistency in author descriptions versus both public releases; not caused by local processing or company-count changes in 2024; no author mapping to 8262 available",
                 target="company-level status repeated across all observed years; no event date supplied; not independent next-year event labels",
                 source_financial_header_order=names, classifier_position="fixed XGBoost base learner; not validation-selected best algorithm",
                 classifier_selected_on_test=False, source_sha256=sources,
                 generated_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in GENERATED.iterdir() if p.is_file()})
    proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "validation_summary": summaries, "generated": str(GENERATED)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
