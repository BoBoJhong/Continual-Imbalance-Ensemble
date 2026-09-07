import pandas as pd

from scripts.analysis.validate_result_artifacts import audit_results


def test_result_audit_profiles_schemas_and_issues(tmp_path):
    pd.DataFrame(
        {
            "method": ["a", "a"],
            "seed": [42, 42],
            "AUC": [0.8, 0.8],
        }
    ).to_csv(tmp_path / "valid.csv", index=False)
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")

    report = audit_results(tmp_path)

    assert report["csv_files"] == 2
    assert report["total_rows"] == 2
    assert report["schema_count"] == 1
    assert report["files_with_exact_duplicates"] == 1
    assert report["files_with_seed_column"] == 1
    assert report["issues"] == [{"path": "empty.csv", "issue": "empty_csv"}]
