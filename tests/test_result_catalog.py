"""Evidence-based classification, byte duplicates and artifact preservation."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analysis import build_result_catalog as catalog

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / catalog.POLICY).read_text(encoding="utf-8"))


def put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("status,expected", [("completed", "current_core"), ("failed", "failed_run"),
                                               ("running", "needs_review"), (None, "needs_review")])
def test_selected_run_needs_completion_evidence(tmp_path, status, expected):
    run = POLICY["selected_runs"][0]["path"]
    if status is not None:
        put(tmp_path, run + "/manifest.json", json.dumps({"status": status}))
    put(tmp_path, run + "/seed_42/metrics.csv", "F1\n0.1\n")
    result = catalog.build_catalog(tmp_path, POLICY)
    row = next(r for r in result["files"] if r["path"].endswith("metrics.csv"))
    assert row["category"] == expected


def test_partial_snapshot_does_not_inherit_full_run_completion(tmp_path):
    run = POLICY["selected_runs"][1]["path"]
    put(tmp_path, run + "/manifest.json", '{"status":"completed"}')
    put(tmp_path, run + "/readout_tables/old/summary.csv", "snapshot")
    put(tmp_path, run + "/seed_42/summary.csv", "full")
    result = catalog.build_catalog(tmp_path, POLICY)
    rows = {r["path"]: r for r in result["files"]}
    assert rows[run + "/readout_tables/old/summary.csv"]["category"] == "interim_snapshot"
    assert rows[run + "/seed_42/summary.csv"]["category"] == "current_extension"


def test_exact_duplicates_keep_primary_identity_and_reject_equal_size_false_match(tmp_path):
    run = POLICY["selected_runs"][0]["path"]
    put(tmp_path, run + "/manifest.json", '{"status":"completed"}')
    a = put(tmp_path, run + "/seed_42/predictions.bin", "AB")
    b = put(tmp_path, run + "/seed_123/predictions.bin", "AB")
    c = put(tmp_path, run + "/seed_456/predictions.bin", "CD")
    result = catalog.build_catalog(tmp_path, POLICY)
    group = next(g for g in result["duplicate_groups"] if a.relative_to(tmp_path).as_posix() in g["paths"])
    assert group["paths"] == sorted(p.relative_to(tmp_path).as_posix() for p in (a, b))
    assert group["kind"] == "repeated_seed_outputs"
    rows = {r["path"]: r for r in result["files"]}
    assert all(rows[p.relative_to(tmp_path).as_posix()]["category"] == "current_core" for p in (a, b, c))
    assert "exact_duplicate_bytes" not in rows[c.relative_to(tmp_path).as_posix()]["tags"]


def test_partial_file_does_not_mark_completed_sibling_failed(tmp_path):
    run = POLICY["selected_runs"][0]["path"]
    put(tmp_path, run + "/manifest.json", '{"status":"completed"}')
    put(tmp_path, run + "/model_raw.partial.csv", "partial")
    put(tmp_path, run + "/metrics.csv", "final")
    rows = {r["path"]: r for r in catalog.build_catalog(tmp_path, POLICY)["files"]}
    assert rows[run + "/model_raw.partial.csv"]["category"] == "checkpoint_partial"
    assert rows[run + "/metrics.csv"]["category"] == "current_core"


def test_generation_preserves_sources_and_excludes_all_catalog_snapshots(tmp_path):
    original = put(tmp_path, "results/unreviewed.bin", "original")
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    first = catalog.build_catalog(tmp_path, POLICY, generated_at="fixed-time")
    catalog.write_catalog(first, tmp_path, tmp_path / "results/catalog/snapshot-a")
    second = catalog.build_catalog(tmp_path, POLICY, generated_at="fixed-time")
    assert second == first
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    assert first["file_count"] == 1
    assert sum(c["files"] for c in first["counts"].values()) == 1
    assert first["files"][0]["category"] == "needs_review"
    with pytest.raises(FileExistsError):
        catalog.write_catalog(first, tmp_path, tmp_path / "results/catalog/snapshot-a")
    with pytest.raises(ValueError, match="below results/catalog"):
        catalog.write_catalog(first, tmp_path, tmp_path / "results/overwrite-originals")


def test_policy_refuses_selected_run_outside_results(tmp_path):
    policy = copy.deepcopy(POLICY)
    policy["selected_runs"][0]["path"] = "results/../../outside"
    with pytest.raises(ValueError, match="stay in results"):
        catalog.build_catalog(tmp_path, policy)


def test_historical_link_refresh_preserves_metadata_and_latest_navigation(tmp_path):
    relative = 'results/phase1_baseline/xgb/stock_year_splits_xgb_raw.csv'
    source = put(tmp_path, relative, 'AUC\n0.6\n')
    recorded = catalog.build_catalog(tmp_path, POLICY, generated_at='historical-time')
    snapshot = tmp_path / 'results/catalog/historical'
    catalog.write_catalog(recorded, tmp_path, snapshot)
    json_before = (snapshot / 'catalog.json').read_bytes()
    nav_before = (tmp_path / 'results/catalog/README.md').read_bytes()
    source.unlink()
    put(tmp_path, catalog.REMOVAL_LEDGER, json.dumps({'files': [{'path': relative}]}))
    catalog.write_catalog(recorded, tmp_path, snapshot, refresh_links=True)
    assert (snapshot / 'catalog.json').read_bytes() == json_before
    assert (tmp_path / 'results/catalog/README.md').read_bytes() == nav_before
    rendered = '\n'.join(p.read_text(encoding='utf-8') for p in snapshot.glob('*.md'))
    assert '已移除' in rendered
    assert catalog.REMOVAL_LEDGER.split('/')[-1] in rendered
    assert 'stock_year_splits_xgb_raw.csv)' not in rendered
    assert recorded['file_count'] == 1
    with pytest.raises(ValueError, match='preserve recorded metadata'):
        changed = copy.deepcopy(recorded)
        changed['file_count'] = 0
        catalog.write_catalog(changed, tmp_path, snapshot, refresh_links=True)


def test_result_removal_links_use_retained_copy_and_removed_directory_ledger(tmp_path):
    retired = 'results/retired-study/seed_42'
    duplicate = 'results/old-analysis/summary.csv'
    kept = 'results/kept-analysis/summary.csv'
    put(tmp_path, kept, 'AUC\n0.7\n')
    result_ledger = catalog.REMOVAL_LEDGERS[1]
    put(tmp_path, result_ledger, json.dumps({
        'files': [{'path': duplicate, 'replacement_path': kept}],
        'removed_directories': ['results/retired-study'],
    }))
    base = tmp_path / 'results/catalog/snapshot'
    linked = catalog.link(duplicate, base, tmp_path)
    assert 'kept-analysis/summary.csv)' in linked
    assert 'old-analysis/summary.csv)' not in linked
    removed = catalog.link(retired, base, tmp_path)
    assert '已移除' in removed
    assert 'REMOVED_RESULTS_20261012.json)' in removed
