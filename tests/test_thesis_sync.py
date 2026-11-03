"""Source binding and version safety of the thesis export pipeline."""
import importlib.util
import json
from pathlib import Path

import pytest

FILE = Path(__file__).resolve().parents[1] / "scripts/thesis/sync_results.py"
SPEC = importlib.util.spec_from_file_location("thesis_sync", FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def workspace(tmp_path):
    run = tmp_path / "results/example/runs/run1"
    (run / "seed_42").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed", "protocol": "test_protocol", "config": {"seeds": [42]}
    }))
    (run / "seed_42/complete.json").write_text('{"status":"completed"}')
    (run / "summary.csv").write_text("method,AUC_mean,AUC_std\nmodel_one,0.5,0.1\n")
    config = {
        "tables": [{"id": "example", "run": "results/example/runs/run1",
                    "expected_protocol": "test_protocol", "expected_seeds": [42],
                    "complete_pattern": "seed_{seed}/complete.json", "csv": "summary.csv",
                    "filters": {}, "row_key": "method", "caption": "Example", "context": "Fixture",
                    "metrics": [{"label": "AUC", "mean": "AUC_mean", "std": "AUC_std"}]}]
    }
    (tmp_path / "config").mkdir()
    (tmp_path / "config/thesis_sources.json").write_text(json.dumps(config))
    return tmp_path, run, config


def test_changed_source_requires_new_snapshot_and_preserves_old(workspace):
    root, run, _ = workspace
    first = MODULE.synchronize(root)
    generated = root / "thesis/latex/generated"
    original_table = generated / first["snapshot"] / "example.tex"
    original_bytes = original_table.read_bytes()
    MODULE.synchronize(root, check=True)
    (run / "summary.csv").write_text("method,AUC_mean,AUC_std\nmodel_one,0.6,0.1\n")
    with pytest.raises(ValueError, match="sources changed"):
        MODULE.synchronize(root, check=True)
    second = MODULE.synchronize(root)
    assert first["snapshot"] != second["snapshot"]
    assert original_table.read_bytes() == original_bytes
    assert "0.6000" in (generated / second["snapshot"] / "example.tex").read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["incomplete", "protocol", "seeds", "metric", "escape"])
def test_reject_invalid_sources_before_creating_output(workspace, failure):
    root, run, config = workspace
    if failure == "incomplete":
        (run / "seed_42/complete.json").write_text('{"status":"running"}')
    elif failure == "protocol":
        config["tables"][0]["expected_protocol"] = "wrong"
    elif failure == "seeds":
        config["tables"][0]["expected_seeds"] = [1, 2, 3]
    elif failure == "metric":
        (run / "summary.csv").write_text("method,AUC_mean,AUC_std\nmodel_one,nan,0.1\n")
    elif failure == "escape":
        config["tables"][0]["run"] = "results/../../outside"
    (root / "config/thesis_sources.json").write_text(json.dumps(config))
    with pytest.raises((ValueError, OSError)):
        MODULE.synchronize(root)
    assert not (root / "thesis/latex/generated").exists()


def test_detect_manual_edit_of_generated_table(workspace):
    root, _, _ = workspace
    result = MODULE.synchronize(root)
    table = root / "thesis/latex/generated" / result["snapshot"] / "example.tex"
    table.write_text("manually changed")
    with pytest.raises(ValueError, match="snapshot has changed"):
        MODULE.synchronize(root, check=True)


@pytest.mark.parametrize("changed", ["source", "generated", "proof"])
@pytest.mark.parametrize("binding_key", ["method_inventory", "method_clarifications", "chapter_results", "chapter_figures", "deep_baseline_reference"])
def test_method_inventory_rejects_changed_evidence(workspace, changed, binding_key):
    root, _, config = workspace
    source = root / "raw-method-data.csv"
    generated = root / "method-table.tex"
    source.write_text("company,year\nA,2014\n")
    generated.write_text("Verified method table")
    proof = root / "method-inventory.json"
    proof.write_text(json.dumps({
        "status": "passed", "source_sha256": {source.name: MODULE.sha(source.read_bytes())},
        "generated_sha256": {generated.name: MODULE.sha(generated.read_bytes())}
    }))
    config[binding_key] = {"proof": proof.name, "sha256": MODULE.sha(proof.read_bytes())}
    (root / "config/thesis_sources.json").write_text(json.dumps(config))
    MODULE.synchronize(root)
    {"source": source, "generated": generated, "proof": proof}[changed].write_text("changed evidence")
    with pytest.raises(ValueError, match="(Method inventory|inventory source)"):
        MODULE.synchronize(root, check=True)


@pytest.mark.parametrize("failure", [None, "tampered_archive", "prediction_override", "unknown_source"])
def test_analysis_snapshot_preserves_historical_code_and_rejects_invalid_overrides(workspace, failure):
    root, _, config = workspace
    live = root / "scripts/analysis.py"
    live.parent.mkdir(parents=True)
    live.write_text("new event-label support")
    archive = root / "thesis/drafts/review/analysis.py"
    archive.parent.mkdir(parents=True)
    archive.write_text("original analysis")
    relative = live.relative_to(root).as_posix()
    source_key = "predictions.csv" if failure == "prediction_override" else relative
    directory = root / "results/statistical_tests/example"
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "completed", "specification": {"protocol": "stats_test"},
        "source_sha256": {source_key: MODULE.sha(archive.read_bytes())}, "output_sha256": {}}))
    config["analyses"] = [{"id": "example", "directory": directory.relative_to(root).as_posix(),
                           "expected_protocol": "stats_test", "manifest_sha256": MODULE.sha(manifest.read_bytes()),
                           "source_snapshots": {"unknown.py" if failure == "unknown_source" else source_key:
                                                archive.relative_to(root).as_posix()}}]
    (root / "config/thesis_sources.json").write_text(json.dumps(config))
    if failure == "tampered_archive":
        archive.write_text("changed old analysis")
    if failure:
        with pytest.raises(ValueError, match="(source snapshot|archived analysis|prediction source changed)"):
            MODULE.synchronize(root)
        assert not (root / "thesis/latex/generated").exists()
    else:
        state = MODULE.synchronize(root)
        assert archive.relative_to(root).as_posix() in state["source_sha256"]
        assert relative not in state["source_sha256"]
        MODULE.synchronize(root, check=True)
