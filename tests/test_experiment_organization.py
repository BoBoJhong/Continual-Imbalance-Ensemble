"""Entrypoint independence and archived path contracts after organization."""
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = json.loads((ROOT / "experiments/legacy/MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))


def test_fixed_year_entry_can_import_without_flexible_controllers():
    # Prevent accidentally reintroducing a dependency on either annual controller.
    code = """
import sys
sys.modules['experiments.phase_flexible.study3b_full_pipeline'] = None
sys.modules['experiments.phase_flexible.rolling_bankruptcy_overlap_ensemble'] = None
from experiments.phase2_ensemble.dynamic.des import bankruptcy_standard_dynamic as study
assert len([name for name in study.static_candidates() if name.startswith('Static_')]) == 49
study.make_model('xgb', 42, 1)
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True, timeout=45)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("entry", MIGRATION["entries"], ids=lambda entry: entry["original_path"])
def test_archived_entry_resolves_repository_and_retains_result_root(entry):
    original = ROOT / entry["original_path"]
    archived = ROOT / entry["archived_path"]
    assert not original.exists()
    assert archived.is_file()
    namespace = runpy.run_path(str(archived), run_name="legacy_import_check")
    assert namespace["project_root"].resolve() == ROOT
    if "OUTPUT_DIR" in namespace:
        assert namespace["OUTPUT_DIR"].is_relative_to(ROOT / "results")
