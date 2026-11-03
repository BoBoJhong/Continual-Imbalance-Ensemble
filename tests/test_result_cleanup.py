"""Cleanup must preserve referenced sources and usable reporting paths."""
import hashlib
import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from experiments.phase1_baseline import bankruptcy_year_splits_tabm as tabm

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('labels', [None, ['split_1+15']])
def test_tabm_report_helper_finishes_without_cli_global(monkeypatch, tmp_path, labels):
    seen = []
    cleared = []
    # Restore globals redirected by the helper after this test.
    for name in ('OUTPUT_DIR', 'CHECKPOINT_PATH', 'PARTIAL_RAW_PATH'):
        monkeypatch.setattr(tabm, name, getattr(tabm, name))
    monkeypatch.setattr(tabm, '_clear_progress_files', lambda logger: cleared.append(True))
    monkeypatch.setattr(tabm, 'get_bankruptcy_year_split', lambda *a, **k: tuple(range(9)))

    def fake_split(label, *args, **kwargs):
        all_rows = args[9]
        for method in ['Old', 'New', 'Retrain']:
            for sampling in tabm.SAMPLING_STRATEGIES:
                all_rows.append({'split': label, 'method': method, 'sampling': sampling, 'AUC': 0.5})

    monkeypatch.setattr(tabm, 'run_split', fake_split)
    monkeypatch.setattr(tabm, 'format_tables', lambda df, logger, **k: seen.append(k['split_labels_filter']))
    monkeypatch.setattr(tabm, 'export_compact_report', lambda *a: None)
    tabm._run_one_output_dir(
        tmp_path, split_iter=[('split_1+15', 1999)], use_tuning=False,
        n_tune_iter=1, device='cpu', use_amp=False, use_resume=False,
        tune_max_epochs=1, tune_patience=1, final_max_epochs=1, final_patience=1,
        batch_size=1, predict_batch_size=1, val_batch_size=1,
        logger=Mock(), split_labels_filter=labels,
    )
    assert seen == [labels]
    assert len(pd.read_csv(tmp_path / 'bankruptcy_year_splits_tabm_raw.csv')) == 12
    assert len(cleared) == 2


def test_archived_dated_audit_fails_before_any_data_read(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('Missing historical inputs must be explained before reading data')

    monkeypatch.setattr(pd, 'read_csv', forbidden)
    path = ROOT / 'scripts/legacy/analysis/audit_thesis_convergence.py'
    module = runpy.run_path(str(path), run_name='historical_import_check')
    assert module['ROOT'] == ROOT
    with pytest.raises(SystemExit, match='Historical 2026-09-21 audit'):
        module['main']()


def test_duplicate_cleanup_keeps_identical_replacement_and_critical_sources():
    ledger = json.loads((ROOT / 'docs/REMOVED_RESULTS_20261012.json').read_text(encoding='utf-8'))
    for entry in ledger['files']:
        assert not (ROOT / entry['path']).exists()
        if entry['action'].startswith('duplicate'):
            replacement = ROOT / entry['replacement_path']
            assert hashlib.sha256(replacement.read_bytes()).hexdigest() == entry['sha256']
    for entry in ledger['code_migrations']:
        assert not (ROOT / entry['original_path']).exists()
        assert hashlib.sha256((ROOT / entry['source_snapshot']).read_bytes()).hexdigest() == entry['original_sha256']
    # Current chart baseline sources include old CSVs; removal must preserve them.
    for relative in ledger['protected_hashes']:
        assert (ROOT / relative).is_file()
