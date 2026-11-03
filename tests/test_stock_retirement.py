"""Retiring stock must leave bankruptcy/medical workflows usable."""
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from experiments._shared import common_dataset
from experiments.phase1_baseline import retrain
from scripts.legacy.reports import generate_advisor_excel
from src.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['stock', 'stock_spx', 'stock_dji', 'stock_ndx'])
def test_retired_dataset_rejected_before_reading_data(name, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail('Retired dataset must be rejected before any data read')

    monkeypatch.setattr(pd, 'read_csv', forbidden)
    with pytest.raises(ValueError, match='Unknown dataset'):
        DataLoader(tmp_path).load_dataset(name)
    with pytest.raises(ValueError, match='不支援'):
        common_dataset.get_splits(name, Mock())


def test_retrain_dispatches_only_retained_domains(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(retrain, 'get_logger', lambda *a, **k: Mock())
    monkeypatch.setattr(retrain, 'set_seed', lambda *a: None)
    monkeypatch.setattr(retrain, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(retrain, 'get_bankruptcy_splits', lambda *a, **k: tuple(range(6)))

    def medical_only(name, *args, **kwargs):
        assert name == 'medical'
        return tuple(range(6))

    monkeypatch.setattr(retrain, 'get_splits', medical_only)
    monkeypatch.setattr(retrain, '_run_and_save', lambda name, *a: saved.append(name))
    retrain.main()
    assert saved == ['bankruptcy', 'medical']


def test_legacy_report_filters_mixed_historical_rows_without_modifying_source(monkeypatch):
    source = pd.DataFrame({'dataset': ['bankruptcy', 'stock', 'Medical'], 'AUC': [0.8, 0.6, 0.7]})
    before = source.copy(deep=True)
    monkeypatch.setattr(pd, 'read_csv', lambda *a, **k: source)
    retained = generate_advisor_excel.read_csv_safe('historical.csv')
    assert retained['dataset'].tolist() == ['bankruptcy', 'Medical']
    pd.testing.assert_frame_equal(source, before)


def test_active_config_and_removed_assets_agree():
    config = yaml.safe_load((ROOT / 'config/base_config.yaml').read_text(encoding='utf-8'))
    assert set(config['datasets']) == {'bankruptcy', 'medical'}
    ledger = json.loads((ROOT / 'docs/REMOVED_STOCK_20261012.json').read_text(encoding='utf-8'))
    # Cache names may reappear if Python compiles a historical source snapshot;
    # check dedicated data, entrypoints and experiment output files.
    for entry in ledger['files']:
        if entry['category'] != 'cache':
            assert not (ROOT / entry['path']).exists(), entry['path']
