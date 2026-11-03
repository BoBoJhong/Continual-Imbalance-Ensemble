"""Guard against mixing excluded methods or replicating full-data references."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/thesis'))
from export_chapter_results import aggregate_baseline, emphasized, group_maxima


def baseline_frame():
    rows = []
    for method, count, score in [('Old', 15, .2), ('New', 15, .6), ('Retrain', 1, .4)]:
        for i in range(1, count+1):
            for sampling in ['none', 'undersampling', 'oversampling', 'hybrid']:
                rows.append(dict(method=method, split=f'split_{i}+{16-i}', sampling=sampling,
                                 AUC=score, F1=score, Recall=score))
    return pd.DataFrame(rows)


def test_retrain_is_not_repeated_and_finetune_is_excluded():
    d = baseline_frame()
    excluded = dict(method='Finetune', split='split_1+15', sampling='none', AUC=1., F1=1., Recall=1.)
    overall, by_sampling = aggregate_baseline(pd.concat([d, pd.DataFrame([excluded])]))
    assert [r['n_rows'] for r in overall] == [60, 60, 4]
    assert [r['AUC'] for r in overall] == pytest.approx([.2, .6, .4])
    assert all(r['n_boundaries'] == 1 for r in by_sampling if r['method'] == 'Retrain')


def test_missing_boundary_fails_instead_of_averaging_an_incomplete_grid():
    with pytest.raises(ValueError, match='Incomplete Baseline'):
        aggregate_baseline(baseline_frame().iloc[1:])


def test_duplicate_saved_result_fails_instead_of_overweighting_it():
    d = baseline_frame()
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate_baseline(pd.concat([d, d.iloc[:1]]))


def test_highlighting_keeps_displayed_ties_and_does_not_bold_lower_values():
    assert emphasized(.8000001, .8000002) == r'\textbf{0.8000}'
    assert emphasized(.7998, .8000002) == '0.7998'


def test_highlighting_compares_within_a_classifier_not_across_versions():
    rows = [dict(candidate='A', method='Old', AUC=.6, F1=.1, Recall=.8),
            dict(candidate='A', method='New', AUC=.7, F1=.2, Recall=.5),
            dict(candidate='B', method='Old', AUC=.9, F1=.3, Recall=.9)]
    best = group_maxima(rows, ['candidate'])
    assert best[('A',)] == dict(AUC=.7, F1=.2, Recall=.8)
    assert emphasized(rows[1]['AUC'], best[('A',)]['AUC']) == r'\textbf{0.7000}'
