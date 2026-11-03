"""Numerical checks for cluster multiplicities, probability ties and pairing."""
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score

spec = importlib.util.spec_from_file_location("fixed_stats", Path(__file__).resolve().parents[1] / "scripts/analysis/fixed_company_statistics.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def test_weighted_auc_ties_and_threshold_match_independent_reference():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.5, 0.5, 0.8, 0.5, 0.9])
    cluster = np.array([0, 0, 1, 1, 2, 3])
    weights = np.array([[1, 1, 1, 1], [2, 1, 0, 1], [0, 2, 2, 0]], dtype=np.int16)
    inputs = stats.sort_inputs(y, p, cluster, 0.5)
    actual = stats.weighted_scores(weights, *(a[None, :] for a in inputs))[0]
    for b, w in enumerate(weights):
        row_weights = w[cluster]
        expected = [roc_auc_score(y, p, sample_weight=row_weights),
                    f1_score(y, p >= 0.5, sample_weight=row_weights),
                    recall_score(y, p >= 0.5, sample_weight=row_weights)]
        np.testing.assert_allclose(actual[b], expected, atol=1e-14)


def test_paired_identical_predictions_have_zero_differences():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.5, 0.8])
    inputs = stats.sort_inputs(y, p, np.arange(4), 0.5)
    w = np.array([[1, 1, 1, 1], [2, 1, 0, 1]], dtype=np.int16)
    actual = stats.weighted_scores(w, *(np.stack([a, a]) for a in inputs))
    np.testing.assert_array_equal(actual[0] - actual[1], 0)


def test_holm_is_monotone_in_sorted_p_and_preserves_order():
    np.testing.assert_allclose(stats.holm([0.03, 0.001, 0.02, 0.9]), [0.06, 0.004, 0.06, 0.9])
