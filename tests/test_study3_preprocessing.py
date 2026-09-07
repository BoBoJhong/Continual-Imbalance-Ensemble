import numpy as np
import pandas as pd

from experiments.phase4_drift.bankruptcy_drift_stream import _preprocess
from experiments.phase4_drift.bankruptcy_ross_fs_static_ensemble import _apply_old_fit_fs


class _SilentLogger:
    def info(self, *_args, **_kwargs):
        pass


def test_preprocess_uses_training_mean_for_validation_and_test():
    train = pd.DataFrame({"a": [1.0, 3.0], "b": [10.0, 14.0]})
    validation = pd.DataFrame({"a": [np.nan, 1000.0], "b": [np.nan, 1000.0]})
    test = pd.DataFrame({"a": [np.nan], "b": [np.nan]})

    _, validation_scaled, test_scaled, scaler = _preprocess(train, validation, test)

    assert scaler.training_fill_values_.to_dict() == {"a": 2.0, "b": 12.0}
    np.testing.assert_allclose(validation_scaled.iloc[0].to_numpy(), [0.0, 0.0])
    np.testing.assert_allclose(test_scaled.iloc[0].to_numpy(), [0.0, 0.0])


def test_feature_selection_fit_excludes_old_validation_tail():
    old = pd.DataFrame(
        {
            "stable_signal": [0.0, 1.0, 0.2, 0.8, 999.0, 999.0],
            "tail_only_signal": [5.0, 5.1, 4.9, 5.0, 0.0, 1.0],
        }
    )
    y_old = np.array([0, 1, 0, 1, 0, 1])
    new = pd.DataFrame({"stable_signal": [0.0], "tail_only_signal": [5.0]})
    test = pd.DataFrame({"stable_signal": [1.0], "tail_only_signal": [5.0]})

    old_fs, new_fs, test_fs, n_selected, _ = _apply_old_fit_fs(
        old,
        y_old,
        new,
        test,
        _SilentLogger(),
        n_old_val=2,
    )

    assert n_selected == 1
    assert old_fs.columns.tolist() == ["stable_signal"]
    assert new_fs.columns.tolist() == ["stable_signal"]
    assert test_fs.columns.tolist() == ["stable_signal"]
