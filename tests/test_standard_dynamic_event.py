"""Event-target entry: same rows as the reference run, one event per failed company, mature labels."""

import pytest

from experiments.phase2_ensemble.dynamic.des import bankruptcy_standard_dynamic as study
from experiments.phase2_ensemble.dynamic.des import bankruptcy_standard_dynamic_event as event


def test_event_target_keeps_reference_rows_and_marks_only_final_failed_year():
    reference, _ = study.load_data()
    data, features, manifest = event.load_data()
    assert data.index.equals(reference.index) and features == [f"X{i}" for i in range(1, 19)]
    assert int(data.target.sum()) == manifest["n_positive"] == reference.loc[reference.target.eq(1)].company_name.nunique()
    positives = data.loc[data.target.eq(1)]
    assert not positives.company_name.duplicated().any()
    last_year = reference.groupby("company_name").fyear.max()
    assert (positives.fyear.to_numpy() == last_year.loc[positives.company_name].to_numpy()).all()
    assert reference.loc[positives.index, "status_label"].eq("failed").all()


def test_partitions_pass_label_maturity_and_late_labels_fail():
    data, _, _ = event.load_data()
    parts = study.partition_data(data)
    maturity = event.check_label_maturity(parts)
    assert maturity["Old_fit"] == 2012 and maturity["New_fit"] == 2014 and maturity["DSEL"] == 2015
    late = dict(parts, Calibration=parts["Calibration"].assign(label_available_year=2016))
    with pytest.raises(ValueError, match="Calibration labels"):
        event.check_label_maturity(late)
