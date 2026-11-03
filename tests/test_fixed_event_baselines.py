"""Baseline candidates are fixed per classifier, start from the default, and sampling stays the only imbalance step."""

from experiments.phase1_baseline import bankruptcy_fixed_event_baselines as baselines


def test_candidates_are_reproducible_and_start_with_default():
    for classifier in baselines.CLASSIFIERS:
        first, second = baselines.candidates(classifier, 8), baselines.candidates(classifier, 8)
        assert first == second and first[0] == {} and len(first) == 8
        assert len({str(sorted(c.items())) for c in first}) == 8
    features = {c["max_features"] for c in baselines.candidates("rf", 24)[1:]}
    assert all(f in ("sqrt", "log2") or isinstance(f, float) for f in features) and any(isinstance(f, float) for f in features)


def test_tuned_parameters_reach_the_fitted_model_without_class_weights():
    xgb = baselines.build("xgb", {"max_depth": 3, "n_estimators": 50}, 7, 1)
    assert xgb.params["max_depth"] == 3 and "scale_pos_weight" not in xgb.params
    lgb = baselines.build("lgb", {"n_estimators": 40, "num_leaves": 31}, 7, 1)
    assert lgb._num_boost_round == 40 and "is_unbalance" not in lgb.params
    rf = baselines.build("rf", {"max_depth": 8}, 7, 1)
    assert rf.model.max_depth == 8 and rf.model.class_weight is None
    lr = baselines.build("lr", {"C": 0.1}, 7, 1)
    assert lr.model.C == 0.1 and lr.model.class_weight is None
