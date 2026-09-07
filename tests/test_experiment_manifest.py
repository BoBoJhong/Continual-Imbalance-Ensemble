from scripts.run.run_all_experiments import EXPERIMENTS, PROJECT_ROOT, _validate_manifest


def test_maintained_experiment_manifest_contains_only_existing_unique_files():
    paths = _validate_manifest(list(EXPERIMENTS))

    assert len(paths) == len(set(paths))
    assert all(path.is_file() for path in paths)
    assert all(path.is_relative_to(PROJECT_ROOT) for path in paths)
