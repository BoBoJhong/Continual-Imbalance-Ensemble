# Configuration contract

The YAML files in this directory define shared defaults. Experiment-specific
protocol choices may remain in an experiment script when they are part of the
named study design, but reusable model, sampling, seed, and evaluation defaults
must come from configuration.

## Active files

- `base_config.yaml`: project paths, datasets, split modes, and canonical seeds.
- `model_config.yaml`: reusable model hyperparameters.
- `sampling_config.yaml`: under-, over-, and hybrid-sampling parameters.
- `des_config.yaml`: ensemble and dynamic-selection defaults.
- `feature_config.yaml`: shared feature-selection defaults.
- `experiment_config.yaml`: metrics, statistical correction, and output policy.

Load values through the cached loader:

```python
from src.utils import get_config_loader

config = get_config_loader()
seeds = config.get("base_config", "base_config.random_seeds")
sampling = config.get(
    "sampling_config", "sampling_strategies.hybrid.params"
)
```

`src.data.ImbalanceSampler` and the model wrappers consume these shared files.
The comprehensive runner can be validated without executing experiments:

```powershell
python scripts/run/run_all_experiments.py --list
```

The maintained statistical workflow uses Holm family-wise correction.
Legacy output tables may contain raw p-values; they must be identified as such
and must not be described as independently replicated evidence.

## Protocol and result navigation

Use the [experiment index](../docs/實驗索引.md) to identify the intended protocol and
completed run. These YAML files are shared defaults, not a substitute for a run's
manifest: a script may record protocol-specific overrides, and historical runs
used different package environments. Do not infer all old results were produced
with today's defaults.

Shared factories and fit-only FS now live in [common_modeling.py](../experiments/_shared/common_modeling.py). This extraction preserves existing parameter choices; new run provenance includes both shared helper files. Historical run manifests remain immutable.

## Thesis result bindings

[thesis_sources.json](thesis_sources.json) pins the completed runs, expected protocols,
seed sets, source CSV fields and filters used by the thesis exporter. It does not
change experiment configuration or select models by test scores.
[thesis_environment.lock.json](thesis_environment.lock.json) separately pins the TeX toolchain.
[thesis_overleaf.json](thesis_overleaf.json) records the user-selected online target
and its access/settings verification status. Saving the target does not start a sync.
See the [thesis workflow](../docs/thesis_environment/README.md) to regenerate tables
and check whether their source hashes changed.

## Result catalog policy

[result_catalog_config.json](result_catalog_config.json) is a metadata classification policy, separate from model hyperparameters. It declares selected runs, explicit scope rules and category interpretation. The [generator](../scripts/analysis/build_result_catalog.py) checks completion evidence, preserves partial snapshots and tags byte-identical files without deleting or invalidating them. Unmatched files remain pending review; modification time is never used to infer obsolescence.
