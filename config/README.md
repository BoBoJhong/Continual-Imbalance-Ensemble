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

The current confirmatory statistical workflow uses Holm family-wise correction.
Legacy output tables may contain raw p-values; they must be identified as such
and must not be described as independently replicated evidence.
