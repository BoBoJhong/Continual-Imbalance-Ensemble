"""
Continual-Imbalance-Ensemble
A framework for handling class imbalance in non-stationary datasets using
Dynamic Ensemble Selection and Hybrid Sampling.

Submodules
----------
src.data        : DataLoader, DataPreprocessor, DataSplitter, ImbalanceSampler
src.models      : LightGBMWrapper, XGBoostWrapper, ModelPool
src.ensemble    : DynamicEnsembleSelector, EnsembleCombiner
src.features    : FeatureSelector
src.evaluation  : compute_metrics, print_results_table, results_to_dataframe
src.utils       : get_logger, set_seed, get_config_loader
"""

__version__ = "0.2.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
