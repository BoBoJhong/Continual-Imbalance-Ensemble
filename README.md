# Continual-Imbalance-Ensemble

A framework for handling class imbalance in non-stationary datasets (Bankruptcy, Stock, Medical) using Dynamic Ensemble Selection (DES) and Hybrid Sampling.

## 🎯 Research Objective

This project addresses the performance degradation of prediction models in non-stationary environments with class imbalance by proposing a continual learning framework that combines:

- **Dynamic Ensemble Selection (DES)** with KNORA-E
- **Hybrid Sampling** with SMOTEENN
- **Feature Selection** impact analysis

## 📊 Datasets

1. **Bankruptcy Prediction** - Taiwan Economic Journal (1999-2018)
2. **Medical Time Series** - MIMIC-III ICU mortality prediction
3. **Stock Market Crash** - Stock market crash prediction dataset

## 🏗️ Project Structure

```
Continual-Imbalance-Ensemble/
├── config/                 # Configuration (YAML)
├── src/                    # Core library (data, models, utils)
├── experiments/            # Experiment scripts 01~08 + common_*.py
├── scripts/                # Run-all, compare, multi-seed
├── results/                # Experiment outputs (CSV)
├── docs/                   # Documentation (STRUCTURE.md, EXPERIMENT_CHECKLIST.md)
├── examples/               # Demos
└── tests/                  # Unit tests
```

詳細目錄與檔案說明見 **docs/STRUCTURE.md**。

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

All configurations are in `config/` directory:

- `base_config.yaml` - Basic settings
- `model_config.yaml` - Model hyperparameters
- `sampling_config.yaml` - Sampling strategies
- `des_config.yaml` - Ensemble configurations
- `feature_config.yaml` - Feature selection
- `experiment_config.yaml` - Experiment setup

### Running Experiments

**一鍵跑完所有實驗（推薦）：**

```bash
python scripts/run_all_experiments.py
```

**或手動逐個執行：**

```bash
python experiments/01_bankruptcy_baseline.py
python experiments/02_bankruptcy_ensemble.py
python experiments/03_bankruptcy_des.py
python experiments/04_bankruptcy_feature_selection_study.py
python experiments/05_stock_baseline_ensemble.py
python experiments/06_medical_baseline_ensemble.py
python experiments/07_stock_des.py
python experiments/08_medical_des.py
```

**看結果彙總：**

```bash
python scripts/compare_baseline_ensemble.py   # Bankruptcy 合併表
python scripts/compare_all_results.py          # 三資料集彙總
```

完整步驟與切割說明見 **EXECUTION_GUIDE.md**。

## 📈 Experiment Design

### Data Splitting

- **Mode A**: Chronological (1999-2011 historical, 2012-2014 new, 2015-2018 test)
- **Mode B**: 5-fold block CV

### Model Pool

**Old Models** (trained on historical data):

1. Model 1: Undersampling
2. Model 2: Oversampling (ADASYN)
3. Model 3: Hybrid (SMOTEENN)

**New Models** (trained on new operating data):
4. Model 4: Undersampling
5. Model 5: Oversampling (ADASYN)
6. Model 6: Hybrid (SMOTEENN)

### Baselines

- Re-training: Combine historical + new data
- Fine-tuning: Pretrain on historical, finetune on new data

### Ensemble Combinations

- 2 models (Old + New pairs)
- 3 models (2 Old + 1 New / 1 Old + 2 New)
- 4, 5, 6 models

## 📊 Evaluation Metrics

- **Primary**: AUC-ROC, F1-Score, G-Mean
- **Secondary**: Recall, Precision, Balanced Accuracy
- **Statistical Test**: Wilcoxon signed-rank test

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: LightGBM, XGBoost
- **Imbalanced Learning**: imbalanced-learn
- **DES**: deslib
- **Feature Selection**: mRMR, LASSO

## 📝 Research Contributions

1. Systematic comparison of DES in imbalanced continual learning
2. Validation of SMOTEENN effectiveness in time series
3. Optimal ensemble combination strategy (Old + New models)
4. Feature selection impact on ensemble models

## 👥 Authors

- Your Name - Master's Thesis Research

## 📄 License

This project is for academic research purposes.

## 🙏 Acknowledgments

- Advisor guidance
- Dataset sources: Kaggle, PhysioNet, UCI Repository
