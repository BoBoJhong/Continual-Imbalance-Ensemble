"""
範例：完整的資料處理和模型訓練流程
演示如何使用所有已建立的模組
"""

import sys
from pathlib import Path

# 將專案根目錄加入 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, f1_score

from src.utils import set_seed, get_logger
from src.data import DataPreprocessor, DataSplitter, ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool


def create_demo_data():
    """創建示範資料集（模擬不平衡時間序列）"""
    print("="*60)
    print("步驟 1: 創建示範資料集")
    print("="*60)
    
    # 創建不平衡分類資料
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],  # 90% vs 10% 不平衡
        random_state=42
    )
    
    # 轉換為 DataFrame
    feature_names = [f"feature_{i}" for i in range(20)]
    X = pd.DataFrame(X, columns=feature_names)
    
    # 添加時間列（模擬 2010-2019）
    X['Year'] = np.repeat(range(2010, 2020), 100)
    
    y = pd.Series(y, name='target')
    
    print(f"✓ 資料集大小: {len(X)} 樣本, {X.shape[1]} 特徵")
    print(f"✓ 類別分布: {y.value_counts().to_dict()}")
    print(f"✓ 不平衡比例: {y.value_counts().min() / y.value_counts().max():.2f}")
    
    return X, y


def split_data(X, y):
    """切割資料為 Historical / New Operating / Testing"""
    print("\n" + "="*60)
    print("步驟 2: 切割資料（時間序列）")
    print("="*60)
    
    splitter = DataSplitter()
    
    splits = splitter.chronological_split(
        X, y,
        time_column="Year",
        historical_end=2015,      # 2010-2015: Historical
        new_operating_end=2017    # 2016-2017: New Operating
    )                             # 2018-2019: Testing
    
    print(f"✓ Historical: {len(splits['historical'][0])} 樣本")
    print(f"✓ New Operating: {len(splits['new_operating'][0])} 樣本")
    print(f"✓ Testing: {len(splits['testing'][0])} 樣本")
    
    return splits


def preprocess_data(splits):
    """前處理資料"""
    print("\n" + "="*60)
    print("步驟 3: 資料前處理")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    
    X_hist, y_hist = splits['historical']
    X_new, y_new = splits['new_operating']
    X_test, y_test = splits['testing']
    
    # 移除時間列
    X_hist = X_hist.drop(columns=['Year'])
    X_new = X_new.drop(columns=['Year'])
    X_test = X_test.drop(columns=['Year'])
    
    # 標準化
    X_hist_scaled, X_test_scaled = preprocessor.scale_features(X_hist, X_test, fit=True)
    X_new_scaled, _ = preprocessor.scale_features(X_new, fit=False)
    
    print("✓ 特徵標準化完成")
    
    return (X_hist_scaled, y_hist), (X_new_scaled, y_new), (X_test_scaled, y_test)


def train_single_model(X_train, y_train, X_test, y_test):
    """訓練單一模型"""
    print("\n" + "="*60)
    print("步驟 4: 訓練單一模型")
    print("="*60)
    
    # 創建採樣器
    sampler = ImbalanceSampler()
    
    # 使用 SMOTEENN 處理不平衡
    X_resampled, y_resampled = sampler.apply_sampling(
        X_train, y_train.values,
        strategy="hybrid"  # SMOTEENN
    )
    
    # 訓練 LightGBM
    model = LightGBMWrapper(name="demo_model")
    model.fit(X_resampled, y_resampled)
    
    # 預測
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # 評估
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"✓ AUC-ROC: {auc:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")
    
    return model


def train_model_pool(X_hist, y_hist, X_new, y_new, X_test, y_test):
    """訓練模型池"""
    print("\n" + "="*60)
    print("步驟 5: 訓練模型池 (3 Old + 3 New)")
    print("="*60)
    
    # 創建 Old Models Pool
    old_pool = ModelPool(pool_name="old_models")
    old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    
    # 創建 New Models Pool
    new_pool = ModelPool(pool_name="new_models")
    new_pool.create_pool(X_new, y_new.values, prefix="new")
    
    # 獲取所有預測
    old_predictions = old_pool.predict_proba(X_test)
    new_predictions = new_pool.predict_proba(X_test)
    
    print("\n✓ Old Models 預測:")
    for name, proba in old_predictions.items():
        auc = roc_auc_score(y_test, proba)
        print(f"  - {name}: AUC = {auc:.4f}")
    
    print("\n✓ New Models 預測:")
    for name, proba in new_predictions.items():
        auc = roc_auc_score(y_test, proba)
        print(f"  - {name}: AUC = {auc:.4f}")
    
    return old_pool, new_pool


def main():
    """主函數"""
    print("\n" + "="*60)
    print("Continual-Imbalance-Ensemble 示範")
    print("="*60)
    
    # 設定隨機種子
    set_seed(42)
    
    # 1. 創建示範資料
    X, y = create_demo_data()
    
    # 2. 切割資料
    splits = split_data(X, y)
    
    # 3. 前處理
    (X_hist, y_hist), (X_new, y_new), (X_test, y_test) = preprocess_data(splits)
    
    # 4. 訓練單一模型
    single_model = train_single_model(X_hist, y_hist, X_test, y_test)
    
    # 5. 訓練模型池
    old_pool, new_pool = train_model_pool(X_hist, y_hist, X_new, y_new, X_test, y_test)
    
    print("\n" + "="*60)
    print("🎉 示範完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 下載真實資料集 (Bankruptcy, Stock, Medical)")
    print("2. 使用 notebooks/ 進行資料探索")
    print("3. 調整 config/ 中的參數")
    print("4. 開始正式實驗！")
    print("="*60)


if __name__ == "__main__":
    main()
