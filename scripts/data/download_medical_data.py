"""
立即可用的醫療資料集下載腳本
無需申請，可直接開始實驗的替代方案
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.datasets import load_breast_cancer

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_logger


def download_diabetes_dataset():
    """
    下載 Diabetes 130-US Hospitals 資料集
    來源: UCI Machine Learning Repository
    """
    logger = get_logger("MedicalDataDownloader", console=True, file=False)
    
    logger.info("="*60)
    logger.info("下載 Diabetes 資料集")
    logger.info("="*60)
    
    # 使用 kaggle API 下載
    import os
    os.system("kaggle datasets download -d brandao/diabetes-130-us-hospitals-for-years-1999-2008")
    
    # 解壓
    import zipfile
    zip_file = "diabetes-130-us-hospitals-for-years-1999-2008.zip"
    
    if Path(zip_file).exists():
        output_dir = Path("data/raw/medical/diabetes")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        logger.info(f"✓ 資料已解壓到: {output_dir}")
        
        # 刪除 zip 檔
        Path(zip_file).unlink()
        
        # 讀取並檢查資料
        data_file = output_dir / "diabetic_data.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            logger.info(f"✓ 資料大小: {df.shape}")
            logger.info(f"✓ 目標變數: readmitted")
            
            # 創建二元分類目標
            df['readmitted_binary'] = (df['readmitted'] != 'NO').astype(int)
            logger.info(f"✓ 再入院率: {df['readmitted_binary'].mean()*100:.2f}%")
            
            return df
    
    return None


def download_heart_disease_uci():
    """
    下載 UCI Heart Disease 資料集
    經典的醫療分類資料集
    """
    logger = get_logger("MedicalDataDownloader", console=True, file=False)
    
    logger.info("="*60)
    logger.info("下載 Heart Disease 資料集")
    logger.info("="*60)
    
    # 使用 kaggle 下載
    import os
    os.system("kaggle datasets download -d johnsmith88/heart-disease-dataset")
    
    # 解壓
    import zipfile
    zip_file = "heart-disease-dataset.zip"
    
    if Path(zip_file).exists():
        output_dir = Path("data/raw/medical/heart_disease")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        logger.info(f"✓ 資料已解壓到: {output_dir}")
        Path(zip_file).unlink()
        
        # 讀取資料
        data_file = output_dir / "heart.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            logger.info(f"✓ 資料大小: {df.shape}")
            logger.info(f"✓ 目標變數: target (心臟病)")
            logger.info(f"✓ 患病率: {df['target'].mean()*100:.2f}%")
            
            return df
    
    return None


def create_synthetic_medical_timeseries():
    """
    創建合成的醫療時間序列資料
    用於快速測試，無需下載
    """
    logger = get_logger("MedicalDataDownloader", console=True, file=False)
    
    logger.info("="*60)
    logger.info("創建合成醫療時間序列資料")
    logger.info("="*60)
    
    np.random.seed(42)
    
    # 模擬 10 年的月度資料
    n_samples = 10 * 12  # 120 months
    
    # 創建時間特徵
    dates = pd.date_range('2010-01', periods=n_samples, freq='ME')  # Monthly End
    
    # 模擬生命徵象
    data = {
        'date': dates,
        'age': np.random.randint(30, 90, n_samples),
        'heart_rate': np.random.normal(75, 12, n_samples),
        'blood_pressure_sys': np.random.normal(120, 15, n_samples),
        'blood_pressure_dia': np.random.normal(80, 10, n_samples),
        'temperature': np.random.normal(37, 0.5, n_samples),
        'oxygen_saturation': np.random.normal(97, 2, n_samples),
        'glucose': np.random.normal(100, 20, n_samples),
        'creatinine': np.random.normal(1.0, 0.3, n_samples),
        'white_blood_cells': np.random.normal(7, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 創建目標變數（死亡率）
    # 基於多個風險因素
    risk_score = (
        (df['age'] > 70).astype(int) * 0.3 +
        (df['heart_rate'] > 100).astype(int) * 0.2 +
        (df['blood_pressure_sys'] > 140).astype(int) * 0.2 +
        (df['glucose'] > 120).astype(int) * 0.15 +
        np.random.random(n_samples) * 0.15
    )
    
    df['mortality'] = (risk_score > 0.5).astype(int)
    
    # 保存
    output_dir = Path("data/raw/medical/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "synthetic_medical_data.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"✓ 資料大小: {df.shape}")
    logger.info(f"✓ 死亡率: {df['mortality'].mean()*100:.2f}%")
    logger.info(f"✓ 資料已保存到: {output_file}")
    
    return df


def main():
    """主函數"""
    print("\n" + "="*60)
    print("醫療資料集下載工具")
    print("="*60)
    print("\n選擇下載方式：\n")
    print("1. Diabetes 130-US Hospitals (Kaggle) - 需要 Kaggle API")
    print("2. Heart Disease UCI (Kaggle) - 需要 Kaggle API")
    print("3. 創建合成資料 (推薦快速測試) ✨")
    print("\n推薦: 選項 3 - 無需下載，立即可用\n")
    
    choice = input("請選擇 (1/2/3，或直接按 Enter 選擇 3): ").strip() or "3"
    
    if choice == "1":
        download_diabetes_dataset()
    elif choice == "2":
        download_heart_disease_uci()
    elif choice == "3":
        df = create_synthetic_medical_timeseries()
        
        print("\n" + "="*60)
        print("✅ 合成資料創建完成！")
        print("="*60)
        print("\n資料特點:")
        print(f"  - 樣本數: {len(df)}")
        print(f"  - 特徵數: {df.shape[1] - 2}")  # 扣除 date 和 target
        print(f"  - 時間跨度: 2010-2019 (10年)")
        print(f"  - 死亡率: {df['mortality'].mean()*100:.2f}%")
        print("\n位置:")
        print("  - data/raw/medical/synthetic/synthetic_medical_data.csv")
        print("\n下一步:")
        print("  使用 DataLoader 載入並開始實驗！")
        print("="*60)
    
    else:
        print("無效選擇")


if __name__ == "__main__":
    main()
