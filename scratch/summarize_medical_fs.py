import pandas as pd
import numpy as np
from pathlib import Path

root = Path(r"c:\0_git workspace\Continual-Imbalance-Ensemble")
static_path = root / "results/phase3_feature/xgb_medical_fs_full_static.csv"
des_path = root / "results/phase3_feature/xgb_medical_fs_full_des.csv"

def summarize():
    print("=== Medical Feature Selection Analysis ===")
    
    # 1. Static Ensemble Analysis (All_6)
    df = pd.read_csv(static_path)
    all6 = df[df['ensemble'] == 'All_6']
    
    no_fs = all6[all6['fs'] == 'no_fs'].set_index('split')['AUC']
    mi_fs = all6[all6['fs'] == 'mutual_info_r80'].set_index('split')['AUC']
    
    diff = mi_fs - no_fs
    
    summary_static = pd.DataFrame({
        'No_FS_AUC': no_fs,
        'MI_FS_AUC': mi_fs,
        'Diff': diff
    })
    
    print("\n[Static All_6 Ensemble Comparison]")
    print(summary_static.to_string())
    print(f"\nAverage Improvement: {diff.mean():.6f}")
    
    # 2. Dynamic Ensemble Analysis (KNORA-U)
    df_des = pd.read_csv(des_path)
    knora = df_des[df_des['ensemble'] == 'Dynamic_KNORA_U']
    
    no_fs_des = knora[knora['fs'] == 'no_fs'].set_index('split')['AUC']
    mi_fs_des = knora[knora['fs'] == 'mutual_info_r80'].set_index('split')['AUC']
    
    diff_des = mi_fs_des - no_fs_des
    
    summary_des = pd.DataFrame({
        'No_FS_AUC': no_fs_des,
        'MI_FS_AUC': mi_fs_des,
        'Diff': diff_des
    })
    
    print("\n[Dynamic KNORA-U Ensemble Comparison]")
    print(summary_des.to_string())
    print(f"\nAverage Improvement: {diff_des.mean():.6f}")

    # Save to a new summary file
    summary_static.to_csv(root / "results/phase3_feature/xgb_medical_fs_summary_diff.csv")

if __name__ == "__main__":
    summarize()
