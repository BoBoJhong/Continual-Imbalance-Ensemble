"""
src/evaluation/metrics.py

統一評估指標模組。
所有實驗腳本應使用此模組計算指標，避免重複程式碼。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_proba, y_pred=None, threshold: float = 0.5) -> dict:
    """
    計算分類評估指標（適用於類別不平衡資料集）。

    禁止單獨使用 Accuracy（見 .agent/rules.md Rule 4）。
    輸出 AUC-ROC、F1-Score、G-Mean、Recall、Precision。

    Args:
        y_true:     真實標籤（0/1 array-like）
        y_proba:    正類機率（array-like，shape: (n,)）
        y_pred:     預測標籤（若為 None，以 threshold 決定）
        threshold:  分類閾值，預設 0.5

    Returns:
        dict: {AUC, F1, G_Mean, Recall, Precision}
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_pred is None:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred)

    # G-Mean = sqrt(Sensitivity * Specificity)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = float(np.sqrt(sensitivity * specificity))

    # Type 1 Error (False Positive Rate) = FP / (FP + TN)
    type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Type 2 Error (False Negative Rate) = FN / (FN + TP)
    type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "G_Mean": g_mean,
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Type1_Error": type1_error,
        "Type2_Error": type2_error,
    }


def print_results_table(results: dict, title: str = "Results") -> None:
    """
    在終端印出結果對照表。

    Args:
        results:  {method_name: {AUC, F1, G_Mean, Recall, Precision}}
        title:    表格標題
    """
    df = pd.DataFrame(results).T
    # 按 AUC 排序
    if "AUC" in df.columns:
        df = df.sort_values("AUC", ascending=False)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(df.to_string(float_format="{:.4f}".format))
    print(f"{'='*60}\n")


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """
    將 results dict 轉為 DataFrame，方便存 CSV。

    Args:
        results: {method_name: {metric_name: value}}

    Returns:
        pd.DataFrame, index = method names
    """
    return pd.DataFrame(results).T
