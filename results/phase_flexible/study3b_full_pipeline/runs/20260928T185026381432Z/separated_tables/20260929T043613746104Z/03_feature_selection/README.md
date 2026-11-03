# 特徵選擇

本快照納入完整完成的 3/10 個 seeds：42, 123, 456。Test 特徵年 2009–2018；每個方法、每個 seed 共 33,636 筆 Test 預測。

所有表皆為已完成 seeds 的平均。Pooled 先於每個 seed 合併年度預測計算指標，再取 seed 平均；年度平均則先對每個 seed 的年度指標等權平均，再取 seed 平均。二者分表，不將重複 seed 視為新的公司樣本。CSV 另保留 seed 標準差與各 seed 數值；標準差不是信賴區間。本次為執行中快照，不能當作十種子最終結論。

配置依名稱排列，不以 Test 排名重新選模型。FS 的基礎集成可能隨 seed／年度不同，FS 表比較的是同一年度已選配置上的 FS 規則。

## 選定集成的 FS 對照（含 no-FS）

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `SelectedEnsemble_cart_r50` | 0.8261 | 0.1131 | 0.1916 | 0.4372 | 0.1926 | 0.2014 |
| `SelectedEnsemble_cart_r80` | 0.8210 | 0.1073 | 0.1784 | 0.4303 | 0.1869 | 0.1783 |
| `SelectedEnsemble_mi_r50` | 0.8278 | 0.0981 | 0.1774 | 0.4420 | 0.1972 | 0.1612 |
| `SelectedEnsemble_mi_r80` | 0.8219 | 0.1005 | 0.1902 | 0.4564 | 0.2111 | 0.1852 |
| `SelectedEnsemble_none_r100` | 0.8249 | 0.0982 | 0.1937 | 0.4294 | 0.1857 | 0.2029 |
| `SelectedEnsemble_rfe_r50` | 0.8274 | 0.1038 | 0.1803 | 0.4220 | 0.1799 | 0.1899 |
| `SelectedEnsemble_rfe_r80` | 0.8208 | 0.1051 | 0.1934 | 0.4511 | 0.2053 | 0.1899 |
| `SelectedEnsemble_shap_r50` | 0.8344 | 0.1152 | 0.1871 | 0.4535 | 0.2076 | 0.1716 |
| `SelectedEnsemble_shap_r80` | 0.8223 | 0.1085 | 0.1878 | 0.4397 | 0.1949 | 0.1838 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `SelectedEnsemble_cart_r50` | 0.8628 | 0.1803 | 0.1938 | 0.3945 | 0.1893 | 0.2974 |
| `SelectedEnsemble_cart_r80` | 0.8632 | 0.1742 | 0.1715 | 0.3896 | 0.1875 | 0.2976 |
| `SelectedEnsemble_mi_r50` | 0.8666 | 0.1607 | 0.1631 | 0.3909 | 0.1925 | 0.2462 |
| `SelectedEnsemble_mi_r80` | 0.8662 | 0.1696 | 0.1844 | 0.4147 | 0.2077 | 0.2699 |
| `SelectedEnsemble_none_r100` | 0.8668 | 0.1728 | 0.1761 | 0.3779 | 0.1841 | 0.2777 |
| `SelectedEnsemble_rfe_r50` | 0.8624 | 0.1735 | 0.1739 | 0.3714 | 0.1782 | 0.2684 |
| `SelectedEnsemble_rfe_r80` | 0.8622 | 0.1722 | 0.1913 | 0.4229 | 0.2038 | 0.2474 |
| `SelectedEnsemble_shap_r50` | 0.8688 | 0.1821 | 0.1772 | 0.4032 | 0.2027 | 0.2425 |
| `SelectedEnsemble_shap_r80` | 0.8635 | 0.1816 | 0.1821 | 0.3905 | 0.1895 | 0.2802 |

[Pooled CSV](fs_comparison/pooled_mean.csv) · [年度平均 CSV](fs_comparison/annual_mean.csv) · [逐年數據](fs_comparison/by_year.csv)

## Validation 再選擇 FS／no-FS 策略

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `ValidationSelectedFS` | 0.8269 | 0.0986 | 0.1891 | 0.4128 | 0.1719 | 0.2152 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `ValidationSelectedFS` | 0.8686 | 0.1774 | 0.1700 | 0.3674 | 0.1741 | 0.2899 |

[Pooled CSV](validation_selected/pooled_mean.csv) · [年度平均 CSV](validation_selected/annual_mean.csv) · [逐年數據](validation_selected/by_year.csv)

