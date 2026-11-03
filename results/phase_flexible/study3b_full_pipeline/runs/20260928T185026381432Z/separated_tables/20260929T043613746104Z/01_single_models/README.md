# 單模型與採樣

本快照納入完整完成的 3/10 個 seeds：42, 123, 456。Test 特徵年 2009–2018；每個方法、每個 seed 共 33,636 筆 Test 預測。

所有表皆為已完成 seeds 的平均。Pooled 先於每個 seed 合併年度預測計算指標，再取 seed 平均；年度平均則先對每個 seed 的年度指標等權平均，再取 seed 平均。二者分表，不將重複 seed 視為新的公司樣本。CSV 另保留 seed 標準差與各 seed 數值；標準差不是信賴區間。本次為執行中快照，不能當作十種子最終結論。

配置依名稱排列，不以 Test 排名重新選模型。FS 的基礎集成可能隨 seed／年度不同，FS 表比較的是同一年度已選配置上的 FS 規則。

## Logistic Regression

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_lr_hybrid` | 0.7994 | 0.0588 | 0.1131 | 0.4346 | 0.1926 | 0.0800 |
| `Newest_lr_none` | 0.7171 | 0.0566 | 0.1232 | 0.3965 | 0.1592 | 0.1004 |
| `Newest_lr_over` | 0.8010 | 0.0577 | 0.0999 | 0.4534 | 0.2111 | 0.0654 |
| `Newest_lr_under` | 0.7203 | 0.0581 | 0.1163 | 0.3790 | 0.1453 | 0.0970 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_lr_hybrid` | 0.8147 | 0.0716 | 0.1105 | 0.4165 | 0.1972 | 0.0813 |
| `Newest_lr_none` | 0.7662 | 0.0746 | 0.1271 | 0.3845 | 0.1637 | 0.1296 |
| `Newest_lr_over` | 0.8150 | 0.0714 | 0.0992 | 0.4295 | 0.2128 | 0.0701 |
| `Newest_lr_under` | 0.7668 | 0.0783 | 0.1181 | 0.3640 | 0.1484 | 0.1287 |

[Pooled CSV](lr/pooled_mean.csv) · [年度平均 CSV](lr/annual_mean.csv) · [逐年數據](lr/by_year.csv)

## Random Forest

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_rf_hybrid` | 0.8387 | 0.0419 | 0.0934 | 0.4318 | 0.1926 | 0.0633 |
| `Newest_rf_none` | 0.8419 | 0.1425 | 0.2132 | 0.4528 | 0.2065 | 0.2205 |
| `Newest_rf_over` | 0.8393 | 0.0411 | 0.0914 | 0.4244 | 0.1869 | 0.0616 |
| `Newest_rf_under` | 0.8416 | 0.1438 | 0.2070 | 0.4350 | 0.1903 | 0.2285 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_rf_hybrid` | 0.8427 | 0.0578 | 0.0877 | 0.3917 | 0.1966 | 0.0766 |
| `Newest_rf_none` | 0.8478 | 0.1740 | 0.2107 | 0.4380 | 0.2087 | 0.2910 |
| `Newest_rf_over` | 0.8432 | 0.0581 | 0.0845 | 0.3826 | 0.1879 | 0.0668 |
| `Newest_rf_under` | 0.8471 | 0.1738 | 0.1951 | 0.4060 | 0.1881 | 0.3027 |

[Pooled CSV](rf/pooled_mean.csv) · [年度平均 CSV](rf/annual_mean.csv) · [逐年數據](rf/by_year.csv)

## XGBoost

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_xgb_hybrid` | 0.8020 | 0.0597 | 0.1112 | 0.4203 | 0.1799 | 0.0805 |
| `Newest_xgb_none` | 0.8391 | 0.0954 | 0.1743 | 0.4265 | 0.1834 | 0.1661 |
| `Newest_xgb_over` | 0.7993 | 0.0559 | 0.1109 | 0.4034 | 0.1661 | 0.0845 |
| `Newest_xgb_under` | 0.8450 | 0.0966 | 0.1656 | 0.3712 | 0.1384 | 0.2062 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_xgb_hybrid` | 0.8059 | 0.0796 | 0.1073 | 0.3687 | 0.1724 | 0.1062 |
| `Newest_xgb_none` | 0.8439 | 0.1242 | 0.1580 | 0.3887 | 0.1739 | 0.2529 |
| `Newest_xgb_over` | 0.8037 | 0.0760 | 0.1093 | 0.3702 | 0.1643 | 0.1075 |
| `Newest_xgb_under` | 0.8533 | 0.1219 | 0.1588 | 0.3472 | 0.1313 | 0.2329 |

[Pooled CSV](xgb/pooled_mean.csv) · [年度平均 CSV](xgb/annual_mean.csv) · [逐年數據](xgb/by_year.csv)

## LightGBM

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_lgb_hybrid` | 0.8112 | 0.0586 | 0.1051 | 0.4042 | 0.1684 | 0.0786 |
| `Newest_lgb_none` | 0.8088 | 0.0850 | 0.1690 | 0.3803 | 0.1453 | 0.2019 |
| `Newest_lgb_over` | 0.8096 | 0.0546 | 0.1043 | 0.4066 | 0.1684 | 0.0762 |
| `Newest_lgb_under` | 0.8226 | 0.0867 | 0.1714 | 0.4185 | 0.1765 | 0.1667 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_lgb_hybrid` | 0.8255 | 0.0807 | 0.1047 | 0.3688 | 0.1623 | 0.1063 |
| `Newest_lgb_none` | 0.8599 | 0.1183 | 0.1514 | 0.3538 | 0.1418 | 0.2103 |
| `Newest_lgb_over` | 0.8229 | 0.0768 | 0.0982 | 0.3549 | 0.1617 | 0.0995 |
| `Newest_lgb_under` | 0.8638 | 0.1241 | 0.1712 | 0.4012 | 0.1747 | 0.1948 |

[Pooled CSV](lgb/pooled_mean.csv) · [年度平均 CSV](lgb/annual_mean.csv) · [逐年數據](lgb/by_year.csv)

## MLP

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_mlp_hybrid` | 0.6369 | 0.0332 | 0.0712 | 0.3345 | 0.1142 | 0.0529 |
| `Newest_mlp_none` | 0.6684 | 0.0591 | 0.1046 | 0.3207 | 0.1038 | 0.1155 |
| `Newest_mlp_over` | 0.6374 | 0.0340 | 0.0697 | 0.3229 | 0.1061 | 0.0519 |
| `Newest_mlp_under` | 0.6667 | 0.0569 | 0.0980 | 0.3394 | 0.1165 | 0.0846 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_mlp_hybrid` | 0.6433 | 0.0519 | 0.0770 | 0.3148 | 0.1215 | 0.0720 |
| `Newest_mlp_none` | 0.6846 | 0.0825 | 0.1065 | 0.3041 | 0.1018 | 0.1992 |
| `Newest_mlp_over` | 0.6446 | 0.0501 | 0.0811 | 0.2992 | 0.1101 | 0.0915 |
| `Newest_mlp_under` | 0.6848 | 0.0741 | 0.0984 | 0.3199 | 0.1208 | 0.1213 |

[Pooled CSV](mlp/pooled_mean.csv) · [年度平均 CSV](mlp/annual_mean.csv) · [逐年數據](mlp/by_year.csv)

## SVM

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_svm_hybrid` | 0.7351 | 0.0516 | 0.0899 | 0.3821 | 0.1488 | 0.0645 |
| `Newest_svm_none` | 0.5613 | 0.0177 | 0.0186 | 0.5182 | 0.4602 | 0.0095 |
| `Newest_svm_over` | 0.7332 | 0.0512 | 0.0883 | 0.3746 | 0.1430 | 0.0639 |
| `Newest_svm_under` | 0.5584 | 0.0164 | 0.0182 | 0.5041 | 0.4083 | 0.0093 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `Newest_svm_hybrid` | 0.7508 | 0.0837 | 0.0881 | 0.3620 | 0.1522 | 0.0757 |
| `Newest_svm_none` | 0.5897 | 0.0383 | 0.0480 | 0.2836 | 0.4681 | 0.0466 |
| `Newest_svm_over` | 0.7472 | 0.0827 | 0.0873 | 0.3513 | 0.1453 | 0.0811 |
| `Newest_svm_under` | 0.5869 | 0.0389 | 0.0574 | 0.2835 | 0.4247 | 0.0611 |

[Pooled CSV](svm/pooled_mean.csv) · [年度平均 CSV](svm/annual_mean.csv) · [逐年數據](svm/by_year.csv)

