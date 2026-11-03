# 集成

本快照納入完整完成的 3/10 個 seeds：42, 123, 456。Test 特徵年 2009–2018；每個方法、每個 seed 共 33,636 筆 Test 預測。

所有表皆為已完成 seeds 的平均。Pooled 先於每個 seed 合併年度預測計算指標，再取 seed 平均；年度平均則先對每個 seed 的年度指標等權平均，再取 seed 平均。二者分表，不將重複 seed 視為新的公司樣本。CSV 另保留 seed 標準差與各 seed 數值；標準差不是信賴區間。本次為執行中快照，不能當作十種子最終結論。

配置依名稱排列，不以 Test 排名重新選模型。FS 的基礎集成可能隨 seed／年度不同，FS 表比較的是同一年度已選配置上的 FS 規則。

## 跨年度 FIFO

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `FIFO_lgb_hybrid` | 0.8139 | 0.0464 | 0.1011 | 0.4078 | 0.1696 | 0.0720 |
| `FIFO_lgb_none` | 0.8330 | 0.1040 | 0.1646 | 0.4263 | 0.1834 | 0.1493 |
| `FIFO_lgb_over` | 0.8107 | 0.0451 | 0.0943 | 0.3835 | 0.1499 | 0.0696 |
| `FIFO_lgb_under` | 0.8383 | 0.1007 | 0.1768 | 0.4305 | 0.1869 | 0.1677 |
| `FIFO_lr_hybrid` | 0.8023 | 0.0700 | 0.1196 | 0.4033 | 0.1649 | 0.0938 |
| `FIFO_lr_none` | 0.7161 | 0.0602 | 0.1154 | 0.3561 | 0.1280 | 0.1051 |
| `FIFO_lr_over` | 0.8034 | 0.0699 | 0.1193 | 0.4142 | 0.1742 | 0.0907 |
| `FIFO_lr_under` | 0.7169 | 0.0615 | 0.1185 | 0.3655 | 0.1349 | 0.1057 |
| `FIFO_mlp_hybrid` | 0.6599 | 0.0327 | 0.0576 | 0.2876 | 0.0842 | 0.0450 |
| `FIFO_mlp_none` | 0.6999 | 0.0760 | 0.1287 | 0.3499 | 0.1234 | 0.1364 |
| `FIFO_mlp_over` | 0.6522 | 0.0301 | 0.0513 | 0.3072 | 0.0969 | 0.0353 |
| `FIFO_mlp_under` | 0.7038 | 0.0755 | 0.1218 | 0.3416 | 0.1176 | 0.1266 |
| `FIFO_rf_hybrid` | 0.8472 | 0.0395 | 0.0746 | 0.3581 | 0.1315 | 0.0525 |
| `FIFO_rf_none` | 0.8606 | 0.1509 | 0.1856 | 0.4637 | 0.2180 | 0.1654 |
| `FIFO_rf_over` | 0.8460 | 0.0390 | 0.0734 | 0.3675 | 0.1384 | 0.0501 |
| `FIFO_rf_under` | 0.8659 | 0.1532 | 0.2118 | 0.4805 | 0.2330 | 0.1945 |
| `FIFO_svm_hybrid` | 0.7384 | 0.0476 | 0.0899 | 0.3732 | 0.1419 | 0.0659 |
| `FIFO_svm_none` | 0.5703 | 0.0258 | 0.0184 | 0.5211 | 0.5063 | 0.0094 |
| `FIFO_svm_over` | 0.7356 | 0.0466 | 0.0871 | 0.3889 | 0.1546 | 0.0607 |
| `FIFO_svm_under` | 0.5734 | 0.0239 | 0.0185 | 0.5216 | 0.5017 | 0.0094 |
| `FIFO_xgb_hybrid` | 0.8118 | 0.0508 | 0.1044 | 0.3983 | 0.1615 | 0.0773 |
| `FIFO_xgb_none` | 0.8576 | 0.1176 | 0.1921 | 0.4386 | 0.1938 | 0.1905 |
| `FIFO_xgb_over` | 0.8156 | 0.0490 | 0.1062 | 0.3797 | 0.1465 | 0.0841 |
| `FIFO_xgb_under` | 0.8576 | 0.1215 | 0.1917 | 0.4611 | 0.2145 | 0.1732 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `FIFO_lgb_hybrid` | 0.8299 | 0.0625 | 0.0968 | 0.3801 | 0.1635 | 0.0783 |
| `FIFO_lgb_none` | 0.8806 | 0.1363 | 0.1519 | 0.4021 | 0.1889 | 0.1790 |
| `FIFO_lgb_over` | 0.8270 | 0.0628 | 0.0902 | 0.3599 | 0.1491 | 0.0793 |
| `FIFO_lgb_under` | 0.8827 | 0.1408 | 0.1571 | 0.3819 | 0.1810 | 0.1862 |
| `FIFO_lr_hybrid` | 0.8149 | 0.0817 | 0.1080 | 0.3731 | 0.1727 | 0.0883 |
| `FIFO_lr_none` | 0.7619 | 0.0755 | 0.1050 | 0.3123 | 0.1318 | 0.1015 |
| `FIFO_lr_over` | 0.8151 | 0.0817 | 0.1072 | 0.3830 | 0.1831 | 0.0853 |
| `FIFO_lr_under` | 0.7620 | 0.0758 | 0.1097 | 0.3206 | 0.1374 | 0.1089 |
| `FIFO_mlp_hybrid` | 0.6685 | 0.0488 | 0.0513 | 0.2434 | 0.0862 | 0.0948 |
| `FIFO_mlp_none` | 0.7177 | 0.0917 | 0.1308 | 0.3431 | 0.1269 | 0.1696 |
| `FIFO_mlp_over` | 0.6614 | 0.0445 | 0.0556 | 0.2644 | 0.0991 | 0.0642 |
| `FIFO_mlp_under` | 0.7235 | 0.0945 | 0.1224 | 0.3333 | 0.1207 | 0.1708 |
| `FIFO_rf_hybrid` | 0.8521 | 0.0524 | 0.0725 | 0.3511 | 0.1426 | 0.0577 |
| `FIFO_rf_none` | 0.8667 | 0.1784 | 0.1837 | 0.4095 | 0.2154 | 0.2441 |
| `FIFO_rf_over` | 0.8509 | 0.0516 | 0.0733 | 0.3624 | 0.1518 | 0.0665 |
| `FIFO_rf_under` | 0.8723 | 0.1807 | 0.1983 | 0.4268 | 0.2257 | 0.2676 |
| `FIFO_svm_hybrid` | 0.7508 | 0.0724 | 0.0893 | 0.3541 | 0.1485 | 0.0776 |
| `FIFO_svm_none` | 0.6153 | 0.0414 | 0.0597 | 0.3306 | 0.5274 | 0.1001 |
| `FIFO_svm_over` | 0.7483 | 0.0736 | 0.0890 | 0.3744 | 0.1607 | 0.0805 |
| `FIFO_svm_under` | 0.6216 | 0.0401 | 0.0572 | 0.3296 | 0.5236 | 0.0727 |
| `FIFO_xgb_hybrid` | 0.8189 | 0.0704 | 0.1010 | 0.3698 | 0.1613 | 0.0927 |
| `FIFO_xgb_none` | 0.8656 | 0.1452 | 0.1721 | 0.3862 | 0.1857 | 0.1927 |
| `FIFO_xgb_over` | 0.8237 | 0.0674 | 0.1007 | 0.3493 | 0.1427 | 0.0885 |
| `FIFO_xgb_under` | 0.8678 | 0.1418 | 0.1922 | 0.4498 | 0.2150 | 0.2183 |

[Pooled CSV](temporal/pooled_mean.csv) · [年度平均 CSV](temporal/annual_mean.csv) · [逐年數據](temporal/by_year.csv)

## 跨採樣（最新窗口）

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `SamplingNewest_lgb` | 0.8193 | 0.0702 | 0.1040 | 0.3768 | 0.1442 | 0.0816 |
| `SamplingNewest_lr` | 0.8000 | 0.0674 | 0.1024 | 0.4170 | 0.1776 | 0.0722 |
| `SamplingNewest_mlp` | 0.6562 | 0.0468 | 0.0657 | 0.3038 | 0.0946 | 0.0549 |
| `SamplingNewest_rf` | 0.8579 | 0.0542 | 0.1150 | 0.4744 | 0.2307 | 0.0767 |
| `SamplingNewest_svm` | 0.7357 | 0.0536 | 0.0946 | 0.3629 | 0.1338 | 0.0739 |
| `SamplingNewest_xgb` | 0.8349 | 0.0809 | 0.1193 | 0.3668 | 0.1361 | 0.1066 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `SamplingNewest_lgb` | 0.8333 | 0.0963 | 0.1012 | 0.3447 | 0.1421 | 0.2025 |
| `SamplingNewest_lr` | 0.8150 | 0.0795 | 0.0936 | 0.3776 | 0.1854 | 0.0797 |
| `SamplingNewest_mlp` | 0.6642 | 0.0587 | 0.0714 | 0.2737 | 0.0989 | 0.1058 |
| `SamplingNewest_rf` | 0.8621 | 0.0752 | 0.1189 | 0.4424 | 0.2326 | 0.0968 |
| `SamplingNewest_svm` | 0.7524 | 0.0747 | 0.0896 | 0.3296 | 0.1351 | 0.1026 |
| `SamplingNewest_xgb` | 0.8396 | 0.1025 | 0.1101 | 0.3262 | 0.1333 | 0.1894 |

[Pooled CSV](sampling/pooled_mean.csv) · [年度平均 CSV](sampling/annual_mean.csv) · [逐年數據](sampling/by_year.csv)

## 跨採樣＋FIFO

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `SamplingFIFO_lgb` | 0.8211 | 0.0644 | 0.1088 | 0.4058 | 0.1684 | 0.0808 |
| `SamplingFIFO_lr` | 0.8025 | 0.0747 | 0.1176 | 0.3877 | 0.1522 | 0.0959 |
| `SamplingFIFO_mlp` | 0.6811 | 0.0463 | 0.0714 | 0.2797 | 0.0807 | 0.0732 |
| `SamplingFIFO_rf` | 0.8642 | 0.0507 | 0.1024 | 0.4117 | 0.1753 | 0.0732 |
| `SamplingFIFO_svm` | 0.7401 | 0.0578 | 0.0852 | 0.3564 | 0.1292 | 0.0636 |
| `SamplingFIFO_xgb` | 0.8370 | 0.0715 | 0.1169 | 0.3889 | 0.1534 | 0.0945 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `SamplingFIFO_lgb` | 0.8376 | 0.0838 | 0.1018 | 0.3793 | 0.1723 | 0.0924 |
| `SamplingFIFO_lr` | 0.8149 | 0.0847 | 0.1023 | 0.3382 | 0.1604 | 0.0970 |
| `SamplingFIFO_mlp` | 0.6902 | 0.0589 | 0.0743 | 0.2575 | 0.0846 | 0.1960 |
| `SamplingFIFO_rf` | 0.8688 | 0.0672 | 0.0975 | 0.3861 | 0.1785 | 0.0780 |
| `SamplingFIFO_svm` | 0.7532 | 0.0764 | 0.0806 | 0.3239 | 0.1344 | 0.1229 |
| `SamplingFIFO_xgb` | 0.8446 | 0.0885 | 0.1042 | 0.3444 | 0.1495 | 0.1008 |

[Pooled CSV](sampling_temporal/pooled_mean.csv) · [年度平均 CSV](sampling_temporal/annual_mean.csv) · [逐年數據](sampling_temporal/by_year.csv)

## 跨分類器（最新窗口）

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `ClassifierNewest_hybrid` | 0.8161 | 0.0619 | 0.1270 | 0.4109 | 0.1719 | 0.1015 |
| `ClassifierNewest_none` | 0.8532 | 0.1318 | 0.1835 | 0.4737 | 0.2272 | 0.1539 |
| `ClassifierNewest_over` | 0.8128 | 0.0614 | 0.1078 | 0.3979 | 0.1615 | 0.0814 |
| `ClassifierNewest_under` | 0.8534 | 0.1242 | 0.1744 | 0.4702 | 0.2238 | 0.1430 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `ClassifierNewest_hybrid` | 0.8240 | 0.0839 | 0.1222 | 0.3951 | 0.1734 | 0.1186 |
| `ClassifierNewest_none` | 0.8700 | 0.1550 | 0.1753 | 0.4280 | 0.2295 | 0.2293 |
| `ClassifierNewest_over` | 0.8193 | 0.0840 | 0.1080 | 0.3806 | 0.1661 | 0.1254 |
| `ClassifierNewest_under` | 0.8701 | 0.1430 | 0.1569 | 0.4123 | 0.2232 | 0.2236 |

[Pooled CSV](classifier/pooled_mean.csv) · [年度平均 CSV](classifier/annual_mean.csv) · [逐年數據](classifier/by_year.csv)

## 跨分類器＋FIFO

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `ClassifierFIFO_hybrid` | 0.8215 | 0.0555 | 0.1089 | 0.3769 | 0.1442 | 0.0876 |
| `ClassifierFIFO_none` | 0.8607 | 0.1544 | 0.1949 | 0.4708 | 0.2238 | 0.1747 |
| `ClassifierFIFO_over` | 0.8183 | 0.0548 | 0.1146 | 0.3888 | 0.1534 | 0.0915 |
| `ClassifierFIFO_under` | 0.8625 | 0.1511 | 0.1943 | 0.4656 | 0.2191 | 0.1759 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `ClassifierFIFO_hybrid` | 0.8282 | 0.0748 | 0.1047 | 0.3610 | 0.1506 | 0.1039 |
| `ClassifierFIFO_none` | 0.8757 | 0.1710 | 0.1884 | 0.4428 | 0.2234 | 0.2394 |
| `ClassifierFIFO_over` | 0.8249 | 0.0749 | 0.1040 | 0.3645 | 0.1610 | 0.0930 |
| `ClassifierFIFO_under` | 0.8780 | 0.1672 | 0.1808 | 0.4367 | 0.2175 | 0.2093 |

[Pooled CSV](classifier_temporal/pooled_mean.csv) · [年度平均 CSV](classifier_temporal/annual_mean.csv) · [逐年數據](classifier_temporal/by_year.csv)

## 跨分類器＋跨採樣（最新窗口）

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `CombinedNewest` | 0.8262 | 0.0878 | 0.1311 | 0.4355 | 0.1938 | 0.1000 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `CombinedNewest` | 0.8320 | 0.1021 | 0.1269 | 0.4041 | 0.1927 | 0.1580 |

[Pooled CSV](combined/pooled_mean.csv) · [年度平均 CSV](combined/annual_mean.csv) · [逐年數據](combined/by_year.csv)

## 跨分類器＋跨採樣＋FIFO

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `CombinedFIFO` | 0.8318 | 0.0803 | 0.1230 | 0.3746 | 0.1419 | 0.1087 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `CombinedFIFO` | 0.8374 | 0.0997 | 0.1116 | 0.3396 | 0.1461 | 0.1568 |

[Pooled CSV](combined_temporal/pooled_mean.csv) · [年度平均 CSV](combined_temporal/annual_mean.csv) · [逐年數據](combined_temporal/by_year.csv)

## Validation 選擇集成策略

### Pooled

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `ValidationSelectedEnsemble` | 0.8249 | 0.0982 | 0.1937 | 0.4294 | 0.1857 | 0.2029 |

### 年度平均

| 配置 | AUC | AP | F1 | G-Mean | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| `ValidationSelectedEnsemble` | 0.8668 | 0.1728 | 0.1761 | 0.3779 | 0.1841 | 0.2777 |

[Pooled CSV](validation_selected/pooled_mean.csv) · [年度平均 CSV](validation_selected/annual_mean.csv) · [逐年數據](validation_selected/by_year.csv)

