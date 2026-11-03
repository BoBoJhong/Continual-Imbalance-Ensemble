# 持續學習定位釐清與引文核對

作者補充：「有，只是受限在資料集」。本輪修正先前僅以 Test 期間未重訓就排除持續學習的過窄判準；將本研究描述為保留 Old 候選並加入 New 模型的兩階段集成更新。程式會在同次離線實驗重建兩期模型，沒有證據表示已做多次真實部署、跨執行載入舊模型檔或多輪遺忘評估，故正文以固定分區離線模擬的一次池擴充界定。

## 檢索與核對

2026-11-03 查核單篇定位文獻，非全面文獻回顧。檢索 Learn++ incremental learning retained classifiers、Learn++ nonstationary ensembles Elwell Polikar，開啟原作者全文及 PubMed 書目核對。另讀 Parisi 等持續學習回顧摘要作概念交叉檢查，沒有將該來源加入本輪論文。

採用：Ryan Elwell、Robi Polikar，2011，Incremental Learning of Concept Drift in Nonstationary Environments，IEEE Transactions on Neural Networks，22(10)，1517–1531，DOI **10.1109/TNN.2011.2160459**。

- 原作者全文：https://users.rowan.edu/~polikar/RESEARCH/PUBLICATIONS/tnn11.pdf
- 書目：https://pubmed.ncbi.nlm.nih.gov/21824845/
- 定位：PDF 第 1 頁／印刷 p.1517 摘要，描述新批次建立新分類器並與其他分類器加權整合。
- 支持：集成式批次增量更新可在模型池與組合層級發生，不能只以基礎模型是否微調或 Test 期是否重訓判定。
- 不支持：本研究已採該文特定權重算法、具相同效能或完成同等長序列適應；也不能由此證明所有 Old／New 並列比較自動構成完整持續學習。

主張 CL-01–03 已同步於 docs/研究方向.md。中英文摘要加入 continual learning 關鍵字，正文交代模型層級兩階段更新及資料／配置限制；研究題目、實驗、數值與統計家族不變。

before-continual-position/ 保存加入此定位前的工作稿與 PDF；既有深度模型暫列修訂仍保留。後續最值得驗證的是多輪更新下的新資訊吸收、舊知識保留及事件標籤可用時間。
