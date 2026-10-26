# 歷史聚合結果

本目錄保存已選定DAN模型的聚合指標、混淆矩陣、訓練曲線與逐epoch指標；**不包含私有CSV、逐張預測或樣本名稱**。

| 檔案 | 內容 |
|---|---|
| `best_checkpoint.json` | Stage2第15輪／global20的validation選模依據及checkpoint SHA256。 |
| `transfer_validation_metrics.json` | 已選模型在3470張validation上的聚合指標。 |
| `transfer_test_metrics.json` | 已選模型在4293張test上的聚合指標。 |
| `baseline_test_metrics.json` | 遷移前AffectNet-7 DAN的happy／neutral pairwise參考結果。 |
| `comparison.csv` | 遷移前後的聚合比較，並非逐樣本資料。 |
| `training_history.csv` | 20輪訓練與validation loss／指標。 |
| `*_confusion_matrix.png`、`training_curves.png` | 聚合視覺化。 |

選模只看Stage2 validation Positive F1，嚴格提升才更新，同分保留較早輪；test不參與epoch或門檻選擇。已選定模型結果：

| Split | Accuracy | Positive Precision | Positive Recall | Positive F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---|
| Validation | 0.9515850 | 0.7420147 | 0.8273973 | 0.7823834 | 3000 / 105 / 63 / 302 |
| Test | 0.9345446 | 0.7331081 | 0.7791741 | 0.7554395 | 3578 / 158 / 123 / 434 |

決策固定Positive機率>0.5，恰好0.5判neutral。遷移前的pairwise機率是七類模型的neutral／happy兩類重新正規化，不等同已訓練的二分類head。

重現或重新評估請將新輸出保存到 `../runs/reproduction`、`../runs/evaluation` 等目錄，保留此處的歷史紀錄。新輸出可能包含私有逐張預測，屬於本機研究產物；本封裝只發佈上述聚合資料。跨硬體／版本重算的數值不保證逐bit相同。
