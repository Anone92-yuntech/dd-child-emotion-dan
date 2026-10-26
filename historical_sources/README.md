# 歷史程式來源

本資料夾保存研究程式的未改寫副本，用來區分確切實跑版本、共享資料路徑搬遷，以及本 repo 的可攜 CLI 封裝。它不是建議直接執行的入口；請使用 repo 根目錄的 `run_experiment.py`。

| 檔案 | 用途／來源 |
|---|---|
| `original_run_experiment.py` | 從搬遷備份取得的確切實跑原版；原始訓練結果記錄的 script SHA256為 `f3a9227bdc3f9064356b5f09c12178a67cf3e6930a742943e1f16e3861623b26`。 |
| `run_experiment.py` | 已遷移為讀取共享 `emotion_dataset` 的研究版本；SHA256為 `21e9d07de5aaedb16b8dc7e7ed5039887a26aedfe5aa133b911dd949ed48bb2d`，路徑搬遷不代表重訓模型。 |
| `emotion.py` | 歷史遷移前評估入口。 |
| `train_transfer.py` | 歷史5＋15 epochs遷移學習入口。 |
| `test_transfer.py` | 歷史二分類checkpoint重載評估入口。 |
| `verify_results.py` | 歷史保存預測、指標及來源雜湊的核對程式。 |

根目錄 `run_experiment.py` 以共享資料版本為封裝起點，加入 CLI 可攜路徑、CPU評估及輸出目錄隔離；不改動所發布的歷史 checkpoint 或其選模紀錄。完整來源與各檔SHA鏈見 `../provenance.json`。

歷史原檔可能保留研究電腦的路徑及本機環境假設，不能把搬遷前後的程式SHA不同解讀成模型權重改變。私有影像、CSV及逐張預測不隨這些程式發布。DAN網路來源與MIT notice另存於 `../networks/`。
