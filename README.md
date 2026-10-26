# 發展遲緩兒童表情二分類：DAN 遷移學習

本專案封裝已完成的 **AffectNet-7 DAN → Neutral／Positive 二分類**模型、訓練／測試程式與聚合結果。標籤為 `0=neutral`、`1=positive`。

這份 DAN 是目前已完成的單一 CNN 實驗中，綜合 Positive F1 與 Recall 較佳的候選；不是融合模型、雙教師集成或蒸餾學生。「最佳」指既有實驗比較，**模型的 epoch 仍只根據 validation 選定，沒有用 test 選 epoch 或調門檻**。不同模型的單次訓練結果不構成跨 seed 的顯著性證明。

## 模型與結果

DAN 採 ResNet-18 骨幹及4個 attention heads，分類端為 `Linear(512,2) → BatchNorm1d(2)`。發布權重是純 tensor `state_dict`，可使用 `torch.load(..., weights_only=True)` 與 strict loading。

最佳 checkpoint 為 Stage 2 第15輪，即 global epoch20；依 Stage 2 validation Positive F1 最大值選擇，同分保留較早 epoch。固定決策為 `positive_probability > 0.5`，恰好0.5判為 neutral。

| 已選定模型 | Accuracy | Positive Precision | Positive Recall | Positive F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.9515850 | 0.7420147 | 0.8273973 | 0.7823834 | 3000 | 105 | 63 | 302 |
| Test | **0.9345446** | **0.7331081** | **0.7791741** | **0.7554395** | 3578 | 158 | 123 | 434 |

這些是歷史實跑保存的結果。CPU／CUDA、套件版本及運算設定可能造成數值差異；封裝驗證不宣稱跨環境逐 bit 相同。未測量 FPS 或延遲，不宣稱即時辨識能力。

## 取得私人 Release 權重

Repository：<https://github.com/Anone92-yuntech/dd-child-emotion-dan>。權重約79 MB，**放在 Release `v1.0.0` 的 asset，沒有放進 Git blob**。

1. 登入有權限的 GitHub 帳號後 clone 此私人 repository。
2. 至 [v1.0.0 Release](https://github.com/Anone92-yuntech/dd-child-emotion-dan/releases/tag/v1.0.0)，下載 `best_transfer_model.pth`，放到 repo 根目錄。
3. 安裝相依套件並執行封裝驗證。

若已安裝 GitHub CLI，也可在 clone 後執行：

```powershell
gh auth login
gh release download v1.0.0 --repo Anone92-yuntech/dd-child-emotion-dan --pattern best_transfer_model.pth --dir .
python -m pip install -r requirements.txt
python -B verify_package.py
```

成品權重 SHA256：

```text
fccef63f6464c678852224f7cbf24c9c6fb9b9e682f0fd15f774ec7df4b245b3
```

`requirements.txt` 是基本相依需求，不是精確鎖版。原實跑使用 Python3.12.7、PyTorch2.12.0.dev20260408+cu128、torchvision0.27.0.dev20260407+cu128、CUDA12.8、RTX5070Ti；完整版本與來源雜湊另由隨附的環境／來源 JSON 保存。

## 單圖與資料夾推論

**輸入必須是已裁切的面部影像。**本模型不含人臉偵測、目標人物選擇、臉部裁切或影片追蹤。影片須先取得目標臉部裁切，再送入模型；不需要 Blendshape，也不會執行 CLAHE、銳化或其他影像增強。

推論流程固定為 RGB → Pillow bilinear Resize224×224 → ToTensor → 一次 ImageNet Normalize，mean=`[0.485,0.456,0.406]`、std=`[0.229,0.224,0.225]`。分類順序為 `[neutral, positive]`。

```powershell
# 單張已裁切面部；CPU推論，不需要資料集或原始七類預訓練檔
python -B predict.py --input path/to/face_crop.jpg --checkpoint best_transfer_model.pth --metadata model_metadata.json --device cpu --output-json runs/inference/prediction.json

# 資料夾，可包含子目錄；auto有CUDA時用GPU，否則用CPU
python -B predict.py --input path/to/face_crops --recursive --device auto --output-csv runs/inference/predictions.csv
```

`--output-json` 與 `--output-csv` 可同時指定；未指定輸出檔時，結果列印至終端。輸出含兩類機率、數值標籤與表情名稱。`--metadata` 是選用的架構、標籤、前處理及 checkpoint SHA 檢查。詳細介面見 `python predict.py --help`。

## 私有資料與切分

影像、私有 CSV、逐張預測及樣本名稱未發布；repo 只保存聚合指標、計數與曲線。所有本機遷移學習實驗直接使用同一份 `../emotion_dataset`，不將資料集複製進模型專案。

| Split | 總數 | Neutral | Positive |
|---|---:|---:|---:|
| balanced_train | 4598 | 2299 | 2299 |
| validation | 3470 | 3105 | 365 |
| test | 4293 | 3736 | 557 |

使用者確認三個集合來自不同兒童；subject-disjoint 是資料提供者確認的前提，不能只由檔名推得。沿用既有 split 與標籤，不重新切分。程式只讀影像與 `image_name`、`label`，不使用52維 Blendshape；`blendshape_valid=0` 的影像也保留。

若具備原資料，目錄應為：

```text
../emotion_dataset/
  balanced_train/  # 圖片及 train_dataset_blendshapes_masked.csv
  validation/      # 圖片及 validation_dataset_blendshapes_masked.csv
  test/            # 圖片及 test_dataset_blendshapes_masked.csv
```

## 訓練設定與重現

| 項目 | 設定 |
|---|---|
| 起點 | 官方 AffectNet-7 DAN 權重，載入後新建二類 fc 與 bn |
| Seed／batch／workers | 42／32／4 |
| Loss／optimizer | 無類別加權 CrossEntropyLoss；Adam；weight decay=0 |
| Stage 1 | 5 epochs，lr=`1e-3`，只更新分類端 fc／bn 參數 |
| Stage 2 | 15 epochs，lr=`1e-5`，全部解凍、建立新的 Adam |
| 階段銜接 | 從 Stage 1 **最後一輪**接 Stage 2，不回復 Stage 1 最佳輪 |
| 選模 | 僅 Stage 2 validation Positive F1 最大；同分取較早輪 |
| 數值設定 | FP32、無AMP、TF32關閉、cuDNN benchmark=False／deterministic=True |
| 額外策略 | 無 scheduler、early stopping、AffinityLoss 或 PartitionLoss |

訓練增強為 Resize224、水平翻轉 p=0.5、旋轉±10°（nearest）、亮度／對比 ColorJitter 各0.2，再 ToTensor／Normalize。Validation/test 僅 Resize／ToTensor／Normalize。

Stage 1 凍結的是骨幹**參數**，仍使用 `model.train()`，因此骨幹 BatchNorm running statistics 會更新；這沿用原研究的 warm-up 政策，不是連 buffers 都完全凍結。

重新訓練另需原始 `affecnet7_epoch6_acc0.6569.pth`（約237 MB），本 repo／Release **未提供此檔**。請從 [官方 DAN repository](https://github.com/yaoing/DAN) 說明取得相應來源，遵循其發佈條件；不以本次二分類成品替代訓練起點。程式核對其 SHA256：

```text
3189a13f28592c20ce9b9dd57e457caabf6a7bb80c36df94d3d0348829cec752
```

訓練需要 CUDA；將 `PATH_TO_AFFECTNET7_CHECKPOINT` 替換為該檔實際路徑：

```powershell
python -B run_experiment.py train --data-root ../emotion_dataset --pretrained PATH_TO_AFFECTNET7_CHECKPOINT --checkpoint runs/reproduction/best.pth --results-dir runs/reproduction
```

重新評估已發布成品可使用 CPU；指定新輸出目錄，以保留 `results/` 的歷史結果：

```powershell
python -B run_experiment.py test --data-root ../emotion_dataset --checkpoint best_transfer_model.pth --selection-metadata results/best_checkpoint.json --results-dir runs/evaluation --device cpu
```

`--device auto` 可自動使用 GPU；runner 也接受 `cuda:0`。已有 checkpoint 時，重新訓練必須明確加 `--overwrite`；一般重現應使用新的 checkpoint／results 路徑。`test` 會評估 validation 及 test，不會重新選 epoch 或調閾值。

## 驗證與來源保存

`verify_package.py` 預設在 CPU 核對 package manifest、SHA256、模型 metadata、聚合指標算術與275個 tensor 的安全 strict 載入，**不需要私有資料、不訓練，也不重跑完整 test**。

若本機具備原資料，可額外檢查一批32張 validation 面部影像；報告應另存 repo 外：

```powershell
python -B verify_package.py --data-root ../emotion_dataset --device cpu --output ../dan_private_verification.json
```

可另加 `--expected-results PATH_TO_PRIVATE_VALIDATION_PREDICTIONS.csv`，對照原 validation 預測；未提供時只做 live batch 的有效性檢查，不宣稱已對照歷史機率。此選項不讀 test 圖片。數值比較使用明確 tolerance，不宣稱 CPU／GPU 逐 bit 一致。

目前 `run_experiment.py` 由原 DAN 實驗程式封裝而來，保留訓練數值協定，加入可攜 CLI 路徑、CPU評估與新輸出隔離。[historical_sources/](historical_sources/README.md) 保留未改寫的確切實跑原版及資料路徑搬遷版；完整雜湊鏈見 `provenance.json`。[results/](results/README.md) 保留聚合研究紀錄。

## 程式來源與限制

網路程式來自 [yaoing/DAN](https://github.com/yaoing/DAN)，來源 commit `fea9a093bc7c44ee3f08dd00310516b6e5a92c09`，保留 [MIT notice](networks/LICENSE.DAN)；本地相容性調整是 torchvision ResNet18 的 `weights=` API。詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。程式的 MIT notice 不會自動替研究資料、AffectNet資料或模型權重授權。

本次為單一 seed、固定協定的研究結果；原始 test 已有歷史評估，不是新外部盲測，也未做跨 seed 顯著性分析。原始 HSEmotion 未固定相同 seed，因此跨模型比較是已完成實驗的參考，不能把差異全部歸因於架構。
