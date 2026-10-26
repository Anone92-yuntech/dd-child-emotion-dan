# 第三方來源與授權範圍

## DAN 網路程式

- 官方專案：[yaoing/DAN](https://github.com/yaoing/DAN)。
- 網路來源 commit：`fea9a093bc7c44ee3f08dd00310516b6e5a92c09`。
- 本 repo 的 `networks/dan.py` 沿用該網路實作；相容性調整為以現行 torchvision `weights=` API 取代舊 ResNet18 positional `pretrained` API。
- 原著作權：**Copyright (c) 2021 Zhengyao Wen**。
- 已保留完整 MIT 文本於 [networks/LICENSE.DAN](networks/LICENSE.DAN)，分發此網路程式時應一併保留 notice。

本研究的訓練／評估 runner 以相同資料切分與兩階段遷移學習協定執行，不採用 DAN 原訓練的 AffinityLoss／PartitionLoss。研究程式的歷史原檔與可攜封裝版之間的來源鏈，記於 `historical_sources/` 及 `provenance.json`。

## 權重與資料來源

發布的 `best_transfer_model.pth` 是從官方 AffectNet-7 DAN 起點進行研究二分類微調後的權重，並非原始七類權重。成品 SHA256：

```text
fccef63f6464c678852224f7cbf24c9c6fb9b9e682f0fd15f774ec7df4b245b3
```

原始預訓練檔名為 `affecnet7_epoch6_acc0.6569.pth`，SHA256：

```text
3189a13f28592c20ce9b9dd57e457caabf6a7bb80c36df94d3d0348829cec752
```

原始約237 MB預訓練檔不隨本 repo 或 Release 發布；取得資訊請依 [官方 DAN 說明](https://github.com/yaoing/DAN)。AffectNet及本研究的兒童資料、私有CSV、逐張預測均未包含於此封裝。

**DAN程式的MIT notice不代表資料集、預訓練權重或本次微調權重自動以MIT授權。**本文件保留已知來源與程式授權文本，不替上游資料或權重新增授權條件。

## 相依套件

執行時使用 PyTorch、torchvision、Pillow、NumPy、pandas、matplotlib 與 scikit-learn。套件由各自發佈來源安裝，其程式及依賴沿用各自授權；本 repo 沒有將其原始碼或 binary vendor進專案。基本需求見 `requirements.txt`，精確實跑版本另由環境JSON保存。
