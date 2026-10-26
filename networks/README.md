# DAN 網路來源與載入

`dan.py` 來自官方 [yaoing/DAN](https://github.com/yaoing/DAN)，來源 commit為 `fea9a093bc7c44ee3f08dd00310516b6e5a92c09`。網路保留 ResNet-18與4個cross-attention heads；相容性調整為 torchvision ResNet18 `weights=` API。原MIT著作權及授權文本保存在 [LICENSE.DAN](LICENSE.DAN)。

本repo發布的是二分類權重：

- `0=neutral`，`1=positive`。
- `fc=Linear(512,2)`，其後為 `BatchNorm1d(2)`。
- 純state_dict共275個tensor，包含BatchNorm running statistics。
- 建模必須傳入 `pretrained=False`，再strict載入成品；不需要下載ImageNet權重或讀取原七類預訓練檔。

```python
import torch
from networks.dan import DAN

model = DAN(num_class=2, num_head=4, pretrained=False)
state = torch.load("best_transfer_model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

`forward()` 回傳 `(out, features, heads)`；`out` 才是BatchNorm後的二類logits，應使用 `softmax(out, dim=1)` 取得情緒機率。attention heads內部的log_softmax不是二類情緒機率。

評估時使用 `eval()`，避免BatchNorm統計量更新。輸入是已裁切面部的RGB影像，bilinear224×224後ToTensor及ImageNet Normalize；不包含人臉偵測、Blendshape或CLAHE。一般使用者可直接用根目錄 `predict.py`，不必自行組裝前處理。

此處MIT notice對應DAN網路程式來源，不替資料或權重新增授權；詳見 `../THIRD_PARTY_NOTICES.md`。
