"""DAN image-only baseline and HSEmotion-matched binary transfer experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from networks.dan import DAN

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT.parent / "emotion_dataset"
PRETRAINED = ROOT / "affecnet7_epoch6_acc0.6569.pth"
BEST = ROOT / "best_transfer_model.pth"
RESULTS = ROOT / "results"
LABELS = ["neutral", "positive"]
AFFECTNET = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]
CSV_NAMES = {"balanced_train": "train_dataset_blendshapes_masked.csv",
             "validation": "validation_dataset_blendshapes_masked.csv",
             "test": "test_dataset_blendshapes_masked.csv"}
CONFIG = {
    "seed": 42, "batch_size": 32, "stage1_epochs": 5, "stage2_epochs": 15,
    "stage1_lr": 1e-3, "stage2_lr": 1e-5, "num_workers": 4,
    "num_heads": 4, "image_size": 224, "loss": "CrossEntropyLoss",
    "optimizer": "Adam", "weight_decay": 0, "amp": False,
    "train_augmentation": "Resize224, HFlip0.5, Rotation10, ColorJitter(brightness=.2,contrast=.2)",
    "normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
    "selection": "maximum validation positive F1 across stage2 only; earliest epoch wins ties",
    "stage_transition": "last stage1 epoch to stage2, matching HSEmotion",
    "warmup_batchnorm": "all running statistics update in model.train(), matching HSEmotion; only fc/bn parameters train",
    "baseline": "happy logit(index1) > neutral logit(index0); other five classes ignored, matching HSEmotion pairwise rule",
    "binary_decision": "positive if p_positive > 0.5; ties are neutral",
    "test_policy": "baseline before training; final model selected on validation only, one final test; no threshold tuning",
    "sample_policy": "all CSV rows retained including blendshape_valid=0; blendshapes not used",
    "precision": "float32; TF32 disabled", "cudnn_benchmark": False,
    "cudnn_deterministic": True,
}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup():
    torch.set_num_threads(8)
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment; refusing silent CPU training")
    RESULTS.mkdir(exist_ok=True)
    return torch.device("cuda:0")


def image_path(split, name):
    name = str(name)
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        name += ".jpg"
    path = (DATASET_ROOT / split / name).resolve()
    if not path.is_relative_to((DATASET_ROOT / split).resolve()):
        raise ValueError(f"Image outside split: {name}")
    return path


def frame(split):
    data = pd.read_csv(DATASET_ROOT / split / CSV_NAMES[split])
    if not {"image_name", "label"}.issubset(data.columns):
        raise ValueError(f"Missing columns in {split}")
    if not data.label.isin([0, 1]).all() or data.image_name.duplicated().any():
        raise ValueError(f"Invalid labels or duplicate names in {split}")
    return data


def video_group(name):
    stem = Path(str(name)).stem.lower()
    match = re.match(r"(dd-?\d+)", stem)
    return match.group(1) if match else re.sub(r"(?:[_-]?(?:frame)?\d+)$", "", stem)


def audit_image(item):
    split, name, label = item
    path = image_path(split, name)
    content = path.read_bytes()
    with Image.open(io.BytesIO(content)) as im:
        im.convert("RGB").load()
    return {"split": split, "image_name": str(name), "label": int(label),
            "sha256": hashlib.sha256(content).hexdigest(), "video_group": video_group(name)}


def audit():
    print("Auditing all images, CSVs and cross-split duplicate bytes...", flush=True)
    splits = {}
    tasks = []
    for split in CSV_NAMES:
        df = frame(split)
        splits[split] = {"rows": len(df), "class_counts": {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()},
                         "csv_sha256": sha256(DATASET_ROOT / split / CSV_NAMES[split]),
                         "video_groups": sorted({video_group(n) for n in df.image_name})}
        tasks.extend((split, n, int(y)) for n, y in zip(df.image_name, df.label))
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(audit_image, tasks))
    manifest = pd.DataFrame(records)
    manifest.to_csv(RESULTS / "dataset_manifest.csv", index=False)
    cross = []
    repeated = manifest[manifest.sha256.duplicated(keep=False)]
    for digest, group in repeated.groupby("sha256"):
        if group.split.nunique() > 1:
            cross.append({"sha256": digest, "samples": group.drop(columns="sha256").to_dict("records")})
    groups = {split: set(info["video_groups"]) for split, info in splits.items()}
    intersections = {f"{a}__{b}": sorted(groups[a] & groups[b]) for i, a in enumerate(groups) for b in list(groups)[i+1:]}
    result = {"dataset_root": str(DATASET_ROOT), "splits": splits, "all_images_decoded": True, "cross_split_byte_duplicates": cross,
              "video_group_intersections": intersections,
              "identity_independence": "not established by filenames; no person identity annotations inspected",
              "pretrained_sha256": sha256(PRETRAINED)}
    save_json(RESULTS / "data_audit.json", result)
    print(json.dumps({"splits": splits, "cross_split_duplicate_groups": len(cross)}, ensure_ascii=False), flush=True)
    return result


def transform(train=False):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=.2, contrast=.2)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(**CONFIG["normalization"])])


class EmotionDataset(Dataset):
    def __init__(self, split, train=False):
        self.split = split
        self.data = frame(split)
        self.paths = [image_path(split, n) for n in self.data.image_name]
        self.labels = self.data.label.to_numpy(dtype=np.int64)
        self.transform = transform(train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            image = self.transform(im.convert("RGB"))
        return image, int(self.labels[index])


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(split, train=False):
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    return DataLoader(EmotionDataset(split, train), batch_size=CONFIG["batch_size"], shuffle=train,
                      num_workers=CONFIG["num_workers"], pin_memory=True,
                      persistent_workers=CONFIG["num_workers"] > 0,
                      worker_init_fn=seed_worker, generator=generator)


def pretrained_model(device, binary=False):
    model = DAN(num_class=7, num_head=4, pretrained=False)
    checkpoint = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if binary:
        model.fc = nn.Linear(512, 2)
        model.bn = nn.BatchNorm1d(2)
    return model.to(device)


def metrics(labels, probabilities):
    predicted = (np.asarray(probabilities) > .5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, predicted)),
            "positive_precision": float(precision), "positive_recall": float(recall), "positive_f1": float(f1),
            "macro_f1": float(precision_recall_fscore_support(labels, predicted, average="macro", zero_division=0)[2]),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            "confusion_matrix": cm.tolist(), "threshold": .5, "decision_operator": ">",
            "classification_report": classification_report(labels, predicted, labels=[0, 1], target_names=LABELS, output_dict=True, zero_division=0)}


def predict(model, data_loader, device, baseline=False, progress=False):
    model.eval()
    all_logits, losses = [], 0.
    with torch.inference_mode():
        for batch, (images, labels) in enumerate(data_loader, 1):
            out, _, _ = model(images.to(device, non_blocking=True))
            if not torch.isfinite(out).all():
                raise FloatingPointError("Nonfinite inference logits")
            all_logits.append(out.cpu().numpy())
            if not baseline:
                losses += nn.functional.cross_entropy(out, labels.to(device), reduction="sum").item()
            if progress and (batch % 40 == 0 or batch == len(data_loader)):
                print(f"  inference {batch}/{len(data_loader)} batches", flush=True)
    logits = np.concatenate(all_logits)
    pair_logits = logits[:, [0, 1]]
    probabilities = torch.softmax(torch.from_numpy(pair_logits), dim=1)[:, 1].numpy()
    result = metrics(data_loader.dataset.labels, probabilities)
    if not baseline:
        result["loss"] = losses / len(data_loader.dataset)
    return result, probabilities, logits


def export_evaluation(name, data_loader, result, probabilities, logits, baseline=False):
    table = data_loader.dataset.data[["image_name", "label"]].copy()
    table["positive_probability"] = probabilities
    table["prediction"] = (probabilities > .5).astype(int)
    table["correct"] = table.label == table.prediction
    for i, label in enumerate(AFFECTNET if baseline else LABELS):
        table[f"logit_{label}"] = logits[:, i]
    table.to_csv(RESULTS / f"{name}_predictions.csv", index=False)
    save_json(RESULTS / f"{name}_metrics.json", result)
    report = classification_report(table.label, table.prediction, labels=[0, 1], target_names=LABELS, digits=4, zero_division=0)
    (RESULTS / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.asarray(result["confusion_matrix"])
    ax.imshow(cm, cmap="Blues" if baseline else "Oranges")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="white" if cm[row, col] > cm.max()/2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted label", ylabel="True label", title=f"DAN {'Before' if baseline else 'After'} Transfer Learning\n{name}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_confusion_matrix.png", dpi=180)
    fig.savefig(ROOT / ("baseline_confusion_matrix.png" if baseline else "transfer_confusion_matrix.png"), dpi=180)
    plt.close(fig)
    print(f"{name}: accuracy={result['accuracy']:.4f}, positive precision={result['positive_precision']:.4f}, recall={result['positive_recall']:.4f}, F1={result['positive_f1']:.4f}", flush=True)
    print(report, flush=True)


def baseline(device):
    print("Evaluating untouched AffectNet-7 DAN: happy[1] vs neutral[0]", flush=True)
    model = pretrained_model(device)
    test_loader = loader("test")
    values = predict(model, test_loader, device, baseline=True, progress=True)
    export_evaluation("baseline_test", test_loader, *values, baseline=True)
    del model, test_loader
    torch.cuda.empty_cache()


def train(device):
    # Reset seeds so baseline inference/loader does not affect training initialization.
    setup()
    model = pretrained_model(device, binary=True)
    train_loader = loader("balanced_train", train=True)
    val_loader = loader("validation")
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1, best_info = -1., None
    started = time.time()
    for stage, epochs, lr in [(1, CONFIG["stage1_epochs"], CONFIG["stage1_lr"]), (2, CONFIG["stage2_epochs"], CONFIG["stage2_lr"])]:
        for name, param in model.named_parameters():
            param.requires_grad = stage == 2 or name.startswith(("fc.", "bn."))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        print(f"Stage {stage}: epochs={epochs}, lr={lr}, trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            model.train()  # Intentional HSEmotion parity: frozen parameters, updating BN statistics.
            total_loss, correct = 0., 0
            for batch, (images, labels) in enumerate(train_loader, 1):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out, _, _ = model(images)
                loss = criterion(out, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in stage {stage} epoch {epoch}")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                correct += (out.argmax(1) == labels).sum().item()
                if batch % 40 == 0 or batch == len(train_loader):
                    print(f"  stage{stage} epoch{epoch} train {batch}/{len(train_loader)} loss={total_loss/min(batch*CONFIG['batch_size'],len(train_loader.dataset)):.4f}", flush=True)
            val_result, _, _ = predict(model, val_loader, device)
            row = {"stage": stage, "epoch": epoch, "global_epoch": len(history)+1, "lr": lr,
                   "train_loss": total_loss / len(train_loader.dataset), "train_accuracy": correct / len(train_loader.dataset),
                   **{f"validation_{k}": val_result[k] for k in ["loss", "accuracy", "positive_precision", "positive_recall", "positive_f1", "macro_f1"]},
                   "seconds": time.time()-epoch_start}
            history.append(row)
            pd.DataFrame(history).to_csv(RESULTS / "training_history.csv", index=False)
            if stage == 2 and val_result["positive_f1"] > best_f1:
                best_f1 = val_result["positive_f1"]
                best_info = {"stage": stage, "epoch": epoch, "global_epoch": row["global_epoch"], "validation_metrics": val_result}
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                temp = BEST.with_suffix(".pth.tmp")
                torch.save(state, temp)
                temp.replace(BEST)
                save_json(RESULTS / "best_checkpoint.json", {**best_info, "sha256": sha256(BEST), "config": CONFIG})
                print(f"  Saved best stage2 checkpoint: positive F1={best_f1:.6f}", flush=True)
            save_json(RESULTS / "progress.json", {"status": "training", "last_epoch": row, "best_checkpoint": best_info, "elapsed_seconds": time.time()-started})
            print(f"Stage{stage} epoch{epoch}/{epochs}: train_loss={row['train_loss']:.4f} val_loss={val_result['loss']:.4f} val_F1={val_result['positive_f1']:.4f} val_recall={val_result['positive_recall']:.4f} seconds={row['seconds']:.1f}", flush=True)
    save_json(RESULTS / "training_summary.json", {"config": CONFIG, "best_checkpoint": best_info, "elapsed_seconds": time.time()-started, "completed_epochs": len(history), "checkpoint_sha256": sha256(BEST)})
    plot_history(pd.DataFrame(history))
    del model, train_loader, val_loader
    torch.cuda.empty_cache()


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.global_epoch, history.train_loss, label="train")
    axes[0].plot(history.global_epoch, history.validation_loss, label="validation")
    axes[0].set(ylabel="Cross entropy", xlabel="Epoch", title="Loss")
    axes[1].plot(history.global_epoch, history.validation_positive_f1, label="positive F1")
    axes[1].plot(history.global_epoch, history.validation_positive_recall, label="positive recall")
    axes[1].set(ylabel="Score", xlabel="Epoch", title="Validation")
    for ax in axes:
        ax.axvline(5.5, linestyle="--", color="gray", label="full fine-tuning starts")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=180)
    plt.close(fig)


def test(device):
    info = json.loads((RESULTS / "best_checkpoint.json").read_text(encoding="utf-8"))
    if sha256(BEST) != info["sha256"]:
        raise ValueError("Best checkpoint hash does not match selection metadata")
    model = DAN(num_class=2, num_head=4, pretrained=False).to(device)
    model.load_state_dict(torch.load(BEST, map_location="cpu", weights_only=True), strict=True)
    for split in ["validation", "test"]:
        dl = loader(split)
        values = predict(model, dl, device, progress=True)
        export_evaluation(f"transfer_{split}", dl, *values)
        if split == "validation" and not np.isclose(values[0]["positive_f1"], info["validation_metrics"]["positive_f1"], atol=1e-12):
            raise AssertionError("Reloaded checkpoint validation F1 differs from selection")
        del dl


def report():
    before = json.loads((RESULTS / "baseline_test_metrics.json").read_text(encoding="utf-8"))
    after = json.loads((RESULTS / "transfer_test_metrics.json").read_text(encoding="utf-8"))
    info = json.loads((RESULTS / "best_checkpoint.json").read_text(encoding="utf-8"))
    table = pd.DataFrame([{ "model": name, **{k: m[k] for k in ["n", "accuracy", "positive_precision", "positive_recall", "positive_f1", "macro_f1", "tn", "fp", "fn", "tp"]}} for name, m in [("DAN before transfer", before), ("DAN after transfer", after)]])
    table.to_csv(RESULTS / "comparison.csv", index=False)
    lines = ["# DAN 中性／正向二分類實驗結果", "", "使用 ../emotion_dataset 共用資料集的相同切分，只使用影像；所有 CSV 樣本皆納入。", "",
             "| 測試模型 | Accuracy | 正向 Precision | 正向 Recall | 正向 F1 |", "|---|---:|---:|---:|---:|"]
    for name, m in [("遷移前 AffectNet-7 DAN", before), ("遷移後二分類 DAN", after)]:
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['positive_precision']:.2%} | {m['positive_recall']:.2%} | {m['positive_f1']:.2%} |")
    lines += ["", f"正向 F1 改變：{(after['positive_f1']-before['positive_f1'])*100:+.2f} 個百分點。",
              f"最佳模型：第二階段第 {info['epoch']} 個 epoch（總第 {info['global_epoch']} 個），驗證集正向 F1={info['validation_metrics']['positive_f1']:.4f}。", "",
              "## 實驗方式", "",
              "- 遷移前：官方 AffectNet-7 權重，直接比較 happy(index 1) 與 neutral(index 0) logits；其餘五類不參與決策。輸出的 positive_probability 是這兩類之間重新正規化的條件機率。",
              "- 遷移後：保留 DAN 骨幹與注意力模組，將 fc 與其後 BatchNorm 改為 2 類（0=neutral，1=positive）。",
              "- 沿用 HSEmotion：224 RGB、ImageNet 正規化、水平翻轉／旋轉10度／亮度對比擴增；batch32、CE、Adam、5 epochs × 1e-3 分類端訓練 + 15 epochs × 1e-5 全面微調。",
              "- 與原 HSEmotion 程式一致，第一階段凍結骨幹參數，但 model.train() 仍更新其 BatchNorm running statistics；第二階段從第一階段最後狀態開始，只在第二階段以驗證集正向 F1 選最佳權重。",
              "- 不使用 DAN 原論文的 AffinityLoss／PartitionLoss，以保持 HSEmotion 遷移學習損失設定一致。",
              "- seed=42、float32、不使用 AMP、固定二分類閾值0.5。資料載入使用4個 workers；系統/初始化差異仍可能影響與舊 HSEmotion 單次訓練的比較。",
              "- 模型選擇只使用 validation，沒有利用 test 選 epoch、學習率或閾值；本次為單一 seed 實驗。",
              "- 按檔名檢查影片群組、並檢查跨集合相同檔案位元組；這無法證明人物身分獨立。詳見 data_audit.json。", "",
              "## 產物與重跑", "", "權重：../best_transfer_model.pth；逐張預測：baseline_test_predictions.csv、transfer_test_predictions.csv；指標：對應 *_metrics.json；曲線：training_curves.png。", "",
              "執行環境沿用 ../HSEmotion/.venv。run_experiment.py 的 all 會先稽核、執行基準、訓練與測試；若已有最佳權重，必須加 --overwrite 才能重訓。", "",
              "官方來源：https://github.com/yaoing/DAN；模型程式採用官方 dan.py，只修改 torchvision resnet18 的 weights API 相容性，保留原 MIT 授權。", ""]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(RESULTS / "progress.json", {"status": "complete", "baseline_positive_f1": before["positive_f1"], "transfer_positive_f1": after["positive_f1"], "checkpoint_sha256": sha256(BEST)})
    print(table.to_string(index=False), flush=True)


def main(command=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["all", "audit", "baseline", "train", "test", "report"], default=command or "all")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing trained model")
    args = parser.parse_args()
    if args.command in ["all", "train"] and BEST.exists() and not args.overwrite:
        raise FileExistsError(f"Existing {BEST}; use --overwrite explicitly to retrain")
    device = setup()
    if args.command in ["all", "audit"]:
        audit()
    if args.command in ["all", "baseline", "train"]:
        save_json(RESULTS / "experiment_config.json", {"config": CONFIG, "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "gpu_bytes": torch.cuda.get_device_properties(device).total_memory,
            "dataset_root": str(DATASET_ROOT), "pretrained_sha256": sha256(PRETRAINED), "script_sha256": sha256(__file__),
            "dan_source_sha256": sha256(ROOT / "networks" / "dan.py"),
            "official_source_commit": "fea9a093bc7c44ee3f08dd00310516b6e5a92c09"})
    if args.command in ["all", "baseline"]:
        baseline(device)
    if args.command in ["all", "train"]:
        train(device)
    if args.command in ["all", "test"]:
        test(device)
    if args.command in ["all", "report"]:
        report()


if __name__ == "__main__":
    main()
