"""Inference on already cropped faces using the binary DAN checkpoint.

No face detection, image enhancement, training data, or pretrained download is
needed. A probability of exactly 0.5 is classified as neutral.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "best_transfer_model.pth"
MEAN, STD = [.485, .456, .406], [.229, .224, .225]
LABELS = ["neutral", "positive"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_metadata(metadata, checkpoint_sha256=None):
    expected_model = {"model_name": "DAN", "num_class": 2, "num_head": 4}
    if metadata["model_config"] != expected_model:
        raise ValueError("Metadata must specify DAN with two classes and four heads")
    if metadata["label_mapping"] != {"0": "neutral", "1": "positive"}:
        raise ValueError("Metadata class order must be 0=neutral, 1=positive")
    if metadata["decision_threshold"] != .5:
        raise ValueError("This model uses the fixed positive probability > 0.5 rule")
    prep = metadata["image_preprocess"]
    expected = {"size": [224, 224], "resize": "PIL bilinear", "rgb": True,
                "mean": MEAN, "std": STD, "enhancement": "none"}
    if any(prep.get(key) != value for key, value in expected.items()):
        raise ValueError("Metadata preprocessing differs from RGB224/ImageNet normalization")
    if checkpoint_sha256 is not None and metadata["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Metadata checkpoint SHA256 mismatch")
    return metadata


def choose_device(device="auto"):
    if str(device) == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, cuda or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def input_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_model(checkpoint=DEFAULT_CHECKPOINT, device="auto", metadata_path=None):
    """Return (eval_model, torch.device); safely and strictly load a pure state dict."""
    from networks.dan import DAN
    checkpoint = Path(checkpoint)
    if metadata_path is not None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        validate_metadata(metadata, sha256(checkpoint))
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or not weights or not all(
            isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("Expected a pure tensor state_dict, not a training checkpoint or module")
    model = DAN(num_class=2, num_head=4, pretrained=False)
    model.load_state_dict(weights, strict=True)
    device = choose_device(device)
    return model.to(device).eval(), device


def collect_images(source, recursive=False):
    source = Path(source)
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        entries = source.rglob("*") if recursive else source.iterdir()
        paths = sorted((p for p in entries if p.is_file() and p.suffix.lower() in EXTENSIONS),
                       key=lambda p: str(p.relative_to(source)).casefold())
    else:
        raise FileNotFoundError("Input image or directory does not exist")
    if not paths:
        raise ValueError("No supported images found")
    return paths


@torch.inference_mode()
def infer_logits(model, paths, device, batch_size=32):
    """Return CPU float32 N x 2 logits. Unreadable inputs raise explicitly."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    transform = input_transform()
    paths = list(paths)
    if not paths:
        raise ValueError("At least one cropped face image is required")
    outputs = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
        batch = torch.stack(images).to(device)
        logits = model(batch)[0]
        if logits.shape != (len(images), 2) or not torch.isfinite(logits).all():
            raise RuntimeError("Model returned invalid binary logits")
        outputs.append(logits.float().cpu())
    return torch.cat(outputs)


def predict_images(checkpoint, paths, device="auto", batch_size=32, metadata_path=None):
    """Predict a sequence of face crop paths; return JSON-serializable probabilities/labels."""
    paths = [Path(p) for p in paths]
    model, selected_device = load_model(checkpoint, device, metadata_path)
    z = infer_logits(model, paths, selected_device, batch_size)
    # Match the original experiment: compute two-class softmax on the CPU.
    probabilities = torch.softmax(z, dim=1).numpy()
    records = []
    for path, p in zip(paths, probabilities):
        label = int(p[1] > .5)
        records.append({"image_name": path.name, "image_path": str(path.resolve()),
                        "neutral_probability": float(p[0]), "positive_probability": float(p[1]),
                        "predicted_label": label, "predicted_emotion": LABELS[label]})
    return {"checkpoint_sha256": sha256(checkpoint), "device": str(selected_device),
            "class_names": LABELS, "decision": "positive_probability > 0.5",
            "preprocessing": "RGB; PIL bilinear 224x224; ToTensor; ImageNet normalize; no enhancement",
            "count": len(records), "predictions": records}


def predict(checkpoint, image_path, device="auto", metadata_path=None):
    return predict_images(checkpoint, [image_path], device, 1, metadata_path)["predictions"][0]


def write_outputs(payload, json_path=None, csv_path=None):
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        with temp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(payload["predictions"][0]))
            writer.writeheader()
            writer.writerows(payload["predictions"])
        temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="An already cropped face image or a directory of face crops")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, help="Optional model_metadata.json validation; no data/teacher dependency")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    paths = collect_images(args.input, args.recursive)
    payload = predict_images(args.checkpoint, paths, args.device, args.batch_size, args.metadata)
    write_outputs(payload, args.output_json, args.output_csv)
    if args.output_json is not None or args.output_csv is not None:
        print(json.dumps({"count": payload["count"], "device": payload["device"],
                          "outputs_saved": True, "checkpoint_sha256": payload["checkpoint_sha256"]}))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
