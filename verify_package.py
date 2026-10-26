"""Verify this DAN package without training or full test inference.

Default checks are self-contained and run on CPU. --data-root optionally adds
one batch of 32 validation face crops. Private filenames are never reported.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predict import load_model, infer_logits, sha256, validate_metadata

ROOT = Path(__file__).resolve().parent
EXPECTED_CHECKPOINT_SHA = "fccef63f6464c678852224f7cbf24c9c6fb9b9e682f0fd15f774ec7df4b245b3"
SCALAR_METRICS = ["n", "accuracy", "positive_precision", "positive_recall", "positive_f1", "tn", "fp", "fn", "tp"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_aggregate(value, context):
    n, tn, fp, fn, tp = [int(value[k]) for k in ["n", "tn", "fp", "fn", "tp"]]
    require(n > 0 and min(tn, fp, fn, tp) >= 0 and tn+fp+fn+tp == n, f"Invalid counts: {context}")
    precision = tp / (tp+fp) if tp+fp else 0.
    recall = tp / (tp+fn) if tp+fn else 0.
    f1 = 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.
    expected = {"accuracy": (tn+tp)/n, "positive_precision": precision,
                "positive_recall": recall, "positive_f1": f1}
    for key, number in expected.items():
        require(abs(value[key] - number) < 1e-12, f"Confusion counts/metric disagree: {context}/{key}")
    if "confusion_matrix" in value:
        require(value["confusion_matrix"] == [[tn, fp], [fn, tp]], f"Confusion matrix differs: {context}")
    require(value.get("threshold", .5) == .5 and value.get("decision_operator", ">") == ">",
            f"Expected the fixed > 0.5 decision: {context}")
    return {key: value[key] for key in SCALAR_METRICS}


def check_manifest(path):
    manifest = read_json(path)
    require(isinstance(manifest.get("files"), dict) and manifest["files"], "Manifest files must be a nonempty path-to-SHA dictionary")
    for relative, expected in manifest["files"].items():
        item = Path(relative)
        require(not item.is_absolute() and item.parts and ".." not in item.parts, "Manifest contains an unsafe path")
        target = (ROOT / item).resolve()
        require(target.is_relative_to(ROOT.resolve()) and target.is_file(), f"Manifest file missing/outside package: {relative}")
        require(sha256(target) == expected, f"Package file SHA256 mismatch: {relative}")
    require(manifest["files"].get("best_transfer_model.pth") == EXPECTED_CHECKPOINT_SHA,
            "Manifest must pin the selected binary DAN checkpoint")
    return {"passed": True, "checked_files": len(manifest["files"]), "manifest_sha256": sha256(path)}


def validation_batch(model, device, data_root, expected_results, atol):
    require(atol > 0, "Absolute tolerance must be positive")
    folder = Path(data_root).resolve() / "validation"
    source = pd.read_csv(folder / "validation_dataset_blendshapes_masked.csv", dtype={"image_name": str})
    require(len(source) >= 32 and source.label.isin([0, 1]).all() and not source.image_name.duplicated().any(),
            "Validation CSV labels/names are invalid")
    sample = source.iloc[:32]
    paths = []
    for name in sample.image_name:
        filename = name if name.lower().endswith((".jpg", ".jpeg", ".png")) else name + ".jpg"
        path = (folder / filename).resolve()
        require(path.is_relative_to(folder) and path.is_file(), "A validation image is missing or outside its split")
        paths.append(path)
    logits = infer_logits(model, paths, device, batch_size=32).numpy()
    p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
    result = {"passed": True, "sample_count": 32, "split": "validation", "device": str(device),
              "sample_predictions_finite": bool(np.isfinite(p).all()), "expected_results_compared": False}
    if expected_results is not None:
        path = Path(expected_results)
        if path.is_dir():
            path = path / "transfer_validation_predictions.csv"
        require(path.suffix.lower() == ".csv", "Live comparison needs a private validation predictions CSV, not test/aggregate metrics")
        expected = pd.read_csv(path, dtype={"image_name": str}, float_precision="round_trip")
        require(len(expected) == len(source) and expected.image_name.equals(source.image_name) and expected.label.equals(source.label),
                "Expected prediction rows do not exactly align with the validation CSV")
        expected = expected.iloc[:32]
        expected_p = expected.positive_probability.to_numpy(dtype=np.float32)
        probability_error = float(np.abs(p - expected_p).max())
        disagreements = int(np.count_nonzero((p > .5) != (expected_p > .5)))
        require(probability_error <= atol, f"Validation probability error {probability_error:.9g} exceeds explicit atol {atol}")
        require(disagreements == 0, f"Validation decisions differ on {disagreements} samples")
        result.update({"expected_results_compared": True, "max_abs_probability_error": probability_error,
                       "decision_disagreements": disagreements, "absolute_tolerance": atol,
                       "comparison_note": "CPU and CUDA arithmetic can differ; this is a numerical/decision check, not a bit-exact claim."})
        if {"logit_neutral", "logit_positive"}.issubset(expected.columns):
            z = expected[["logit_neutral", "logit_positive"]].to_numpy(dtype=np.float32)
            result["max_abs_logit_error"] = float(np.abs(logits - z).max())
    return result


def verify(args):
    manifest_result = check_manifest(args.manifest)
    checkpoint = ROOT / "best_transfer_model.pth"
    require(sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA, "Selected checkpoint SHA changed")
    metadata = validate_metadata(read_json(args.metadata), EXPECTED_CHECKPOINT_SHA)
    aggregates = {}
    for split in ["validation", "test"]:
        path = ROOT / "results" / f"transfer_{split}_metrics.json"
        value = read_json(path)
        aggregates[split] = validate_aggregate(value, split)
        reference = metadata.get("reference_results", {}).get(split)
        if reference is not None:
            for key in SCALAR_METRICS:
                require(key in reference and abs(reference[key] - value[key]) < 1e-12,
                        f"Metadata/reference metrics differ: {split}/{key}")
    require(aggregates["test"]["n"] == 4293 and
            [aggregates["test"][k] for k in ["tn", "fp", "fn", "tp"]] == [3578, 158, 123, 434],
            "Historical selected model aggregate result changed")
    model, device = load_model(checkpoint, args.device, args.metadata)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(len(weights) == 275 and all(torch.equal(v.cpu(), weights[k]) for k, v in model.state_dict().items()),
            "Strict reconstructed model state differs from the 275-tensor checkpoint")
    result = {"passed": True, "manifest": manifest_result, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
              "metadata_sha256": sha256(args.metadata), "weights_only_safe_strict_load": True,
              "state_tensor_count": len(weights), "device": str(device), "reference_aggregate_metrics": aggregates,
              "scope": "File integrity, metadata, aggregate arithmetic and strict state reload; no training or full test inference.",
              "validation_batch": "not requested; provide --data-root to enable"}
    if args.data_root is not None:
        result["validation_batch"] = validation_batch(model, device, args.data_root, args.expected_results, args.atol)
    elif args.expected_results is not None:
        raise ValueError("--expected-results requires --data-root")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "package_manifest.json")
    parser.add_argument("--metadata", type=Path, default=ROOT / "model_metadata.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--data-root", type=Path, help="Optional private dataset root; reads only the first 32 validation face crops")
    parser.add_argument("--expected-results", type=Path, help="Optional private validation predictions CSV (never copied into the package)")
    parser.add_argument("--atol", type=float, default=1e-4, help="Explicit probability tolerance for cross-backend validation comparison")
    parser.add_argument("--output", type=Path, help="Optional report path; private-data checks must save outside this repository")
    args = parser.parse_args()
    if args.output is not None and args.data_root is not None and args.output.resolve().is_relative_to(ROOT.resolve()):
        parser.error("Save private-data verification reports outside the repository")
    try:
        result = verify(args)
    except Exception as exc:
        result = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
