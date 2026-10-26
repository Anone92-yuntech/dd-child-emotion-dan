"""Independently recompute saved DAN experiment metrics (no new inference)."""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, classification_report)

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT.parent / "emotion_dataset"
RESULTS = ROOT / "results"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_recorded_source(path, recorded_hash):
    """Retain original run provenance across the recorded dataset-path migration."""
    current_hash = digest(path)
    if current_hash == recorded_hash:
        return "exact original source"
    migration = json.loads((ROOT.parent / "dataset_migration" / "migration_manifest.json").read_text(encoding="utf-8"))
    entry = migration["files"][path.relative_to(ROOT.parent).as_posix()]
    assert entry["before_sha256"] == recorded_hash
    assert entry["after_sha256"] == current_hash
    assert digest(Path(entry["backup_path"])) == recorded_hash
    assert Path(migration["dataset_root"]).resolve() == DATASET_ROOT.resolve()
    return "original source archived; dataset-path migration hashes verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify without replacing the historical verification JSON")
    args = parser.parse_args()
    verified = {}
    for name, split, positive_column in [
        ("baseline_test", "test", "logit_happy"),
        ("transfer_validation", "validation", "logit_positive"),
        ("transfer_test", "test", "logit_positive"),
    ]:
        predictions = pd.read_csv(RESULTS / f"{name}_predictions.csv")
        source = pd.read_csv(next((DATASET_ROOT / split).glob("*.csv")))
        assert predictions.image_name.tolist() == source.image_name.tolist(), name
        assert np.array_equal(predictions.label, source.label), name
        direct_decisions = (predictions[positive_column] > predictions.logit_neutral).astype(int)
        assert np.array_equal(direct_decisions, predictions.prediction), f"{name}: logit vs probability decisions differ"
        assert np.array_equal((predictions.positive_probability > .5).astype(int), predictions.prediction)
        assert np.isfinite(predictions.select_dtypes(include="number").to_numpy()).all()
        assert predictions.positive_probability.between(0, 1).all()
        assert np.array_equal(predictions.correct, predictions.label == predictions.prediction)
        gap = predictions[positive_column].to_numpy() - predictions.logit_neutral.to_numpy()
        reconstructed_probability = 1 / (1 + np.exp(-np.clip(gap, -700, 700)))
        assert np.allclose(predictions.positive_probability, reconstructed_probability, atol=2e-7, rtol=0)
        values = json.loads((RESULTS / f"{name}_metrics.json").read_text(encoding="utf-8"))
        calculated = {"accuracy": accuracy_score(source.label, predictions.prediction),
                      "positive_precision": precision_score(source.label, predictions.prediction),
                      "positive_recall": recall_score(source.label, predictions.prediction),
                      "positive_f1": f1_score(source.label, predictions.prediction),
                      "macro_f1": f1_score(source.label, predictions.prediction, average="macro"),
                      "roc_auc": roc_auc_score(source.label, predictions.positive_probability),
                      "average_precision": average_precision_score(source.label, predictions.positive_probability)}
        for key, value in calculated.items():
            assert abs(value - values[key]) < 1e-12, (name, key)
        assert values["n"] == len(source)
        assert confusion_matrix(source.label, predictions.prediction, labels=[0, 1]).tolist() == values["confusion_matrix"]
        recalculated_report = classification_report(source.label, predictions.prediction, labels=[0, 1],
                                                     target_names=["neutral", "positive"], output_dict=True, zero_division=0)
        for report_key, report_value in recalculated_report.items():
            if isinstance(report_value, dict):
                for metric_key, metric_value in report_value.items():
                    assert abs(values["classification_report"][report_key][metric_key] - metric_value) < 1e-12
            else:
                assert abs(values["classification_report"][report_key] - report_value) < 1e-12
        verified[name] = {"rows": len(source), **calculated, "logit_decisions_match": True, "metrics_recomputed_exact": True}
    history = pd.read_csv(RESULTS / "training_history.csv")
    assert len(history) == 20
    assert history.groupby("stage").size().to_dict() == {1: 5, 2: 15}
    assert np.isfinite(history.select_dtypes(include="number").to_numpy()).all()
    best = history[history.stage == 2].sort_values(["validation_positive_f1", "global_epoch"], ascending=[False, True]).iloc[0]
    selected = json.loads((RESULTS / "best_checkpoint.json").read_text(encoding="utf-8"))
    assert int(best.global_epoch) == selected["global_epoch"]
    assert abs(best.validation_positive_f1 - verified["transfer_validation"]["positive_f1"]) < 1e-12
    assert digest(ROOT / "best_transfer_model.pth") == selected["sha256"]
    audit = json.loads((RESULTS / "data_audit.json").read_text(encoding="utf-8"))
    assert not audit["cross_split_byte_duplicates"]
    assert not any(audit["video_group_intersections"].values())
    assert digest(ROOT / "affecnet7_epoch6_acc0.6569.pth") == audit["pretrained_sha256"]
    config = json.loads((RESULTS / "experiment_config.json").read_text(encoding="utf-8"))
    source_verification = verify_recorded_source(ROOT / "run_experiment.py", config["script_sha256"])
    assert digest(ROOT / "networks" / "dan.py") == config["dan_source_sha256"]
    for split, summary in audit["splits"].items():
        assert digest(next((DATASET_ROOT / split).glob("*.csv"))) == summary["csv_sha256"]
    output = {"status": "passed", "saved_prediction_rechecks": verified,
              "completed_epochs": len(history), "best_global_epoch": selected["global_epoch"],
              "checkpoint_and_source_history_verified": True, "source_verification": source_verification,
              "no_cross_split_identical_image_bytes_or_inferred_video_groups": True,
              "method": "independent sklearn recomputation from saved predictions; no new inference"}
    if not args.check_only:
        (RESULTS / "verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
