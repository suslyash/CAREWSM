"""Read-only CARE Parkinson error audit for existing DEEP-008 checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from chimera_ml.data.masking_collate import MaskingCollate
from audio.data.carewsm_audio_dataset import CAREWSMAudioDataset
from audio.data.carewsm_audio_datamodule import CAREWSMAudioDataModule
from audio.models.carewsm_audio_models import CAREWSMPooledMLPModel
from audio.models.carewsm_representation_pilot_models import CAREWSMGatedFusionModel


REPO_ROOT = Path(__file__).resolve().parents[1]
CARE_ROOT = Path("/media/maxim/Databases/CARE")
MANIFEST = REPO_ROOT / "reports/feature_audit/sample_feature_availability.csv"
REGISTRY = CARE_ROOT / "splits/features.json"
RUN_ROOT = REPO_ROOT / "logs/audio_representation_pilot/audio_representation_pilot_deep008"
OUTPUT = REPO_ROOT / "logs/audio_representation_pilot/deep008_error_audit/audit.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validation_records(
    fold: int,
    feature_family: str | list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build only CARE fold-validation records; no TEST split is read."""
    families = [feature_family] if isinstance(feature_family, str) else feature_family
    helper = CAREWSMAudioDataModule.__new__(CAREWSMAudioDataModule)
    helper.roots = {"CARE": CARE_ROOT}
    helper.feature_specs = helper._read_registry(REGISTRY)
    manifest = {
        row["sample_id"]: row for row in read_rows(MANIFEST) if row["corpus"] == "CARE"
    }
    source: dict[str, dict[str, str]] = {}
    for split_name in ("dev", "train"):
        for row in read_rows(CARE_ROOT / "splits" / f"{split_name}.csv"):
            source[row["sample_id"]] = row
    fold_rows = read_rows(CARE_ROOT / "splits/folds" / f"fold_{fold}_val.csv")

    records: list[dict[str, Any]] = []
    for fold_row in fold_rows:
        sample_id = fold_row["sample_id"]
        source_row = source[sample_id]
        record = dict(manifest[sample_id])
        record.update(helper._source_metadata("CARE", source_row))
        for target in ("depression", "parkinson"):
            raw = record.get(target, "")
            observed = bool(str(raw).strip())
            record[target] = float(raw) if observed else None
            record[f"{target}_observed"] = observed
        record["feature_paths"] = helper._feature_paths("CARE", source_row)
        if all(record.get(f"{family}_status") == "VALID" for family in families):
            records.append(record)

    records.sort(key=lambda item: (item["corpus"], item["sample_id"]))
    return records, helper.feature_specs


def selected_checkpoint(prefix: str, fold: int) -> Path:
    directories = sorted(RUN_ROOT.glob(f"{prefix}_fold{fold}_*"))
    if len(directories) != 1:
        raise RuntimeError(f"expected one runtime directory for {prefix} fold {fold}")
    with (directories[0] / "summary.txt").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    selected = max(rows, key=lambda row: float(row["val_mean_corpora_tasks_macro_recall"]))
    checkpoints = list(
        (directories[0] / "checkpoints").glob(f"epoch={selected['epoch']}_*.pt")
    )
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one selected checkpoint for {prefix} fold {fold}")
    return checkpoints[0]


def model_for(prefix: str) -> torch.nn.Module:
    if prefix == "A_native_mlp64":
        return CAREWSMPooledMLPModel(input_dim=1024, hidden_dim=64, dropout=0.1)
    if prefix == "F_gated_emotion_trillsson_native64":
        return CAREWSMGatedFusionModel(
            input_dim=2048, family_dims=[1024, 1024], hidden_dim=64, dropout=0.1
        )
    raise ValueError(prefix)


def infer(prefix: str, fold: int) -> tuple[dict[str, Any], dict[str, bool], dict[str, int], dict[str, int]]:
    families = "emotionmsp_embeds" if prefix == "A_native_mlp64" else ["emotionmsp_embeds", "trillsson4_1"]
    records, specs = validation_records(fold, families)
    dataset = CAREWSMAudioDataset(records, families, specs)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=MaskingCollate())
    checkpoint = selected_checkpoint(prefix, fold)
    model = model_for(prefix)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    predictions: dict[str, bool] = {}
    labels: dict[str, int] = {}
    lengths: dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            logits = model(batch).preds[:, 1]
            predicted = logits > 0
            observed = torch.stack([meta["observed_targets"] for meta in batch.meta["sample_meta"]])[:, 1].bool()
            sequence_lengths = batch.get_masks("sequence_mask").sum(dim=1)
            for index, meta in enumerate(batch.meta["sample_meta"]):
                if bool(observed[index]):
                    sample_id = meta["sample_id"]
                    predictions[sample_id] = bool(predicted[index])
                    labels[sample_id] = int(batch.targets[index, 1].item())
                    lengths[sample_id] = int(sequence_lengths[index].item())

    patients = [sample for sample, label in labels.items() if label == 1]
    healthy = [sample for sample, label in labels.items() if label == 0]
    tp = sum(predictions[sample] for sample in patients)
    fn = len(patients) - tp
    fp = sum(predictions[sample] for sample in healthy)
    tn = len(healthy) - fp
    summary = {
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "observed_patients": len(patients),
        "observed_healthy": len(healthy),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "patient_recall": tp / len(patients),
        "healthy_recall": tn / len(healthy),
        "macro_recall": (tp / len(patients) + tn / len(healthy)) / 2,
        "patient_tp_length_mean": sum(lengths[sample] for sample in patients if predictions[sample]) / max(tp, 1),
        "patient_fn_length_mean": sum(lengths[sample] for sample in patients if not predictions[sample]) / max(fn, 1),
    }
    return summary, predictions, labels, lengths


def paired_summary(
    a_predictions: dict[str, bool],
    f_predictions: dict[str, bool],
    labels: dict[str, int],
) -> dict[str, int]:
    patients = [sample for sample, label in labels.items() if label == 1]
    healthy = [sample for sample, label in labels.items() if label == 0]
    return {
        "patients_correct_both": sum(a_predictions[s] and f_predictions[s] for s in patients),
        "patients_missed_both": sum(not a_predictions[s] and not f_predictions[s] for s in patients),
        "patients_detected_only_a": sum(a_predictions[s] and not f_predictions[s] for s in patients),
        "patients_detected_only_f": sum(f_predictions[s] and not a_predictions[s] for s in patients),
        "healthy_false_positive_both": sum(a_predictions[s] and f_predictions[s] for s in healthy),
        "healthy_false_positive_only_a": sum(a_predictions[s] and not f_predictions[s] for s in healthy),
        "healthy_false_positive_only_f": sum(f_predictions[s] and not a_predictions[s] for s in healthy),
        "healthy_false_positive_either": sum(a_predictions[s] or f_predictions[s] for s in healthy),
    }


def main() -> None:
    audit: dict[str, Any] = {"folds": {}}
    for fold in range(5):
        a_summary, a_predictions, labels, lengths = infer("A_native_mlp64", fold)
        f_summary, f_predictions, f_labels, f_lengths = infer(
            "F_gated_emotion_trillsson_native64", fold
        )
        if labels != f_labels or set(a_predictions) != set(f_predictions):
            raise RuntimeError(f"A/F CARE validation populations differ on fold {fold}")
        audit["folds"][str(fold)] = {
            "A": a_summary,
            "F": f_summary,
            "paired": paired_summary(a_predictions, f_predictions, labels),
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote aggregate audit: {OUTPUT}")


if __name__ == "__main__":
    main()
