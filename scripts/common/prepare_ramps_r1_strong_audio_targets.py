#!/usr/bin/env python3
"""Prepare audited RAMPS R1 frozen-strong-audio missing-head targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from chimera_ml.core.batch import Batch
from common.callbacks.wsm_segment_callback import compute_sparse_two_task_metrics
from fusion.data.wsm_av_fusion_datamodule import WSMAVFusionDataModule
from fusion.loss.ramps_r1_teacher import (
    binary_brier,
    binary_ece,
    binary_nll,
    fit_binary_temperature,
    select_class_threshold,
)
from fusion.models.frozen_audio_temporal_adapter import FrozenAudioTemporalAdapter

TASK_NAMES = ("depression", "parkinson")
CHECKPOINT_SHA256 = "0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2"
VERSION = "ramps-r1-strong-audio-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        inputs={key: value.to(device) for key, value in batch.inputs.items()},
        targets=batch.targets.to(device) if batch.targets is not None else None,
        masks={
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in (batch.masks or {}).items()
        },
        meta=batch.meta,
    )


def _infer_dataset(
    teacher: FrozenAudioTemporalAdapter,
    dataset: Any,
    collate_fn: Any,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    observed: list[torch.Tensor] = []
    segment_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            segment_ids.extend(
                str(item["segment_id"]) for item in batch.meta["sample_meta"]
            )
            batch = _move_batch(batch, device)
            outputs = teacher(batch)
            logits.append(outputs["base_logits"].cpu())
            targets.append(batch.targets.cpu())
            observed.append(batch.get_masks("observed_mask").cpu())
    return {
        "segment_ids": segment_ids,
        "raw_logits": torch.cat(logits),
        "targets": torch.cat(targets),
        "observed_mask": torch.cat(observed).to(torch.bool),
    }


def _metric_record(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return {
        "nll": binary_nll(logits, targets),
        "brier": binary_brier(logits, targets),
        "ece": binary_ece(logits, targets, bins=15),
    }


def _stats(values: torch.Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(values.mean().item()), "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()), "max": float(values.max().item()),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        Path(handle.name).replace(path)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".tmp-", suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--audio-feature-cache-root", required=True)
    parser.add_argument("--video-cache-root", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--precision-target", type=float, default=0.90)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(42)
    torch.manual_seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; production inference is blocked")
    device = torch.device(args.device)
    checkpoint = Path(args.teacher_checkpoint).expanduser().resolve()
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("teacher checkpoint SHA256 does not match the required frozen checkpoint")
    output = Path(args.output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output root refuses overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = WSMAVFusionDataModule(
        data_root=args.data_root,
        audio_feature_cache_root=args.audio_feature_cache_root,
        video_cache_root=args.video_cache_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        shuffle_train=False, drop_last_train=False,
    )
    expected = {"train": 6325, "dev": 933, "test_none": 1364, "test_soft": 1208, "test_hard": 1014}
    if data.audit["counts"] != expected or data.audit["joined_total"] != 8622:
        raise RuntimeError(f"canonical counts failed: {data.audit}")
    teacher = FrozenAudioTemporalAdapter(str(checkpoint)).to(device)
    teacher.eval()
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("teacher is not frozen/eval-only")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("teacher has stale gradients")
    dev = _infer_dataset(teacher, data.val_dataset, data.collate_fn, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    calibration: dict[str, Any] = {
        "version": VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_checkpoint_path": str(checkpoint), "teacher_checkpoint_sha256": CHECKPOINT_SHA256,
        "adapter_source": "fusion.models.frozen_audio_temporal_adapter.FrozenAudioTemporalAdapter",
        "task_names": list(TASK_NAMES), "device": str(device),
        "data_root": data.data_root, "audio_feature_cache_root": str(data.audio_feature_cache_root),
        "video_cache_root": str(data.video_cache_root), "counts": data.audit["counts"],
        "joined_total": data.audit["joined_total"], "missing_audio": data.audit["missing_audio"],
        "missing_video": data.audit["missing_video"], "calibration_procedure": "binary BCE/NLL fit on observed DEV rows only",
        "threshold_procedure": "fixed class-specific precision-target threshold grid on observed DEV calibrated probabilities only",
        "precision_target": args.precision_target, "min_support": args.min_support,
        "threshold_step": args.threshold_step, "ece_bins": args.ece_bins,
        "test_statement": "No Test rows or Test metrics were used.",
    }
    thresholds: dict[str, Any] = {}
    temperatures: dict[str, float] = {}
    dev_task_records: dict[str, Any] = {}
    for task, name in enumerate(TASK_NAMES):
        selected = dev["observed_mask"][:, task]
        raw = dev["raw_logits"][selected, task]
        labels = dev["targets"][selected, task]
        fit = fit_binary_temperature(raw, labels)
        temperature = float(fit["temperature"])
        calibrated = raw / temperature
        raw_metrics = _metric_record(raw, labels)
        calibrated_metrics = {
            "nll": binary_nll(calibrated, labels),
            "brier": binary_brier(calibrated, labels),
            "ece": binary_ece(calibrated, labels, bins=args.ece_bins),
        }
        probabilities = torch.sigmoid(calibrated)
        positive_threshold = select_class_threshold(
            probabilities, labels.bool(), positive=True, precision_target=args.precision_target,
            min_support=args.min_support, threshold_step=args.threshold_step,
        )
        negative_threshold = select_class_threshold(
            probabilities, labels.bool(), positive=False, precision_target=args.precision_target,
            min_support=args.min_support, threshold_step=args.threshold_step,
        )
        thresholds[name] = {"positive": positive_threshold, "negative": negative_threshold}
        temperatures[name] = temperature
        dev_task_records[name] = {
            "observed_count": int(selected.sum()), "positive_count": int(labels.sum()),
            "negative_count": int((~labels.bool()).sum()), "temperature": fit,
            "raw_metrics": raw_metrics, "calibrated_metrics": calibrated_metrics,
            "positive_threshold": positive_threshold, "negative_threshold": negative_threshold,
        }
    calibration["temperatures"] = temperatures
    calibration["thresholds"] = thresholds
    calibration["dev_tasks"] = dev_task_records

    train = _infer_dataset(teacher, data.train_dataset, data.collate_fn, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    if len(train["segment_ids"]) != 6325 or len(set(train["segment_ids"])) != 6325:
        raise RuntimeError("TRAIN inference did not produce exactly 6325 unique canonical rows")
    if train["segment_ids"] != [str(item["meta"]["segment_id"]) for item in data.train_dataset._samples]:
        raise RuntimeError("TRAIN canonical order changed")
    calibrated_probs = torch.empty_like(train["raw_logits"])
    pseudo_targets = torch.full_like(train["raw_logits"], float("nan"))
    pseudo_accept = torch.zeros_like(train["observed_mask"])
    pseudo_reliability = torch.zeros_like(train["raw_logits"])
    pseudo_class = torch.full(train["observed_mask"].shape, -1, dtype=torch.int64)
    missing_audits: dict[str, Any] = {}
    for task, name in enumerate(TASK_NAMES):
        temperature = temperatures[name]
        probabilities = torch.sigmoid(train["raw_logits"][:, task] / temperature)
        calibrated_probs[:, task] = probabilities
        observed = train["observed_mask"][:, task]
        positive = torch.zeros_like(observed)
        negative = torch.zeros_like(observed)
        pos = thresholds[name]["positive"]
        neg = thresholds[name]["negative"]
        if pos["enabled"]:
            positive = (~observed) & (probabilities >= pos["threshold"])
        if neg["enabled"]:
            negative = (~observed) & (probabilities <= neg["threshold"])
        accepted = positive | negative
        pseudo_accept[:, task] = accepted
        pseudo_targets[accepted, task] = probabilities[accepted]
        pseudo_reliability[accepted, task] = 2.0 * (probabilities[accepted] - 0.5).abs()
        pseudo_class[positive, task] = 1
        pseudo_class[negative, task] = 0
        accepted_probabilities = probabilities[accepted]
        missing = ~observed
        missing_audits[name] = {
            "missing_count": int(missing.sum()), "accepted_total": int(accepted.sum()),
            "rejected_total": int((missing & ~accepted).sum()),
            "coverage": float(accepted.sum() / missing.sum()) if bool(missing.any()) else 0.0,
            "accepted_positive_count": int(positive.sum()), "accepted_negative_count": int(negative.sum()),
            "accepted_positive_fraction": float(positive.sum() / accepted.sum()) if bool(accepted.any()) else 0.0,
            "accepted_probability": _stats(accepted_probabilities),
            "accepted_reliability": _stats(pseudo_reliability[accepted, task]),
        }
    accepted_by_task = (pseudo_accept & ~train["observed_mask"]).any(dim=0)
    if not bool(accepted_by_task.all()):
        raise RuntimeError("R1 blocked: at least one missing disease head has no accepted pseudo-target")
    observed_overwrite = int((pseudo_accept & train["observed_mask"]).sum())
    pseudo_on_observed = int(torch.isfinite(pseudo_targets[train["observed_mask"]]).sum())
    accepted_values = pseudo_targets[pseudo_accept]
    nonfinite_accepted = int((~torch.isfinite(accepted_values)).sum()) if accepted_values.numel() else 0
    rejected_values = pseudo_targets[~pseudo_accept]
    if observed_overwrite or pseudo_on_observed or nonfinite_accepted:
        raise RuntimeError("pseudo-target invariant failed")
    if accepted_values.numel() and not bool(((accepted_values >= 0) & (accepted_values <= 1)).all()):
        raise RuntimeError("accepted pseudo-target outside [0,1]")
    if accepted_values.numel() and not bool(torch.isfinite(pseudo_reliability[pseudo_accept]).all()):
        raise RuntimeError("accepted reliability is nonfinite")
    if not bool((pseudo_reliability[~pseudo_accept] == 0).all()):
        raise RuntimeError("reliability must be zero on observed/rejected entries")
    if not bool(torch.isin(pseudo_class, torch.tensor([-1, 0, 1])).all()):
        raise RuntimeError("pseudo_class contains an invalid value")
    audit = {
        "version": VERSION, "train_rows": 6325,
        "train_missing_entries": int((~train["observed_mask"]).sum()),
        "accepted_missing_entries": int(pseudo_accept.sum()),
        "accepted_overall_coverage": float(pseudo_accept.sum() / (~train["observed_mask"]).sum()),
        "per_missing_task": missing_audits, "observed_overwrite_violations": observed_overwrite,
        "duplicate_segment_ids": 6325 - len(set(train["segment_ids"])),
        "nonfinite_accepted_targets": nonfinite_accepted,
        "pseudo_values_on_observed_entries": pseudo_on_observed,
        "pseudo_acceptance_on_observed_entries": observed_overwrite,
        "missing_labels_are_unknown": True,
        "no_missing_label_correctness_claim": True,
    }
    artifact = {
        "version": VERSION, "task_names": list(TASK_NAMES), "segment_ids": train["segment_ids"],
        "observed_mask": train["observed_mask"], "observed_targets": train["targets"],
        "raw_teacher_logits": train["raw_logits"], "calibrated_probs": calibrated_probs,
        "pseudo_accept_mask": pseudo_accept, "pseudo_targets": pseudo_targets,
        "pseudo_reliability": pseudo_reliability, "pseudo_class": pseudo_class,
        "temperatures": temperatures, "thresholds": thresholds,
        "teacher_checkpoint_path": str(checkpoint), "teacher_checkpoint_sha256": CHECKPOINT_SHA256,
        "adapter_source": "fusion.models.frozen_audio_temporal_adapter.FrozenAudioTemporalAdapter",
    }
    _atomic_json(output / "teacher_calibration.json", calibration)
    _atomic_torch(output / "train_missing_targets.pt", artifact)
    _atomic_json(output / "audit.json", audit)
    print(json.dumps({
        "device": str(device), "checkpoint_sha256": CHECKPOINT_SHA256,
        "temperatures": temperatures, "accepted_missing": audit["accepted_missing_entries"],
        "per_missing_task": missing_audits, "output_root": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
