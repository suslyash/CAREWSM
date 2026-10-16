#!/usr/bin/env python3
"""Independently audit the DEPART-compatible full-frame-fallback video cache."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.data.wsm_manifest import build_manifest
from video.features.clip_video_features import PREPROCESSING_VERSION, ROI_POLICY_VERSION
from video.features.yolov8_body_roi import weights_sha256


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def _load_index(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"cache index is missing: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _validate_artifact(record: dict, args: argparse.Namespace) -> dict:
    import torch

    artifact = torch.load(Path(record["cache_path"]), map_location="cpu", weights_only=False)
    if artifact.get("segment_id") != record["segment_id"]:
        raise ValueError("segment_id mismatch")
    if artifact.get("cache_fingerprint") != record.get("cache_fingerprint"):
        raise ValueError("fingerprint mismatch")
    if artifact.get("model_name") != args.model_name or artifact.get("model_revision") != args.model_revision:
        raise ValueError("CLIP identity mismatch")
    if artifact.get("detector_weights_sha256") != args.yolo_sha:
        raise ValueError("YOLO SHA mismatch")
    if artifact.get("detection_parameters") != {"confidence": 0.5, "iou": 0.5, "imgsz": 640}:
        raise ValueError("detector settings mismatch")
    preprocessing = artifact.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("preprocessing metadata missing")
    if preprocessing.get("version") != PREPROCESSING_VERSION:
        raise ValueError("preprocessing version mismatch")
    if preprocessing.get("roi_policy_version") != ROI_POLICY_VERSION:
        raise ValueError("ROI policy version mismatch")
    if preprocessing.get("target_frames") != args.target_frames:
        raise ValueError("target_frames metadata mismatch")
    features, mask = artifact.get("features"), artifact.get("valid_mask")
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("feature shape mismatch")
    temporal_length = int(features.shape[0])
    if not 1 <= temporal_length <= args.target_frames:
        raise ValueError("temporal length out of range")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != (temporal_length,):
        raise ValueError("valid_mask mismatch")
    if not bool(mask.all()):
        raise ValueError("valid_mask is not all true")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("non-finite feature")
    sampled_indices = artifact.get("sampled_frame_indices")
    if not isinstance(sampled_indices, list) or len(sampled_indices) != temporal_length:
        raise ValueError("sampled indices mismatch")
    if any(not isinstance(index, int) or index < 0 for index in sampled_indices):
        raise ValueError("invalid sampled index")
    if any(left > right for left, right in zip(sampled_indices, sampled_indices[1:])):
        raise ValueError("sampled indices not chronological")
    sources = artifact.get("frame_sources")
    if not isinstance(sources, list) or len(sources) != temporal_length:
        raise ValueError("frame_sources mismatch")
    if any(source not in {"body_roi", "full_frame_fallback"} for source in sources):
        raise ValueError("invalid frame source")
    detected = int(artifact.get("detected_body_count", -1))
    fallback = int(artifact.get("full_frame_fallback_count", -1))
    if detected + fallback != temporal_length:
        raise ValueError("detection/fallback count mismatch")
    if detected != sources.count("body_roi") or fallback != sources.count("full_frame_fallback"):
        raise ValueError("detection/fallback provenance mismatch")
    boxes = artifact.get("selected_boxes")
    if not isinstance(boxes, list) or len(boxes) != temporal_length:
        raise ValueError("selected_boxes mismatch")
    for source, box in zip(sources, boxes, strict=True):
        if source == "body_roi" and box is None:
            raise ValueError("body ROI source has null box")
        if source == "full_frame_fallback" and box is not None:
            raise ValueError("fallback source has selected box")
    detection_coverage = float(artifact.get("detection_coverage", -1.0))
    fallback_coverage = float(artifact.get("fallback_coverage", -1.0))
    if detection_coverage != detected / temporal_length:
        raise ValueError("detection coverage mismatch")
    if fallback_coverage != fallback / temporal_length:
        raise ValueError("fallback coverage mismatch")
    return {
        "temporal_length": temporal_length, "detected_body_count": detected,
        "full_frame_fallback_count": fallback, "detection_coverage": detection_coverage,
        "fallback_coverage": fallback_coverage, "segments_with_fallback": fallback > 0,
    }


def _stats(records: list[dict]) -> dict:
    valid = [record for record in records if record.get("status") in {"extracted", "reused"}]
    detections = [float(record["detection_coverage"]) for record in valid]
    fallbacks = [float(record["fallback_coverage"]) for record in valid]
    lengths = [int(record["temporal_length"]) for record in valid]
    return {
        "expected": len(records),
        "success": len(valid),
        "failure": sum(record.get("status") == "failed" for record in records),
        "body_detections": sum(int(record.get("detected_body_count", 0)) for record in valid),
        "full_frame_fallbacks": sum(int(record.get("full_frame_fallback_count", 0)) for record in valid),
        "segments_with_fallback": sum(bool(record.get("segments_with_fallback", False)) for record in valid),
        "mean_detection_coverage": sum(detections) / len(detections) if detections else None,
        "min_detection_coverage": min(detections) if detections else None,
        "max_detection_coverage": max(detections) if detections else None,
        "mean_fallback_coverage": sum(fallbacks) / len(fallbacks) if fallbacks else None,
        "min_fallback_coverage": min(fallbacks) if fallbacks else None,
        "max_fallback_coverage": max(fallbacks) if fallbacks else None,
        "temporal_length": {
            "count": len(lengths), "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": sum(lengths) / len(lengths) if lengths else None,
            "shorter_than_60_count": sum(length < 60 for length in lengths),
            "exact_target_frames_count": sum(length == 60 for length in lengths),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--target-frames", type=int, required=True)
    args = parser.parse_args()
    requested = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not requested or len(set(requested)) != len(requested) or any(split not in {"train", "dev", "test"} for split in requested):
        raise ValueError("--splits must be a unique non-empty subset of train,dev,test")
    if args.target_frames != 60:
        raise ValueError("TASK-002K2 requires target_frames=60")
    if not args.yolo_weights.is_file():
        raise FileNotFoundError(args.yolo_weights)
    args.yolo_sha = weights_sha256(args.yolo_weights)
    rows, audit = build_manifest(args.data_root)
    selected = [row for row in rows if row["split"] in requested and row["video_available"]]
    expected_ids = {row["segment_id"] for row in selected}
    records = _load_index(args.cache_root / "cache_index.jsonl")
    selected_records = [record for record in records if record.get("segment_id") in expected_ids]
    selected_ids = [record.get("segment_id") for record in selected_records]
    duplicate_ids = len(selected_ids) != len(set(selected_ids))
    missing = expected_ids - set(selected_ids)
    artifact_errors = []
    validated = []
    for record in selected_records:
        if record.get("status") in {"extracted", "reused"}:
            try:
                validated.append({**record, **_validate_artifact(record, args)})
            except Exception as exc:
                artifact_errors.append({"segment_id": record.get("segment_id"), "reason": str(exc)})
    by_split = {split: _stats([record for record in validated if record.get("split") == split] + [record for record in selected_records if record.get("split") == split and record.get("status") == "failed"]) for split in requested}
    report = {
        "schema_version": "wsm-video-cache-audit-v2",
        "requested_splits": requested,
        "manifest_fingerprint": audit["manifest_fingerprint"]["value"],
        "expected_by_split": {split: sum(row["split"] == split for row in selected) for split in requested},
        "expected_total": len(selected),
        "indexed_selected_count": len(selected_records),
        "missing_record_count": len(missing),
        "missing_segment_ids": sorted(missing),
        "test_rows_indexed": sum(record.get("split") == "test" for record in records),
        "test_rows_processed": "test" in requested,
        "duplicate_segment_id": duplicate_ids,
        "cache_fingerprints_unique": len([record.get("cache_fingerprint") for record in selected_records]) == len({record.get("cache_fingerprint") for record in selected_records}),
        "successful_artifacts_valid": not artifact_errors,
        "artifact_errors": artifact_errors,
        "success_by_split": {split: by_split[split]["success"] for split in requested},
        "failed_by_split": {split: by_split[split]["failure"] for split in requested},
        "coverage_by_split": by_split,
        "coverage_overall": _stats(validated + [record for record in selected_records if record.get("status") == "failed"]),
        "failure_categories": {category: sum(record.get("failure_category") == category for record in selected_records) for category in sorted({record.get("failure_category") for record in selected_records if record.get("status") == "failed"})},
        "detector_weights_sha256": args.yolo_sha,
        "model_name": args.model_name, "model_revision": args.model_revision,
        "target_frames": args.target_frames,
        "preprocessing_version": PREPROCESSING_VERSION,
        "roi_policy_version": ROI_POLICY_VERSION,
        "git": _git_state(),
        "package_versions": {name: _version(name) for name in ("torch", "transformers", "ultralytics")},
        "test_usage": {"model_predictions_inspected": False, "performance_metrics_inspected": False, "selection_or_tuning_performed": False},
    }
    report["complete_for_requested_splits"] = (
        not missing and not duplicate_ids and not artifact_errors and
        len(selected_records) == len(expected_ids) and
        all(record.get("status") in {"extracted", "reused"} for record in selected_records)
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"expected_total": len(selected), "missing": len(missing), "complete": report["complete_for_requested_splits"]}, sort_keys=True))
    return 0 if report["complete_for_requested_splits"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
