#!/usr/bin/env python3
"""Run the fixed six-segment real DEPART video extraction audit."""
from __future__ import annotations
import argparse
import importlib.metadata
import os
import json
import subprocess
from pathlib import Path
import sys
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
from common.data.wsm_manifest import build_manifest
from video.features.clip_video_features import (
    ClipVideoFeatureExtractor, PREPROCESSING_VERSION, ROI_POLICY_VERSION,
    read_video_rgb, uniform_frame_indices, save_cache_artifact, build_cache_fingerprint,
)
from video.features.yolov8_body_roi import YOLOv8BodyDetector, weights_sha256

UPSTREAM_REPOSITORY = "J3lly-Been/YOLOv8-HumanDetection"
UPSTREAM_COMMIT = "ce2aae2e821100aee58ce2e7f75994a7e6c2ab9e"
UPSTREAM_PATH = "best.pt"
GIT_BLOB_SHA = "afa44d4fcd0ff54691912bf8960d4fbb98ae1278"

def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"

def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}

def _source_path(data_root: Path, row: dict[str, Any]) -> Path:
    corpus, video_id, segment_file = json.loads(row["segment_id"])
    return data_root / corpus / f"{row['split']}_labels" / video_id / "segments" / segment_file

def _artifact_validation(path: Path, local_sha: str, target_frames: int) -> dict[str, Any]:
    import torch
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    features, mask = artifact["features"], artifact["valid_mask"]
    assert path.is_file()
    assert features.ndim == 2 and mask.ndim == 1 and mask.dtype == torch.bool
    assert features.shape[0] == mask.shape[0] == target_frames
    assert torch.equal(features[~mask], torch.zeros_like(features[~mask]))
    assert bool(torch.isfinite(features[mask]).all())
    assert artifact["detector_weights_sha256"] == local_sha
    assert artifact["detection_parameters"] == {"confidence": 0.5, "iou": 0.5, "imgsz": 640}
    assert artifact["preprocessing"]["target_frames"] == target_frames
    assert artifact["cache_fingerprint"]
    indices = artifact["sampled_frame_indices"]
    assert len(indices) == target_frames and indices == sorted(indices)
    assert len(set(indices)) == target_frames
    assert len(artifact["selected_boxes"]) == target_frames
    return {
        "temporal_length": int(features.shape[0]),
        "valid_detection_count": int(artifact["valid_detection_count"]),
        "detection_coverage": float(artifact["detection_coverage"]),
        "cache_fingerprint": artifact["cache_fingerprint"],
    }

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--target-frames", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> int:
    args = _parse_args()
    if args.per_split != 3:
        raise ValueError("TASK-002C fixes --per-split to exactly 3")
    if args.target_frames != 60:
        raise ValueError("TASK-002C fixes --target-frames to exactly 60")
    if not args.yolo_weights.is_file():
        raise FileNotFoundError(f"YOLO weights file is missing: {args.yolo_weights}")
    local_sha = weights_sha256(args.yolo_weights)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    rows, audit = build_manifest(args.data_root)
    selected_by_split = {}
    selected = []
    for split in ("train", "dev"):
        eligible = sorted((row for row in rows if row["split"] == split and row["video_available"]), key=lambda row: row["segment_id"])
        if len(eligible) < args.per_split:
            raise RuntimeError(f"{split} has only {len(eligible)} eligible video rows; exactly 3 are required")
        chosen = eligible[:args.per_split]
        selected_by_split[split] = [row["segment_id"] for row in chosen]
        selected.extend(chosen)
    per_segment = []
    short_blockers = []
    for row in selected:
        source = _source_path(args.data_root, row)
        try:
            frame_count = len(read_video_rgb(source))
            sampled = uniform_frame_indices(frame_count, args.target_frames)
            if len(sampled) != args.target_frames:
                short_blockers.append({"segment_id": row["segment_id"], "frame_count": frame_count, "temporal_length": len(sampled)})
            per_segment.append({"segment_id": row["segment_id"], "split": row["split"], "source_path": str(source),
                                "source_frame_count": frame_count, "temporal_length": len(sampled)})
        except Exception as exc:
            per_segment.append({"segment_id": row["segment_id"], "split": row["split"], "source_path": str(source),
                                "failure": True, "failure_category": "video_read", "failure_reason": f"{type(exc).__name__}: {exc}"})
    checkpoint = {
        "upstream_repository": UPSTREAM_REPOSITORY, "upstream_commit": UPSTREAM_COMMIT,
        "upstream_path": UPSTREAM_PATH, "git_blob_sha": GIT_BLOB_SHA,
        "local_path": str(args.yolo_weights), "local_size_bytes": args.yolo_weights.stat().st_size,
        "local_sha256": local_sha, "local_sha256_authoritative": True,
        "ultralytics_version": _version("ultralytics"), "transformers_version": _version("transformers"),
        "torch_version": _version("torch"), "wsm_git": _git_state(),
    }
    base_report = {
        "schema_version": "wsm-depart-real-audit-v1", "checkpoint": checkpoint,
        "manifest_fingerprint": audit["manifest_fingerprint"]["value"],
        "selected_by_split": selected_by_split, "selected_count": len(selected),
        "requested_target_frames": args.target_frames, "device": args.device,
        "short_video_blockers": short_blockers, "per_segment": per_segment,
        "test_usage": {"test_rows_processed": False, "model_predictions_inspected": False,
                       "performance_metrics_inspected": False, "selection_or_tuning_performed": False},
    }
    if short_blockers:
        base_report.update({"status": "blocked_short_video", "success_count": 0, "failure_count": len(selected),
                            "all_success_artifacts_valid": False})
        args.report_output.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("short-video blocker recorded; extraction not started")
        return 2
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    detector = YOLOv8BodyDetector(args.yolo_weights, 0.5, 0.5, 640)
    extractor = ClipVideoFeatureExtractor(model_name=args.model_name, model_revision=args.model_revision,
                                          target_frames=args.target_frames, device=args.device, detector=detector)
    fingerprints = set()
    success_count = 0
    no_body_count = 0
    other_failure_count = 0
    coverages = []
    for item, row in zip(per_segment, selected):
        if item.get("failure"):
            other_failure_count += 1
            continue
        source = Path(item["source_path"])
        fingerprint = build_cache_fingerprint(
            manifest_fingerprint=base_report["manifest_fingerprint"], segment_id=row["segment_id"],
            source_path=str(source), model_name=args.model_name, model_revision=args.model_revision,
            target_frames=args.target_frames, preprocessing_version=PREPROCESSING_VERSION,
            detector_identity=detector.identity, detector_weights_sha256=local_sha,
            detector_confidence=0.5, detector_iou=0.5, detector_imgsz=640,
            roi_policy_version=ROI_POLICY_VERSION,
        )
        cache_path = args.cache_root / f"{fingerprint}.pt"
        if cache_path.exists() and not args.overwrite:
            item.update({"success": False, "failure_category": "cache_exists", "failure_reason": str(cache_path)})
            other_failure_count += 1
            continue
        result = extractor.extract(source)
        if result.frame_count:
            item["source_frame_count"] = result.frame_count
        if result.sampled_indices:
            item["temporal_length"] = len(result.sampled_indices)
        if not result.success:
            item.update({"success": False, "failure_category": result.failure_kind or "extraction_failure",
                         "failure_reason": result.error, "valid_detection_count": result.valid_detection_count,
                         "detection_coverage": result.detection_coverage})
            if result.failure_kind == "no_body_detected":
                no_body_count += 1
            else:
                other_failure_count += 1
            continue
        save_cache_artifact(cache_path, segment_id=row["segment_id"], source_path=str(source), result=result,
                            target_frames=args.target_frames, model_name=args.model_name,
                            model_revision=args.model_revision, cache_fingerprint=fingerprint, detector=detector)
        validation = _artifact_validation(cache_path, local_sha, args.target_frames)
        if fingerprint in fingerprints:
            raise AssertionError("cache fingerprint is not unique among sampled segments")
        fingerprints.add(fingerprint)
        item.update({"success": True, "cache_path": str(cache_path), **validation})
        success_count += 1
        coverages.append(result.detection_coverage)
    all_valid = all(item.get("success") and item.get("temporal_length") == args.target_frames for item in per_segment if item.get("success"))
    base_report.update({
        "status": "complete" if success_count else "partial_blocked_no_success",
        "per_segment": per_segment, "success_count": success_count, "failure_count": len(selected) - success_count,
        "no_body_detected_count": no_body_count, "other_extraction_failure_count": other_failure_count,
        "mean_detection_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
        "min_detection_coverage": min(coverages) if coverages else None,
        "max_detection_coverage": max(coverages) if coverages else None,
        "all_temporal_lengths_equal_60": all(item.get("temporal_length") == 60 for item in per_segment if item.get("success")),
        "all_success_artifacts_valid": all_valid, "cache_fingerprints_unique": len(fingerprints) == success_count,
    })
    args.report_output.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"success_count": success_count, "failure_count": len(selected) - success_count,
                      "mean_detection_coverage": base_report["mean_detection_coverage"]}, sort_keys=True))
    return 0 if success_count else 2

if __name__ == "__main__":
    raise SystemExit(main())
