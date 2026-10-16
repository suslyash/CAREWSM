#!/usr/bin/env python3
"""Extract resumable DEPART-style frozen CLIP features for selected splits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
from common.data.wsm_manifest import build_manifest
from video.features.clip_video_features import (
    PREPROCESSING_VERSION, ROI_POLICY_VERSION, SAMPLING_METHOD,
    ClipVideoFeatureExtractor, build_cache_fingerprint, save_cache_artifact,
)
from video.features.yolov8_body_roi import YOLOv8BodyDetector
PROTECTED_NAMES = {"src", "configs", "docs"}
VALID_SPLITS = {"train", "dev", "test"}
MANIFEST_COLUMNS = {"segment_id", "video_id", "speaker_id", "corpus", "split", "y_depression", "y_parkinson",
    "observed_depression", "observed_parkinson", "audio_available", "video_available", "text_available", "description_available"}

def _refuse_protected(path: Path, option: str) -> None:
    resolved = path.resolve()
    if PROJECT_ROOT in resolved.parents:
        relative = resolved.relative_to(PROJECT_ROOT)
        if relative.parts and relative.parts[0] in PROTECTED_NAMES:
            raise ValueError(f"{option} may not write inside a protected project directory: {resolved}")

def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = MANIFEST_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]

def _source_path(data_root: Path, row: dict[str, Any]) -> Path:
    corpus, video_id, segment_file = json.loads(row["segment_id"])
    return data_root / corpus / f"{row['split']}_labels" / video_id / "segments" / segment_file

def _parse_splits(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--splits must contain at least one split")
    if any(item not in VALID_SPLITS for item in values):
        raise ValueError(f"--splits accepts only train,dev,test; got {values}")
    if len(values) != len(set(values)):
        raise ValueError("--splits must not repeat a split")
    return values

def _load_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["segment_id"]] = record
    return records

def _write_index(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".cache_index.", suffix=".jsonl", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for key in records:
                handle.write(json.dumps(records[key], sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def _validate_artifact(path: Path, expected: dict[str, Any], target_frames: int, local_sha: str,
                       model_name: str, model_revision: str) -> dict[str, Any]:
    import torch
    if not path.is_file():
        raise ValueError("cache artifact is missing")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("segment_id") != expected["segment_id"]:
        raise ValueError("segment_id mismatch")
    if artifact.get("cache_fingerprint") != expected["cache_fingerprint"]:
        raise ValueError("cache_fingerprint mismatch")
    features, mask = artifact.get("features"), artifact.get("valid_mask")
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("features must have shape [T,512]")
    temporal_length = int(features.shape[0])
    if not 1 <= temporal_length <= target_frames:
        raise ValueError(f"features temporal length must be in [1,{target_frames}]")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != (temporal_length,):
        raise ValueError("valid_mask must be bool [T]")
    sampled_indices = artifact.get("sampled_frame_indices")
    if not isinstance(sampled_indices, list) or len(sampled_indices) != temporal_length:
        raise ValueError("sampled_frame_indices must have length T")
    if any(not isinstance(index, int) or index < 0 for index in sampled_indices):
        raise ValueError("sampled_frame_indices must contain non-negative integers")
    if any(left > right for left, right in zip(sampled_indices, sampled_indices[1:])):
        raise ValueError("sampled_frame_indices must be chronological")
    preprocessing = artifact.get("preprocessing")
    if not isinstance(preprocessing, dict) or preprocessing.get("target_frames") != target_frames:
        raise ValueError("preprocessing target_frames mismatch")
    if preprocessing.get("version") != PREPROCESSING_VERSION:
        raise ValueError("preprocessing version mismatch")
    if preprocessing.get("roi_policy_version") != ROI_POLICY_VERSION:
        raise ValueError("ROI policy version mismatch")
    if not bool(mask.all()):
        raise ValueError("valid_mask must be all true for readable sampled frames")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must be finite")
    frame_sources = artifact.get("frame_sources")
    if not isinstance(frame_sources, list) or len(frame_sources) != temporal_length:
        raise ValueError("frame_sources must have length T")
    if any(source not in {"body_roi", "full_frame_fallback"} for source in frame_sources):
        raise ValueError("invalid frame source")
    detected_count = int(artifact.get("detected_body_count", -1))
    fallback_count = int(artifact.get("full_frame_fallback_count", -1))
    if detected_count + fallback_count != temporal_length:
        raise ValueError("detection/fallback counts do not equal T")
    if detected_count != frame_sources.count("body_roi") or fallback_count != frame_sources.count("full_frame_fallback"):
        raise ValueError("detection/fallback counts disagree with frame_sources")
    if artifact.get("valid_detection_count") != detected_count:
        raise ValueError("valid_detection_count must equal detected_body_count")
    if artifact.get("selected_boxes") is not None:
        boxes = artifact["selected_boxes"]
        if len(boxes) != temporal_length:
            raise ValueError("selected_boxes must have length T")
        for source, box in zip(frame_sources, boxes, strict=True):
            if source == "body_roi" and box is None:
                raise ValueError("body_roi frame must have a selected box")
            if source == "full_frame_fallback" and box is not None:
                raise ValueError("fallback frame must have no selected box")
    detection_coverage = float(artifact.get("detection_coverage", -1.0))
    fallback_coverage = float(artifact.get("fallback_coverage", -1.0))
    if detection_coverage != detected_count / temporal_length:
        raise ValueError("detection coverage mismatch")
    if fallback_coverage != fallback_count / temporal_length:
        raise ValueError("fallback coverage mismatch")
    if artifact.get("detector_weights_sha256") != local_sha:
        raise ValueError("detector weights SHA-256 mismatch")
    if artifact.get("detection_parameters") != {"confidence": 0.5, "iou": 0.5, "imgsz": 640}:
        raise ValueError("detector settings mismatch")
    if artifact.get("model_name") != model_name or artifact.get("model_revision") != model_revision:
        raise ValueError("CLIP model identity mismatch")
    return {"temporal_length": temporal_length, "feature_dim": 512,
            "valid_detection_count": int(artifact["valid_detection_count"]),
            "detected_body_count": int(artifact["detected_body_count"]),
            "full_frame_fallback_count": int(artifact["full_frame_fallback_count"]),
            "detection_coverage": float(artifact["detection_coverage"]),
            "fallback_coverage": float(artifact["fallback_coverage"]),
            "fallback_segment": int(artifact["full_frame_fallback_count"]) > 0}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--yolo-weights", type=Path)
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--target-frames", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--raw-frame-debug", action="store_true")
    return parser.parse_args()

def main() -> int:
    args = _parse_args()
    try:
        requested_splits = _parse_splits(args.splits)
        _refuse_protected(args.cache_root, "--cache-root")
        _refuse_protected(args.report_output, "--report-output")
        if args.target_frames != 60:
            raise ValueError("TASK-002G requires --target-frames 60")
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        if not args.raw_frame_debug:
            if args.yolo_weights is None or not args.yolo_weights.is_file():
                raise FileNotFoundError("--yolo-weights must name an existing local checkpoint")
            os.environ.setdefault("YOLO_AUTOINSTALL", "false")
            detector = YOLOv8BodyDetector(args.yolo_weights, args.yolo_conf, args.yolo_iou, args.yolo_imgsz)
        else:
            detector = None
        if args.manifest_path is None:
            rows, audit = build_manifest(args.data_root)
            manifest_path = Path("<canonical-manifest-built-from-data-root>")
            manifest_fingerprint = audit["manifest_fingerprint"]["value"]
        else:
            manifest_path = args.manifest_path
            rows = _manifest_rows(manifest_path)
            manifest_fingerprint = _file_fingerprint(manifest_path)
        candidates = [row for row in rows if row["split"] in requested_splits and
                      (row.get("video_available") is True or str(row.get("video_available", "")).lower() == "true")]
        if args.limit is not None:
            candidates = candidates[:args.limit]
        candidate_counts = {split: sum(row["split"] == split for row in candidates) for split in requested_splits}
        index_path = args.cache_root / "cache_index.jsonl"
        index = _load_index(index_path) if args.resume else {}
        extractor = ClipVideoFeatureExtractor(model_name=args.model_name, model_revision=args.model_revision,
            target_frames=args.target_frames, device=args.device, detector=detector, raw_frame_debug=args.raw_frame_debug)
        local_sha = None if detector is None else detector.weights_sha256
        extracted, reused, failures = [], [], []
        for row in candidates:
            segment_id = row["segment_id"]
            source = _source_path(args.data_root, row)
            fingerprint = build_cache_fingerprint(
                manifest_fingerprint=manifest_fingerprint, segment_id=segment_id, source_path=str(source),
                model_name=args.model_name, model_revision=args.model_revision, target_frames=args.target_frames,
                preprocessing_version=PREPROCESSING_VERSION, sampling_method=SAMPLING_METHOD,
                detector_identity=None if detector is None else detector.identity,
                detector_weights_sha256=local_sha, detector_confidence=None if detector is None else detector.confidence,
                detector_iou=None if detector is None else detector.iou, detector_imgsz=None if detector is None else detector.imgsz,
                roi_policy_version=ROI_POLICY_VERSION)
            cache_path = args.cache_root / f"{fingerprint}.pt"
            expected = {"segment_id": segment_id, "cache_fingerprint": fingerprint}
            if args.resume and cache_path.exists():
                try:
                    details = _validate_artifact(cache_path, expected, args.target_frames, local_sha, args.model_name, args.model_revision)
                    record = {"segment_id": segment_id, "split": row["split"], "source_path": str(source), "status": "reused",
                              "cache_path": str(cache_path), "cache_fingerprint": fingerprint, **details}
                    index[segment_id] = record; reused.append(record); _write_index(index_path, index)
                    continue
                except Exception as exc:
                    if not args.overwrite:
                        record = {"segment_id": segment_id, "split": row["split"], "source_path": str(source), "status": "failed",
                                  "cache_path": None, "cache_fingerprint": fingerprint, "failure_category": "invalid_cache",
                                  "failure_reason": str(exc), "temporal_length": None, "feature_dim": None}
                        index[segment_id] = record; failures.append(record); _write_index(index_path, index)
                        continue
            result = extractor.extract(source)
            if not result.success:
                record = {"segment_id": segment_id, "split": row["split"], "source_path": str(source), "status": "failed",
                          "cache_path": None, "cache_fingerprint": fingerprint, "failure_category": result.failure_kind or "extraction_failure",
                          "failure_reason": result.error, "temporal_length": len(result.sampled_indices or []), "feature_dim": None,
                          "valid_detection_count": result.valid_detection_count, "detection_coverage": result.detection_coverage}
                index[segment_id] = record; failures.append(record); _write_index(index_path, index)
                continue
            try:
                save_cache_artifact(cache_path, segment_id=segment_id, source_path=str(source), result=result,
                    target_frames=args.target_frames, model_name=args.model_name, model_revision=args.model_revision,
                    cache_fingerprint=fingerprint, detector=detector)
                details = _validate_artifact(cache_path, expected, args.target_frames, local_sha, args.model_name, args.model_revision)
            except Exception as exc:
                if cache_path.exists():
                    cache_path.unlink()
                record = {"segment_id": segment_id, "split": row["split"], "source_path": str(source), "status": "failed",
                          "cache_path": None, "cache_fingerprint": fingerprint, "failure_category": "invalid_extraction_contract",
                          "failure_reason": str(exc), "temporal_length": int(result.valid_mask.shape[0]) if result.valid_mask is not None else None,
                          "feature_dim": int(result.features.shape[1]) if result.features is not None and result.features.ndim == 2 else None}
                index[segment_id] = record; failures.append(record); _write_index(index_path, index)
                continue
            record = {"segment_id": segment_id, "split": row["split"], "source_path": str(source), "status": "extracted",
                      "cache_path": str(cache_path), "cache_fingerprint": fingerprint, **details}
            index[segment_id] = record; extracted.append(record); _write_index(index_path, index)
        def cover(items):
            successful = [x for x in items if x.get("status") in {"extracted", "reused"}]
            detections = [float(x["detection_coverage"]) for x in successful]
            fallbacks = [float(x["fallback_coverage"]) for x in successful]
            return {
                "count": len(successful),
                "mean_detection_coverage": sum(detections) / len(detections) if detections else None,
                "min_detection_coverage": min(detections) if detections else None,
                "max_detection_coverage": max(detections) if detections else None,
                "mean_fallback_coverage": sum(fallbacks) / len(fallbacks) if fallbacks else None,
                "min_fallback_coverage": min(fallbacks) if fallbacks else None,
                "max_fallback_coverage": max(fallbacks) if fallbacks else None,
                "segments_with_fallback": sum(x.get("fallback_segment", False) for x in successful),
            }
        all_records = [index[row["segment_id"]] for row in candidates if row["segment_id"] in index]
        test_rows_selected = sum(row["split"] == "test" for row in candidates)
        test_rows_indexed = sum(record.get("split") == "test" for record in index.values())
        report = {"schema_version": "wsm-depart-video-cache-report-v3", "manifest_path": str(manifest_path),
                  "manifest_fingerprint": manifest_fingerprint, "requested_splits": requested_splits,
                  "candidate_counts_by_split": candidate_counts, "test_rows_selected": test_rows_selected,
                  "test_rows_processed": test_rows_selected > 0, "test_rows_indexed": test_rows_indexed,
                  "cache_root": str(args.cache_root),
                  "cache_index": str(index_path), "cache_fingerprints": [x["cache_fingerprint"] for x in all_records],
                  "model_name": args.model_name, "model_revision": args.model_revision,
                  "detector_identity": None if detector is None else detector.identity, "detector_weights_sha256": local_sha,
                  "detection_parameters": {"confidence": args.yolo_conf, "iou": args.yolo_iou, "imgsz": args.yolo_imgsz},
                  "target_frames": args.target_frames, "preprocessing_version": PREPROCESSING_VERSION,
                  "sampling_method": SAMPLING_METHOD, "roi_policy_version": ROI_POLICY_VERSION,
                  "limit": args.limit, "resume": args.resume, "selected_count": len(candidates),
                  "extracted_success_count": len(extracted), "reused_success_count": len(reused),
                  "failure_count": len(failures), "no_body_detected_count": 0,
                  "other_failure_count": len(failures),
                  "success_by_split": {split: sum(x.get("status") in {"extracted", "reused"} and x.get("split") == split for x in all_records) for split in requested_splits},
                  "failure_by_split": {split: sum(x.get("status") == "failed" and x.get("split") == split for x in all_records) for split in requested_splits},
                  "coverage_by_split": {split: cover([x for x in all_records if x.get("split") == split]) for split in requested_splits},
                  "coverage_overall": cover(all_records), "records": all_records,
                  "test_metrics_inspected": False, "labels_passed_to_encoder": False}
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"selected_count": len(candidates), "extracted_success_count": len(extracted),
                          "reused_success_count": len(reused), "failure_count": len(failures)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"extract_clip_video_features: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
if __name__ == "__main__":
    raise SystemExit(main())
