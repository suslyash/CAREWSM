"""Video feature extraction utilities."""

from .yolov8_body_roi import BodyDetection, YOLOv8BodyDetector, select_body_detection, weights_sha256
from .clip_video_features import (
    ClipVideoFeatureExtractor,
    ExtractionResult,
    build_cache_fingerprint,
    uniform_frame_indices,
)

__all__ = [
    "ClipVideoFeatureExtractor",
    "ExtractionResult",
    "build_cache_fingerprint",
    "uniform_frame_indices",
    "BodyDetection",
    "YOLOv8BodyDetector",
    "select_body_detection",
    "weights_sha256",
]
