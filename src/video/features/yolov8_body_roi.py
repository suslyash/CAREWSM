"""Local-only YOLOv8 single-body detection and deterministic ROI selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DETECTOR_FAMILY = "yolov8-single-human-body"
ROI_POLICY_VERSION = "depart-body-roi-v1"


@dataclass(frozen=True)
class BodyDetection:
    box: tuple[int, int, int, int]
    confidence: float


def weights_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_body_detection(
    frame: Any,
    boxes: Iterable[Iterable[float]],
    confidences: Iterable[float],
    confidence_threshold: float = 0.5,
) -> BodyDetection | None:
    """Select one clipped body box by confidence, area, then coordinates."""
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("frame must expose height and width")
    height, width = int(shape[0]), int(shape[1])
    candidates: list[tuple[float, int, tuple[int, int, int, int]]] = []
    for raw_box, raw_confidence in zip(boxes, confidences):
        confidence = float(raw_confidence)
        if confidence < confidence_threshold:
            continue
        values = list(raw_box)
        if len(values) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in values)
        clipped = (max(0, min(width, x1)), max(0, min(height, y1)),
                   max(0, min(width, x2)), max(0, min(height, y2)))
        area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
        if area <= 0:
            continue
        candidates.append((confidence, area, clipped))
    if not candidates:
        return None
    confidence, _, box = min(candidates, key=lambda item: (-item[0], -item[1], item[2]))
    return BodyDetection(box=box, confidence=confidence)


def _to_list(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


class YOLOv8BodyDetector:
    """YOLOv8 adapter that only loads an explicitly supplied local checkpoint."""

    def __init__(self, weights_path: str | Path | None, confidence: float = 0.5,
                 iou: float = 0.5, imgsz: int = 640, detector: Any = None) -> None:
        if detector is None:
            if weights_path is None:
                raise ValueError("a local YOLOv8 weights path is required")
            self.weights_path = Path(weights_path).expanduser().resolve()
            if not self.weights_path.is_file():
                raise FileNotFoundError(f"YOLOv8 weights file is missing: {self.weights_path}")
            self.weights_sha256 = weights_sha256(self.weights_path)
        else:
            self.weights_path = Path(weights_path).expanduser().resolve() if weights_path else None
            self.weights_sha256 = weights_sha256(self.weights_path) if self.weights_path else None
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1 or imgsz <= 0:
            raise ValueError("confidence and IoU must be in [0,1], imgsz must be positive")
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self._detector = detector

    @property
    def identity(self) -> str:
        return DETECTOR_FAMILY

    def _load(self) -> Any:
        if self._detector is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("ultralytics is required for YOLOv8 extraction") from exc
            self._detector = YOLO(str(self.weights_path))
        return self._detector

    def detect(self, frame: Any) -> BodyDetection | None:
        detector = self._load()
        if hasattr(detector, "predict"):
            results = detector.predict(frame, conf=self.confidence, iou=self.iou, imgsz=self.imgsz, verbose=False)
        else:
            results = detector(frame)
        if not results:
            return None
        result = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(getattr(result, "boxes", result), "xyxy", None)
        confidences = getattr(getattr(result, "boxes", result), "conf", None)
        if boxes is None or confidences is None:
            return None
        return select_body_detection(frame, _to_list(boxes), _to_list(confidences), self.confidence)
