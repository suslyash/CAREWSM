"""Chimera data consumer for the canonical WSM partial-label manifest."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule
from common.data.wsm_manifest import MANIFEST_COLUMNS, build_manifest


MODALITY_COLUMNS = (
    "audio_available",
    "video_available",
    "text_available",
    "description_available",
)
TASK_NAMES = ("depression", "parkinson")


class ManifestDataError(ValueError):
    """Raised when a canonical manifest violates the consumer contract."""


def _parse_nullable_target(value: str, field: str) -> float | None:
    value = value.strip()
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ManifestDataError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed) or parsed not in (0.0, 1.0):
        raise ManifestDataError(f"{field} must be missing, 0, or 1: {value!r}")
    return parsed


def _parse_bool(value: str, field: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ManifestDataError(f"{field} must be true or false: {value!r}")


def _validate_row(row: dict[str, str], line_number: int) -> dict[str, Any]:
    missing = [column for column in MANIFEST_COLUMNS if column not in row]
    if missing:
        raise ManifestDataError(f"manifest row {line_number} is missing columns: {missing}")
    if not row["segment_id"] or not row["video_id"] or not row["corpus"] or not row["split"]:
        raise ManifestDataError(f"manifest row {line_number} has an empty identity field")
    if row["corpus"] not in {"depression", "parkinson"}:
        raise ManifestDataError(f"manifest row {line_number} has invalid corpus")
    if row["split"] not in {"train", "dev", "test"}:
        raise ManifestDataError(f"manifest row {line_number} has invalid split")
    speaker_id = row["speaker_id"].strip() or None
    targets = [
        _parse_nullable_target(row["y_depression"], "y_depression"),
        _parse_nullable_target(row["y_parkinson"], "y_parkinson"),
    ]
    observed = [
        _parse_bool(row["observed_depression"], "observed_depression"),
        _parse_bool(row["observed_parkinson"], "observed_parkinson"),
    ]
    if any(observed[index] != (targets[index] is not None) for index in range(2)):
        raise ManifestDataError(f"manifest row {line_number} observed mask does not match targets")
    if not any(observed):
        raise ManifestDataError(f"manifest row {line_number} has no observed disease target")
    expected_observed = [row["corpus"] == "depression", row["corpus"] == "parkinson"]
    if observed != expected_observed:
        raise ManifestDataError(f"manifest row {line_number} has incorrect corpus ownership mask")
    modalities = [
        _parse_bool(row[column], column)
        for column in MODALITY_COLUMNS
    ]
    return {
        "segment_id": row["segment_id"],
        "video_id": row["video_id"],
        "speaker_id": speaker_id,
        "corpus": row["corpus"],
        "split": row["split"],
        "targets": targets,
        "observed_mask": observed,
        "modality_available": modalities,
    }


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ManifestDataError(f"manifest file is missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or list(reader.fieldnames) != list(MANIFEST_COLUMNS):
            raise ManifestDataError(
                f"manifest columns must be exactly {list(MANIFEST_COLUMNS)}"
            )
        rows = [_validate_row(row, line_number) for line_number, row in enumerate(reader, start=2)]
    if len({row["segment_id"] for row in rows}) != len(rows):
        raise ManifestDataError("manifest segment_id values are not unique")
    return rows


class WSMManifestDataset(Sequence[dict[str, Any]]):
    """Small in-memory dataset exposing masked two-task contract fields."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        targets = torch.tensor(
            [value if value is not None else float("nan") for value in row["targets"]],
            dtype=torch.float32,
        )
        observed_mask = torch.tensor(row["observed_mask"], dtype=torch.bool)
        modality_available = torch.tensor(row["modality_available"], dtype=torch.bool)
        return {
            "segment_id": row["segment_id"],
            "video_id": row["video_id"],
            "speaker_id": row["speaker_id"],
            "corpus": row["corpus"],
            "split": row["split"],
            "targets": targets,
            "observed_mask": observed_mask,
            "modality_available": modality_available,
        }


def collate_wsm_manifest(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ManifestDataError("cannot collate an empty sample list")
    return {
        "segment_id": [sample["segment_id"] for sample in samples],
        "video_id": [sample["video_id"] for sample in samples],
        "speaker_id": [sample["speaker_id"] for sample in samples],
        "corpus": [sample["corpus"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "targets": torch.stack([sample["targets"] for sample in samples]),
        "observed_mask": torch.stack([sample["observed_mask"] for sample in samples]),
        "modality_available": torch.stack([sample["modality_available"] for sample in samples]),
    }


@dataclass
class WSMManifestDataModule(DataModule):
    data_root: str = "/media/maxim/Databases/WSM_NEW"
    manifest_path: str | None = None
    manifest_audit_path: str | None = None

    def __post_init__(self) -> None:
        if self.manifest_path is None:
            canonical_rows, generated_audit = build_manifest(self.data_root)
            manifest_rows = [self._normalize_builder_row(row) for row in canonical_rows]
            self.manifest_audit = generated_audit
        else:
            manifest_rows = load_manifest_rows(self.manifest_path)
            self.manifest_audit = self._load_audit(self.manifest_audit_path)
        self.rows = manifest_rows
        self.train_dataset = WSMManifestDataset(self._rows_for_split("train"))
        self.val_dataset = WSMManifestDataset(self._rows_for_split("dev"))
        self.test_dataset = WSMManifestDataset(self._rows_for_split("test"))
        self.collate_fn = collate_wsm_manifest
        self.speaker_independence_verified = bool(
            self.manifest_audit.get("speaker_independence_verified", False)
        )
        if self.speaker_independence_verified:
            raise ManifestDataError("speaker independence must remain unverified")
        self.manifest_schema = self.manifest_audit.get("schema")
        self.manifest_version = self.manifest_audit.get("version")

    @staticmethod
    def _normalize_builder_row(row: dict[str, Any]) -> dict[str, Any]:
        raw = {
            "segment_id": row["segment_id"],
            "video_id": row["video_id"],
            "speaker_id": row["speaker_id"] or "",
            "corpus": row["corpus"],
            "split": row["split"],
            "y_depression": "" if row["y_depression"] is None else str(row["y_depression"]),
            "y_parkinson": "" if row["y_parkinson"] is None else str(row["y_parkinson"]),
            "observed_depression": str(row["observed_depression"]),
            "observed_parkinson": str(row["observed_parkinson"]),
            "audio_available": str(row["audio_available"]),
            "video_available": str(row["video_available"]),
            "text_available": str(row["text_available"]),
            "description_available": str(row["description_available"]),
        }
        return _validate_row(raw, 0)

    def _rows_for_split(self, split: str) -> list[dict[str, Any]]:
        return [row for row in self.rows if row["split"] == split]

    @staticmethod
    def _load_audit(path: str | None) -> dict[str, Any]:
        if path is None:
            return {}
        audit_path = Path(path).expanduser().resolve()
        if not audit_path.is_file():
            raise ManifestDataError(f"manifest audit file is missing: {audit_path}")
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestDataError(f"manifest audit is invalid JSON: {audit_path}") from exc
        if not isinstance(audit, dict):
            raise ManifestDataError("manifest audit must be a JSON object")
        return audit

    def describe_context(self, context: Any) -> None:
        context.set("data.num_tasks", 2)
        context.set("data.task_names", list(TASK_NAMES))
        context.set("data.modality_names", ["audio", "video", "text", "description"])
        context.set("data.manifest_schema", self.manifest_schema)
        context.set("data.manifest_version", self.manifest_version)
        context.set("data.speaker_independence_verified", False)
        context.set("data.train_rows", len(self.train_dataset))
        context.set("data.dev_rows", len(self.val_dataset))
        context.set("data.test_rows", len(self.test_dataset))


@DATAMODULES.register("wsm_manifest_datamodule")
def wsm_manifest_datamodule(
    context: Any | None = None,
    **params: Any,
) -> WSMManifestDataModule:
    return WSMManifestDataModule(**params)
