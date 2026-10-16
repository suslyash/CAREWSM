"""Canonical partial-label segment manifest builder for WSM Stage 1."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

CORPORA = ("depression", "parkinson")
SPLITS = ("train", "dev", "test")
SPLIT_PRIORITY = {"train": 0, "dev": 1, "test": 2}
REQUIRED_RAW_COLUMNS = ("video_id", "diagnosis", "segment_file")
BAD_SEGMENTS = {
    ("depression", "dev", "3s14gCKn-yA", "3s14gCKn-yA_005.mp4"),
    ("depression", "dev", "49LbNLMWZwM", "49LbNLMWZwM_001.mp4"),
    ("depression", "dev", "4QIa1kSG45A", "4QIa1kSG45A_001.mp4"),
}
MANIFEST_COLUMNS = (
    "segment_id",
    "video_id",
    "speaker_id",
    "corpus",
    "split",
    "y_depression",
    "y_parkinson",
    "observed_depression",
    "observed_parkinson",
    "audio_available",
    "video_available",
    "text_available",
    "description_available",
)


class ManifestError(ValueError):
    """Raised when source metadata violates the canonical manifest contract."""


def _source_path(data_root: Path, corpus: str, source_split: str) -> Path:
    suffix = "_mishas" if source_split == "test" else ""
    return data_root / corpus / f"{source_split}_labels_segments_min_filtered{suffix}.csv"


def _read_source(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ManifestError(f"required source file is missing: {path}")
    delimiter = ";" if path.stem.endswith("_mishas") else ","
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ManifestError(f"source has no header: {path}")
        columns = [str(column).strip() for column in reader.fieldnames]
        if len(columns) != len(set(columns)):
            raise ManifestError(f"source has duplicate columns: {path}")
        missing = [column for column in REQUIRED_RAW_COLUMNS if column not in columns]
        if missing:
            raise ManifestError(f"{path} is missing required columns: {missing}")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            row = {str(key).strip(): (str(value).strip() if value is not None else "") for key, value in raw.items()}
            if any(not row.get(column) for column in REQUIRED_RAW_COLUMNS):
                raise ManifestError(f"{path}:{line_number} has an empty required field")
            try:
                diagnosis = int(row["diagnosis"])
            except ValueError as exc:
                raise ManifestError(f"{path}:{line_number} diagnosis is not an integer") from exc
            if diagnosis not in (0, 1):
                raise ManifestError(f"{path}:{line_number} diagnosis must be 0 or 1")
            rows.append(row)
    return columns, rows


def _load_records(data_root: Path) -> tuple[list[dict[str, Any]], OrderedDict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    source_audit: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for corpus in CORPORA:
        for source_split in SPLITS:
            path = _source_path(data_root, corpus, source_split)
            columns, rows = _read_source(path)
            retained = []
            for row in rows:
                key = (corpus, source_split, row["video_id"], row["segment_file"])
                if key in BAD_SEGMENTS:
                    continue
                item = {
                    "corpus": corpus,
                    "source_split": source_split,
                    "video_id": row["video_id"],
                    "segment_file": row["segment_file"],
                    "diagnosis": int(row["diagnosis"]),
                }
                retained.append(item)
                records.append(item)
            source_audit[f"{corpus}/{source_split}"] = {
                "source_file": str(path),
                "delimiter": ";" if source_split == "test" else ",",
                "raw_columns": columns,
                "raw_row_count": len(rows),
                "excluded_by_existing_bad_segment_rules": len(rows) - len(retained),
                "post_existing_filter_row_count": len(retained),
                "post_existing_filter_unique_video_id_count": len({item["video_id"] for item in retained}),
            }

    assigned_split: dict[str, str] = {}
    for item in records:
        previous = assigned_split.get(item["video_id"])
        if previous is None or SPLIT_PRIORITY[item["source_split"]] > SPLIT_PRIORITY[previous]:
            assigned_split[item["video_id"]] = item["source_split"]

    retained_records = [
        item for item in records if assigned_split[item["video_id"]] == item["source_split"]
    ]
    for item in retained_records:
        item["split"] = assigned_split[item["video_id"]]
    retained_records.sort(
        key=lambda item: (
            SPLIT_PRIORITY[item["split"]],
            CORPORA.index(item["corpus"]),
            item["video_id"],
            item["segment_file"],
        )
    )
    return retained_records, source_audit


def _segment_id(corpus: str, video_id: str, segment_file: str) -> str:
    return json.dumps([corpus, video_id, segment_file], ensure_ascii=True, separators=(",", ":"))


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _availability(item: dict[str, Any], data_root: Path) -> tuple[bool, bool, bool]:
    folder = data_root / item["corpus"] / f"{item['source_split']}_labels" / item["video_id"]
    segment_file = Path(item["segment_file"])
    audio = folder / "segments" / f"{segment_file.stem}.wav"
    video = folder / "segments" / item["segment_file"]
    transcript = folder / f"{item['video_id']}.txt"
    return _nonempty_file(audio), _nonempty_file(video), _nonempty_file(transcript)


def _build_rows(records: list[dict[str, Any]], data_root: Path) -> list[dict[str, Any]]:
    rows = []
    for item in records:
        audio_available, video_available, _ = _availability(item, data_root)
        is_depression = item["corpus"] == "depression"
        rows.append({
            "segment_id": _segment_id(item["corpus"], item["video_id"], item["segment_file"]),
            "video_id": item["video_id"],
            "speaker_id": None,
            "corpus": item["corpus"],
            "split": item["split"],
            "y_depression": item["diagnosis"] if is_depression else None,
            "y_parkinson": None if is_depression else item["diagnosis"],
            "observed_depression": is_depression,
            "observed_parkinson": not is_depression,
            "audio_available": audio_available,
            "video_available": video_available,
            "text_available": False,
            "description_available": False,
        })
    ids = [row["segment_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ManifestError("canonical segment_id is not unique")
    if any(row["speaker_id"] is not None for row in rows):
        raise ManifestError("speaker_id must remain null in this task")
    return rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: "" if row[column] is None else row[column] for column in MANIFEST_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def _overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {
        split: {row["video_id"] for row in rows if row["split"] == split}
        for split in SPLITS
    }
    pairwise = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        values = sorted(by_split[left] & by_split[right])
        pairwise[f"{left}_vs_{right}"] = {"count": len(values), "values": values}
    return {
        "pairwise": pairwise,
        "any_video_overlap": any(item["count"] for item in pairwise.values()),
    }


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for corpus in CORPORA:
        for split in SPLITS:
            subset = [row for row in rows if row["corpus"] == corpus and row["split"] == split]
            values = {}
            for disease, field in (("depression", "y_depression"), ("parkinson", "y_parkinson")):
                known = [row[field] for row in subset if row[field] is not None]
                values[disease] = {
                    "positive": sum(value == 1 for value in known),
                    "negative": sum(value == 0 for value in known),
                    "unknown": sum(row[field] is None for row in subset),
                }
            result[f"{corpus}/{split}"] = {"rows": len(subset), "diseases": values}
    return result


def _availability_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        field: {
            "true": sum(bool(row[field]) for row in rows),
            "false": sum(not bool(row[field]) for row in rows),
        }
        for field in ("audio_available", "video_available", "text_available", "description_available")
    }


def build_manifest(data_root: str | Path = "/media/maxim/Databases/WSM_NEW") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"data root is not a directory: {root}")
    records, source_audit = _load_records(root)
    rows = _build_rows(records, root)
    serialized = _csv_bytes(rows)
    transcript_coverage = sum(
        _availability(item, root)[2] for item in records
    )
    counts = _counts(rows)
    split_rows = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    audit = {
        "schema": "wsm_canonical_partial_label_segment_manifest",
        "version": "stage1_manifest_v1",
        "canonical_columns": list(MANIFEST_COLUMNS),
        "data_root": str(root),
        "source_audit": source_audit,
        "row_counts": {"total": len(rows), "by_split": split_rows, "by_corpus_split": counts},
        "speaker_id": {
            "null_count": sum(row["speaker_id"] is None for row in rows),
            "speaker_independence_verified": False,
            "reason": "TASK-001A found no authoritative speaker identifier; video_id was not used as fallback",
        },
        "speaker_independence_verified": False,
        "segment_identity": {
            "rule": "JSON array encoding of [corpus, video_id, segment_file]",
            "row_count": len(rows),
            "unique_count": len({row["segment_id"] for row in rows}),
            "collision_count": 0,
        },
        "split_overlap": _overlap(rows),
        "availability": _availability_counts(rows),
        "text_source_only_evidence": {
            "video_level_transcript_coverage": {"available": transcript_coverage, "total": len(rows)},
            "segment_text_alignment_established": False,
        },
        "manifest_fingerprint": {
            "algorithm": "sha256",
            "value": hashlib.sha256(serialized).hexdigest(),
            "input": "canonical CSV serialization with UTF-8, fixed column order, LF line endings",
        },
        "pseudo_labels_created": False,
        "test_usage": {
            "model_predictions_inspected": False,
            "performance_metrics_inspected": False,
            "selection_or_tuning_performed": False,
        },
        "writes": {
            "dataset_root_modified": False,
            "production_manifest_created": True,
            "datamodule_created": False,
        },
    }
    return rows, audit


def write_manifest(rows: list[dict[str, Any]], output: str | Path) -> None:
    Path(output).write_bytes(_csv_bytes(rows))


def write_audit(audit: dict[str, Any], output: str | Path) -> None:
    Path(output).write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
