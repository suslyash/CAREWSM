"""Authoritative speaker-map validation and split leakage audit."""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from common.data.wsm_manifest import SPLITS


REQUIRED_SPEAKER_MAP_COLUMNS = [
    "corpus",
    "video_id",
    "speaker_id",
    "source_reference",
]
TASK1A_UNRESOLVED_REASON = (
    "TASK-001A found no authoritative speaker identifier in the audited raw segment CSVs "
    "or adjacent JSON metadata; no video_id fallback or identity inference is permitted."
)


class SpeakerMapError(ValueError):
    """Raised when an authoritative speaker map violates its strict contract."""


def load_speaker_map(path: str | Path) -> dict[tuple[str, str], str]:
    map_path = Path(path).expanduser().resolve()
    if not map_path.is_file():
        raise SpeakerMapError(f"speaker map is missing: {map_path}")
    mapping: dict[tuple[str, str], str] = {}
    with map_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or list(reader.fieldnames) != REQUIRED_SPEAKER_MAP_COLUMNS:
            raise SpeakerMapError(
                "speaker map columns must be exactly "
                f"{REQUIRED_SPEAKER_MAP_COLUMNS}"
            )
        for line_number, raw in enumerate(reader, start=2):
            row = {str(key).strip(): (str(value).strip() if value is not None else "") for key, value in raw.items()}
            if any(not row[column] for column in REQUIRED_SPEAKER_MAP_COLUMNS):
                raise SpeakerMapError(f"speaker map row {line_number} has an empty required field")
            if row["corpus"] not in {"depression", "parkinson"}:
                raise SpeakerMapError(f"speaker map row {line_number} has invalid corpus")
            key = (row["corpus"], row["video_id"])
            if key in mapping:
                raise SpeakerMapError(
                    f"speaker map row {line_number} duplicates or conflicts with pair {key}"
                )
            mapping[key] = row["speaker_id"]
    return mapping


def _video_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {
        split: {row["video_id"] for row in rows if row["split"] == split}
        for split in SPLITS
    }
    return _pairwise_overlap(by_split, "video")


def _pairwise_overlap(by_split: dict[str, set[str]], kind: str) -> dict[str, Any]:
    pairwise = OrderedDict()
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        values = sorted(by_split[left] & by_split[right])
        pairwise[f"{left}_vs_{right}"] = {
            "count": len(values),
            "values": values,
        }
    return {
        "kind": kind,
        "by_split": {split: sorted(values) for split, values in by_split.items()},
        "pairwise": pairwise,
        "any_overlap": any(item["count"] > 0 for item in pairwise.values()),
    }


def _canonical_pairs(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(row["corpus"], row["video_id"]) for row in rows}


def audit_speaker_map(
    rows: list[dict[str, Any]],
    speaker_map_path: str | Path | None = None,
    *,
    manifest_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_pairs = _canonical_pairs(rows)
    canonical_manifest_fingerprint = (
        (manifest_audit or {}).get("manifest_fingerprint", {})
    )
    video_overlap = _video_overlap(rows)
    base = {
        "schema": "wsm_authoritative_speaker_split_gate",
        "version": "stage1_speaker_map_v1",
        "canonical_manifest_fingerprint": canonical_manifest_fingerprint,
        "canonical_row_count": len(rows),
        "canonical_unique_video_pair_count": len(canonical_pairs),
        "required_speaker_map_columns": list(REQUIRED_SPEAKER_MAP_COLUMNS),
        "map_path": str(Path(speaker_map_path).expanduser().resolve()) if speaker_map_path else None,
        "video_split_overlap": video_overlap,
        "video_independence_verified": not video_overlap["any_overlap"],
        "coverage": {
            "canonical_pairs": len(canonical_pairs),
            "mapped_pairs": 0,
            "unmapped_pairs": len(canonical_pairs),
            "extra_pairs": 0,
            "unmapped_pair_values": [],
            "extra_pair_values": [],
        },
        "conflicts": {"count": 0, "details": []},
        "speaker_split_overlap": None,
        "speaker_independence_verified": False,
        "stage1_split_gate_passed": False,
        "test_usage": {
            "model_predictions_inspected": False,
            "performance_metrics_inspected": False,
            "selection_or_tuning_performed": False,
        },
    }
    if speaker_map_path is None:
        base.update({
            "status": "unresolved",
            "reason": TASK1A_UNRESOLVED_REASON,
            "next_required_evidence": "authoritative speaker map",
        })
        return base

    mapping = load_speaker_map(speaker_map_path)
    mapped_pairs = canonical_pairs & set(mapping)
    unmapped = sorted(canonical_pairs - set(mapping))
    extra = sorted(set(mapping) - canonical_pairs)
    speaker_by_split = {split: set() for split in SPLITS}
    conflict_details = []
    for row in rows:
        key = (row["corpus"], row["video_id"])
        speaker = mapping.get(key)
        if speaker is not None:
            speaker_by_split[row["split"]].add(speaker)
    # Each map key is unique by construction; this second pass explicitly checks
    # that every canonical row resolves consistently.
    seen_row_speakers: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["corpus"], row["video_id"])
        speaker = mapping.get(key)
        if speaker is None:
            continue
        previous = seen_row_speakers.get(key)
        if previous is not None and previous != speaker:
            conflict_details.append({
                "pair": list(key),
                "speaker_ids": sorted({previous, speaker}),
            })
        seen_row_speakers[key] = speaker
    speaker_overlap = _pairwise_overlap(speaker_by_split, "speaker")
    complete = not unmapped and not conflict_details
    verified = complete and not speaker_overlap["any_overlap"] and not video_overlap["any_overlap"]
    base.update({
        "status": "verified" if verified else "failed",
        "reason": (
            "complete authoritative map and non-leaking speaker/video splits verified"
            if verified
            else "authoritative map supplied but coverage, conflicts, or split leakage failed"
        ),
        "next_required_evidence": None if verified else "corrected authoritative speaker map",
        "coverage": {
            "canonical_pairs": len(canonical_pairs),
            "mapped_pairs": len(mapped_pairs),
            "unmapped_pairs": len(unmapped),
            "extra_pairs": len(extra),
            "unmapped_pair_values": [list(pair) for pair in unmapped],
            "extra_pair_values": [list(pair) for pair in extra],
        },
        "conflicts": {"count": len(conflict_details), "details": conflict_details},
        "speaker_split_overlap": speaker_overlap,
        "speaker_independence_verified": verified,
        "stage1_split_gate_passed": bool(video_overlap["any_overlap"] is False and verified),
    })
    return base


def build_unresolved_report(
    rows: list[dict[str, Any]],
    *,
    manifest_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return audit_speaker_map(rows, None, manifest_audit=manifest_audit)
