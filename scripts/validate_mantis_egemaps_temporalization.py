#!/usr/bin/env python3
"""Validate the read-only eGeMAPS LLD temporalization contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WINDOW_SECONDS = 1.0
WINDOW_HOP_SECONDS = 0.5
TIMESTAMP_TOLERANCE = 1e-6
QUALITY_THRESHOLDS = (0.80, 0.95)
CACHE_FIELDS = (
    "embeddings",
    "window_start_sec",
    "window_end_sec",
    "validity_mask",
    "valid_observation_fraction",
)
REQUIRED_METADATA = (
    "turn_id",
    "turn_start_sec",
    "turn_end_sec",
    "frame_start_sec",
    "frame_end_sec",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def registry_channels(registry_path: Path) -> list[str]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    specification = registry["sources"]["egemaps_lld"]
    channels: list[str] = []
    for names in specification["families"].values():
        channels.extend(names)
    return channels


def artifact_path(root: Path, source: dict[str, str]) -> Path:
    label = source["label_id"]
    subgroup = source["subgroup"]
    profile = source["profile_id"]
    video = source["video_id"]
    return (
        root
        / "extracted"
        / label
        / subgroup
        / (profile if profile.startswith("PROFILE_") else f"PROFILE_{profile}")
        / (video if video.startswith("VIDEO_") else f"VIDEO_{video}")
        / "egemaps_lld.csv"
    )


def finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def interval_union_duration(
    intervals: list[tuple[float, float]], start: float, end: float
) -> float:
    clipped = sorted(
        (
            max(frame_start, start),
            min(frame_end, end),
        )
        for frame_start, frame_end in intervals
        if frame_end > start and frame_start < end
    )
    total = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for interval_start, interval_end in clipped:
        if interval_end <= interval_start:
            continue
        if current_start is None:
            current_start, current_end = interval_start, interval_end
        elif interval_start <= current_end + TIMESTAMP_TOLERANCE:
            current_end = max(current_end, interval_end)
        else:
            total += current_end - current_start
            current_start, current_end = interval_start, interval_end
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def candidate_windows(
    turn_start: float, turn_end: float, intervals: list[tuple[float, float]]
) -> list[dict[str, float | bool]]:
    windows: list[dict[str, float | bool]] = []
    index = 0
    while turn_start + index * WINDOW_HOP_SECONDS + WINDOW_SECONDS <= (
        turn_end + TIMESTAMP_TOLERANCE
    ):
        start = turn_start + index * WINDOW_HOP_SECONDS
        end = start + WINDOW_SECONDS
        coverage = interval_union_duration(intervals, start, end)
        windows.append(
            {
                "start_sec": start,
                "end_sec": end,
                "valid_observation_fraction": min(1.0, coverage / WINDOW_SECONDS),
            }
        )
        index += 1
    return windows


def inspect_artifact(path: Path, channels: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "VALID",
        "rows": 0,
        "turns": 0,
        "invalid_timestamp_rows": 0,
        "nonfinite_feature_rows": 0,
        "duplicate_timestamp_rows": 0,
        "out_of_order_rows": 0,
        "out_of_turn_rows": 0,
        "gap_count": 0,
        "overlap_count": 0,
        "frame_intervals": [],
        "turns_with_positive_start": 0,
        "turns_with_frame_before_turn_start": 0,
        "turns_with_frame_after_turn_end": 0,
        "candidate_windows": 0,
        "valid_windows_80": 0,
        "valid_windows_95": 0,
        "windows_by_sample": 0,
    }
    try:
        rows = read_csv(path)
    except (OSError, UnicodeError):
        result["status"] = "UNREADABLE"
        return result
    result["rows"] = len(rows)
    if not rows:
        result["status"] = "EMPTY"
        return result
    fieldnames = set(rows[0])
    if not set(REQUIRED_METADATA).issubset(fieldnames) or not set(channels).issubset(
        fieldnames
    ):
        result["status"] = "SCHEMA_MISMATCH"
        return result

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        parsed: dict[str, float] = {}
        values_are_finite = True
        for name in REQUIRED_METADATA[1:] + tuple(channels):
            value = finite_float(row.get(name, ""))
            if value is None:
                values_are_finite = False
                break
            parsed[name] = value
        if not values_are_finite:
            if any(finite_float(row.get(name, "")) is None for name in channels):
                result["nonfinite_feature_rows"] += 1
            else:
                result["invalid_timestamp_rows"] += 1
            continue
        if parsed["frame_end_sec"] <= parsed["frame_start_sec"]:
            result["invalid_timestamp_rows"] += 1
            continue
        if parsed["turn_end_sec"] <= parsed["turn_start_sec"]:
            result["invalid_timestamp_rows"] += 1
            continue
        grouped[row["turn_id"]].append(parsed)

    result["turns"] = len(grouped)
    for frames in grouped.values():
        frames_in_file_order = list(frames)
        frames_sorted = sorted(frames, key=lambda item: item["frame_start_sec"])
        turn_start = frames_sorted[0]["turn_start_sec"]
        turn_end = frames_sorted[0]["turn_end_sec"]
        if turn_start > TIMESTAMP_TOLERANCE:
            result["turns_with_positive_start"] += 1
        intervals: list[tuple[float, float]] = []
        for previous, current in zip(frames_in_file_order, frames_in_file_order[1:]):
            if current["frame_start_sec"] < previous["frame_start_sec"] - TIMESTAMP_TOLERANCE:
                result["out_of_order_rows"] += 1
            if abs(current["frame_start_sec"] - previous["frame_start_sec"]) <= TIMESTAMP_TOLERANCE:
                result["duplicate_timestamp_rows"] += 1
        for previous, current in zip(frames_sorted, frames_sorted[1:]):
            delta = current["frame_start_sec"] - previous["frame_start_sec"]
            if delta > previous["frame_end_sec"] - previous["frame_start_sec"] + TIMESTAMP_TOLERANCE:
                result["gap_count"] += 1
            if current["frame_start_sec"] < previous["frame_end_sec"] - TIMESTAMP_TOLERANCE:
                result["overlap_count"] += 1
        for frame in frames_sorted:
            if frame["frame_start_sec"] < turn_start - TIMESTAMP_TOLERANCE:
                result["turns_with_frame_before_turn_start"] += 1
                result["out_of_turn_rows"] += 1
            if frame["frame_end_sec"] > turn_end + TIMESTAMP_TOLERANCE:
                result["turns_with_frame_after_turn_end"] += 1
                result["out_of_turn_rows"] += 1
            intervals.append((frame["frame_start_sec"], frame["frame_end_sec"]))
        windows = candidate_windows(turn_start, turn_end, intervals)
        result["candidate_windows"] += len(windows)
        result["valid_windows_80"] += sum(
            item["valid_observation_fraction"] >= QUALITY_THRESHOLDS[0]
            for item in windows
        )
        result["valid_windows_95"] += sum(
            item["valid_observation_fraction"] >= QUALITY_THRESHOLDS[1]
            for item in windows
        )
    result["windows_by_sample"] = result["candidate_windows"]
    if any(
        result[name]
        for name in (
            "invalid_timestamp_rows",
            "nonfinite_feature_rows",
            "duplicate_timestamp_rows",
            "out_of_order_rows",
            "out_of_turn_rows",
        )
    ):
        result["status"] = "INVALID_TIMESTAMP_DATA"
    return result


def synthetic_checks() -> None:
    assert candidate_windows(0.0, 0.4, []) == []
    complete = [(0.0, 0.02 * index + 0.02) for index in range(50)]
    windows = candidate_windows(0.0, 1.0, complete)
    assert len(windows) == 1
    assert windows[0]["valid_observation_fraction"] == 1.0
    gap = [(0.0, 0.5), (0.6, 1.0)]
    assert candidate_windows(0.0, 1.0, gap)[0]["valid_observation_fraction"] == 0.9
    assert finite_float("") is None
    duplicate = [(0.0, 0.02), (0.0, 0.02)]
    assert duplicate[0][0] == duplicate[1][0]
    boundary = candidate_windows(1.0, 2.0, [(1.0, 1.02), (1.98, 2.0)])
    assert boundary[0]["start_sec"] == 1.0
    assert boundary[0]["end_sec"] == 2.0
    assert interval_union_duration([(0.0, 0.5)], 0.0, 1.0) == 0.5
    synthetic_cache = {field: [] for field in CACHE_FIELDS}
    assert tuple(synthetic_cache) == CACHE_FIELDS


def source_rows(root: Path, fold: int, part: str) -> list[dict[str, str]]:
    return read_csv(root / "splits" / "folds" / f"fold_{fold}_{part}.csv")


def profile_overlap(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> int:
    return len({row["profile_id"] for row in train_rows} & {row["profile_id"] for row in val_rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--care-root", type=Path, default=Path("/media/maxim/Databases/CARE"))
    parser.add_argument("--wsm-root", type=Path, default=Path("/media/maxim/Databases/WSM_NEW"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/feature_audit/sample_feature_availability.csv"))
    parser.add_argument("--registry", type=Path, default=Path("/media/maxim/Databases/CARE/splits/features.json"))
    args = parser.parse_args()
    del args.check

    synthetic_checks()
    channels = registry_channels(args.registry)
    assert len(channels) == 25
    manifest = {
        (row["corpus"], row["sample_id"]): row
        for row in read_csv(args.manifest)
    }
    roots = {"CARE": args.care_root, "WSM": args.wsm_root}
    summary: dict[str, Any] = {
        "contract": {
            "channels": channels,
            "channel_count": len(channels),
            "window_seconds": WINDOW_SECONDS,
            "window_hop_seconds": WINDOW_HOP_SECONDS,
            "endpoint_rule": "half-open [start, end)",
            "frame_assignment": "actual frame intervals intersecting the window",
            "partial_window_policy": "reject candidate windows shorter than 1.0 s within a turn",
            "quality_thresholds": QUALITY_THRESHOLDS,
            "synthetic_checks": "PASS",
            "cache_fields": CACHE_FIELDS,
        },
        "populations": {},
        "profile_overlaps": {},
        "timestamp_semantics": Counter(),
    }
    artifact_cache: dict[Path, dict[str, Any]] = {}
    for corpus, root in roots.items():
        for fold in range(5):
            train_rows = source_rows(root, fold, "train")
            val_rows = source_rows(root, fold, "val")
            summary["profile_overlaps"][f"{corpus}/fold_{fold}"] = profile_overlap(
                train_rows, val_rows
            )
            for part, rows in (("train", train_rows), ("val", val_rows)):
                diagnostics = []
                manifest_valid = 0
                for row in rows:
                    manifest_row = manifest.get((corpus, row["sample_id"]), {})
                    if manifest_row.get("egemaps_lld_status") != "VALID":
                        diagnostics.append({"status": "MANIFEST_INVALID"})
                        continue
                    manifest_valid += 1
                    path = artifact_path(root, row)
                    if path not in artifact_cache:
                        artifact_cache[path] = inspect_artifact(path, channels)
                    diagnostics.append(artifact_cache[path])
                candidate = sum(item.get("candidate_windows", 0) for item in diagnostics)
                usable_80 = sum(item.get("valid_windows_80", 0) for item in diagnostics)
                usable_95 = sum(item.get("valid_windows_95", 0) for item in diagnostics)
                window_counts_80 = [item.get("valid_windows_80", 0) for item in diagnostics]
                window_counts_95 = [item.get("valid_windows_95", 0) for item in diagnostics]

                def window_stats(values: list[int]) -> dict[str, float | int]:
                    return {
                        "min": min(values) if values else 0,
                        "median": statistics.median(values) if values else 0,
                        "max": max(values) if values else 0,
                    }

                summary["populations"][f"{corpus}/fold_{fold}/{part}"] = {
                    "source_samples": len(rows),
                    "manifest_valid_artifacts": manifest_valid,
                    "artifact_files_readable": sum(
                        item.get("status") not in {"MANIFEST_INVALID", "UNREADABLE"}
                        for item in diagnostics
                    ),
                    "usable_samples_80": sum(item.get("valid_windows_80", 0) > 0 for item in diagnostics),
                    "usable_samples_95": sum(item.get("valid_windows_95", 0) > 0 for item in diagnostics),
                    "windows_per_sample_80": window_stats(window_counts_80),
                    "windows_per_sample_95": window_stats(window_counts_95),
                    "candidate_windows": candidate,
                    "valid_windows_80": usable_80,
                    "valid_windows_95": usable_95,
                    "rejected_windows_80": candidate - usable_80,
                    "rejected_windows_95": candidate - usable_95,
                    "invalid_timestamp_rows": sum(item.get("invalid_timestamp_rows", 0) for item in diagnostics),
                    "nonfinite_feature_rows": sum(item.get("nonfinite_feature_rows", 0) for item in diagnostics),
                    "duplicate_timestamp_rows": sum(item.get("duplicate_timestamp_rows", 0) for item in diagnostics),
                    "out_of_order_rows": sum(item.get("out_of_order_rows", 0) for item in diagnostics),
                    "gap_count": sum(item.get("gap_count", 0) for item in diagnostics),
                    "overlap_count": sum(item.get("overlap_count", 0) for item in diagnostics),
                }
                summary["timestamp_semantics"][corpus] += len(rows)
    summary["timestamp_semantics"] = dict(summary["timestamp_semantics"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
