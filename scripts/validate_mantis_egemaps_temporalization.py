#!/usr/bin/env python3
"""Fail-closed, read-only validation of the eGeMAPS LLD time contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WINDOW_SECONDS = 1.0
WINDOW_HOP_SECONDS = 0.5
TIMESTAMP_TOLERANCE = 1e-6
SAMPLE_RATE = 16000
MAX_TERMINAL_EXCESS = 1 / SAMPLE_RATE + TIMESTAMP_TOLERANCE
QUALITY_THRESHOLDS = (0.80, 0.95)
CACHE_FIELDS = (
    "embeddings",
    "window_start_sec",
    "window_end_sec",
    "validity_mask",
    "valid_observation_fraction",
    "frame_exclusion_mask",
    "frame_exclusion_reason",
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


def read_first_rows(path: Path, count: int = 3) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [row for _, row in zip(range(count), reader)]


ARTIFACT_CHANNEL_ORDER = (
    "Loudness_sma3",
    "alphaRatio_sma3",
    "hammarbergIndex_sma3",
    "slope0-500_sma3",
    "slope500-1500_sma3",
    "spectralFlux_sma3",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz",
    "F2bandwidth_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz",
    "F3bandwidth_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
)


def registry_channels(registry_path: Path) -> list[str]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered = [
        channel
        for names in registry["sources"]["egemaps_lld"]["families"].values()
        for channel in names
    ]
    if len(registered) != 25 or set(registered) != set(ARTIFACT_CHANNEL_ORDER):
        raise ValueError("eGeMAPS registry channel set does not match the artifact contract")
    return list(ARTIFACT_CHANNEL_ORDER)


def artifact_path(root: Path, source: dict[str, str]) -> Path:
    profile = source["profile_id"]
    video = source["video_id"]
    return (
        root
        / "extracted"
        / source["label_id"]
        / source["subgroup"]
        / (profile if profile.startswith("PROFILE_") else f"PROFILE_{profile}")
        / (video if video.startswith("VIDEO_") else f"VIDEO_{video}")
        / "egemaps_lld.csv"
    )


def finite_float(value: str | None) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def interval_union_duration(
    intervals: list[tuple[float, float]], start: float, end: float
) -> float:
    clipped = sorted(
        (max(frame_start, start), min(frame_end, end))
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
    turn_start: float,
    turn_end: float,
    intervals: list[tuple[float, float]],
    frame_starts: list[float],
    nominal_hop: float,
) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    index = 0
    expected_frames = round(WINDOW_SECONDS / nominal_hop)
    while turn_start + index * WINDOW_HOP_SECONDS + WINDOW_SECONDS <= (
        turn_end + TIMESTAMP_TOLERANCE
    ):
        start = turn_start + index * WINDOW_HOP_SECONDS
        end = start + WINDOW_SECONDS
        observed_frames = sum(
            start - TIMESTAMP_TOLERANCE <= frame_start < end - TIMESTAMP_TOLERANCE
            for frame_start in frame_starts
        )
        windows.append(
            {
                "valid_observation_fraction": min(
                    1.0,
                    interval_union_duration(intervals, start, end)
                    / WINDOW_SECONDS,
                ),
                "expected_frame_density": observed_frames / expected_frames,
            }
        )
        index += 1
    return windows


def empty_diagnostics(status: str = "VALID") -> dict[str, Any]:
    return {
        "status": status,
        "rows": 0,
        "original_rows": 0,
        "retained_rows": 0,
        "excluded_frame_count": 0,
        "excluded_terminal_frame_count": 0,
        "exclusion_reasons": Counter(),
        "turns": 0,
        "invalid_timestamp_rows": 0,
        "nonfinite_feature_rows": 0,
        "duplicate_timestamp_rows": 0,
        "out_of_order_rows": 0,
        "out_of_turn_rows": 0,
        "original_out_of_turn_rows": 0,
        "inconsistent_turn_boundaries": 0,
        "interval_gap_count": 0,
        "interval_overlap_count": 0,
        "missing_expected_frame_starts": 0,
        "irregular_cadence_count": 0,
        "policy_rejected_boundary_count": 0,
        "frame_duration_counts": Counter(),
        "frame_hop_counts": Counter(),
        "candidate_windows": 0,
        "valid_windows_80": 0,
        "valid_windows_95": 0,
        "candidate_windows_before_exclusion": 0,
        "valid_windows_80_before_exclusion": 0,
        "valid_windows_95_before_exclusion": 0,
        "density_sum": 0.0,
        "density_min": None,
        "density_max": None,
        "density_count": 0,
        "density_sum_before_exclusion": 0.0,
        "density_min_before_exclusion": None,
        "density_max_before_exclusion": None,
        "density_count_before_exclusion": 0,
    }


def inspect_artifact(
    path: Path,
    channels: list[str],
    nominal_hop: float,
    corpus: str = "CARE",
    boundary_policy: str = "strict",
) -> dict[str, Any]:
    result = empty_diagnostics()
    try:
        rows = read_csv(path)
    except (OSError, UnicodeError):
        return empty_diagnostics("UNREADABLE")
    result["rows"] = len(rows)
    result["original_rows"] = len(rows)
    if not rows:
        return empty_diagnostics("EMPTY")

    expected_columns = list(REQUIRED_METADATA) + channels
    if list(rows[0]) != expected_columns:
        return empty_diagnostics("SCHEMA_MISMATCH")

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row_order, row in enumerate(rows):
        parsed: dict[str, float] = {"row_order": float(row_order)}
        timestamp_valid = True
        for name in REQUIRED_METADATA[1:]:
            value = finite_float(row.get(name))
            if value is None:
                timestamp_valid = False
                break
            parsed[name] = value
        feature_valid = all(finite_float(row.get(name)) is not None for name in channels)
        if not timestamp_valid:
            result["invalid_timestamp_rows"] += 1
        if not feature_valid:
            result["nonfinite_feature_rows"] += 1
        if not timestamp_valid or not feature_valid:
            continue
        if parsed["frame_end_sec"] <= parsed["frame_start_sec"]:
            result["invalid_timestamp_rows"] += 1
            continue
        if parsed["turn_end_sec"] <= parsed["turn_start_sec"]:
            result["invalid_timestamp_rows"] += 1
            continue
        grouped[row["turn_id"]].append(parsed)

    result["turns"] = len(grouped)
    exclusions: set[float] = set()
    for frames in grouped.values():
        frames_sorted = sorted(frames, key=lambda item: item["frame_start_sec"])
        turn_starts = {frame["turn_start_sec"] for frame in frames}
        turn_ends = {frame["turn_end_sec"] for frame in frames}
        inconsistent = (
            max(turn_starts) - min(turn_starts) > TIMESTAMP_TOLERANCE
            or max(turn_ends) - min(turn_ends) > TIMESTAMP_TOLERANCE
        )
        if inconsistent:
            result["inconsistent_turn_boundaries"] += 1
        turn_start = frames_sorted[0]["turn_start_sec"]
        turn_end = frames_sorted[0]["turn_end_sec"]

        for previous, current in zip(frames, frames[1:]):
            if current["frame_start_sec"] < previous["frame_start_sec"] - TIMESTAMP_TOLERANCE:
                result["out_of_order_rows"] += 1

        for frame in frames_sorted:
            duration = frame["frame_end_sec"] - frame["frame_start_sec"]
            result["frame_duration_counts"][f"{duration:.6f}"] += 1
        for previous, current in zip(frames_sorted, frames_sorted[1:]):
            duration = previous["frame_end_sec"] - previous["frame_start_sec"]
            delta = current["frame_start_sec"] - previous["frame_start_sec"]
            result["frame_hop_counts"][f"{delta:.6f}"] += 1
            if delta <= TIMESTAMP_TOLERANCE:
                result["duplicate_timestamp_rows"] += 1
                continue
            expected_steps = round(delta / nominal_hop)
            residual = abs(delta - expected_steps * nominal_hop)
            if expected_steps < 1 or residual > TIMESTAMP_TOLERANCE:
                result["irregular_cadence_count"] += 1
            elif expected_steps > 1:
                result["missing_expected_frame_starts"] += expected_steps - 1
            if delta > duration + TIMESTAMP_TOLERANCE:
                result["interval_gap_count"] += 1
            if (
                current["frame_start_sec"]
                < previous["frame_end_sec"] - TIMESTAMP_TOLERANCE
            ):
                result["interval_overlap_count"] += 1

        boundary_indices = [
            index
            for index, frame in enumerate(frames_sorted)
            if (
                frame["frame_start_sec"] < turn_start - TIMESTAMP_TOLERANCE
                or frame["frame_end_sec"] > turn_end + TIMESTAMP_TOLERANCE
            )
        ]
        result["original_out_of_turn_rows"] += len(boundary_indices)
        allowed_indices: list[int] = []
        if boundary_indices:
            if (
                boundary_policy == "terminal-exclude"
                and corpus == "CARE"
                and not inconsistent
                and len(boundary_indices) == 1
                and boundary_indices[0] == len(frames_sorted) - 1
            ):
                frame = frames_sorted[boundary_indices[0]]
                excess = frame["frame_end_sec"] - turn_end
                start_inside = (
                    turn_start - TIMESTAMP_TOLERANCE
                    <= frame["frame_start_sec"]
                    <= turn_end + TIMESTAMP_TOLERANCE
                )
                if (
                    frame["frame_end_sec"] > turn_end + TIMESTAMP_TOLERANCE
                    and start_inside
                    and excess <= MAX_TERMINAL_EXCESS
                ):
                    allowed_indices = boundary_indices
            if not allowed_indices:
                result["policy_rejected_boundary_count"] += len(boundary_indices)

        exclusions.update(
            frames_sorted[index]["row_order"]
            for index in allowed_indices
        )
        retained = [
            frame
            for index, frame in enumerate(frames_sorted)
            if index not in allowed_indices
        ]
        result["out_of_turn_rows"] += len(boundary_indices) - len(allowed_indices)

        original_windows = candidate_windows(
            turn_start,
            turn_end,
            [
                (frame["frame_start_sec"], frame["frame_end_sec"])
                for frame in frames_sorted
            ],
            [frame["frame_start_sec"] for frame in frames_sorted],
            nominal_hop,
        )
        windows = candidate_windows(
            turn_start,
            turn_end,
            [
                (frame["frame_start_sec"], frame["frame_end_sec"])
                for frame in retained
            ],
            [frame["frame_start_sec"] for frame in retained],
            nominal_hop,
        )
        result["candidate_windows_before_exclusion"] += len(original_windows)
        result["valid_windows_80_before_exclusion"] += sum(
            window["valid_observation_fraction"] >= QUALITY_THRESHOLDS[0]
            for window in original_windows
        )
        result["valid_windows_95_before_exclusion"] += sum(
            window["valid_observation_fraction"] >= QUALITY_THRESHOLDS[1]
            for window in original_windows
        )
        result["candidate_windows"] += len(windows)
        result["valid_windows_80"] += sum(
            window["valid_observation_fraction"] >= QUALITY_THRESHOLDS[0]
            for window in windows
        )
        result["valid_windows_95"] += sum(
            window["valid_observation_fraction"] >= QUALITY_THRESHOLDS[1]
            for window in windows
        )
        for key, selected in (("_before_exclusion", original_windows), ("", windows)):
            densities = [window["expected_frame_density"] for window in selected]
            if not densities:
                continue
            if key == "":
                result["density_sum"] += sum(densities)
                result["density_count"] += len(densities)
                minimum_key = "density_min"
                maximum_key = "density_max"
            else:
                result["density_sum_before_exclusion"] += sum(densities)
                result["density_count_before_exclusion"] += len(densities)
                minimum_key = "density_min_before_exclusion"
                maximum_key = "density_max_before_exclusion"
            result[minimum_key] = (
                min(densities)
                if result[minimum_key] is None
                else min(result[minimum_key], min(densities))
            )
            result[maximum_key] = (
                max(densities)
                if result[maximum_key] is None
                else max(result[maximum_key], max(densities))
            )

    invalid_fields = (
        "invalid_timestamp_rows",
        "nonfinite_feature_rows",
        "duplicate_timestamp_rows",
        "out_of_order_rows",
        "out_of_turn_rows",
        "inconsistent_turn_boundaries",
        "missing_expected_frame_starts",
        "irregular_cadence_count",
    )
    if any(result[field] for field in invalid_fields):
        if exclusions:
            result["out_of_turn_rows"] = result["original_out_of_turn_rows"]
            exclusions.clear()
        result["status"] = "INVALID_TEMPORAL_DATA"
    result["excluded_frame_count"] = len(exclusions)
    result["excluded_terminal_frame_count"] = len(exclusions)
    if exclusions:
        result["exclusion_reasons"][
            "excluded_terminal_out_of_turn_1sample"
        ] = len(exclusions)
    result["retained_rows"] = sum(
        len(frames) for frames in grouped.values()
    ) - len(exclusions)
    return result


def derive_nominal_hop(paths: set[Path]) -> tuple[float | None, Counter[str]]:
    hops: Counter[str] = Counter()
    for path in paths:
        try:
            rows = read_first_rows(path)
        except (OSError, UnicodeError):
            continue
        for previous, current in zip(rows, rows[1:]):
            start_a = finite_float(previous.get("frame_start_sec"))
            start_b = finite_float(current.get("frame_start_sec"))
            if start_a is not None and start_b is not None and start_b > start_a:
                hops[f"{start_b - start_a:.6f}"] += 1
    if not hops:
        return None, hops
    nominal_key, nominal_count = hops.most_common(1)[0]
    if nominal_count / sum(hops.values()) < 0.99:
        return None, hops
    return float(nominal_key), hops


def merge_counter(target: Counter[str], source: Counter[str]) -> None:
    target.update(source)


def density_stats(total: float, count: int, minimum: float | None, maximum: float | None) -> dict[str, float | None]:
    return {
        "mean": total / count if count else None,
        "min": minimum,
        "max": maximum,
    }


def fixture_row(
    turn_id: str,
    turn_start: str,
    turn_end: str,
    frame_start: str,
    frame_end: str,
    channels: list[str],
) -> dict[str, str]:
    row = {
        "turn_id": turn_id,
        "turn_start_sec": turn_start,
        "turn_end_sec": turn_end,
        "frame_start_sec": frame_start,
        "frame_end_sec": frame_end,
    }
    row.update({channel: "0.0" for channel in channels})
    return row


def write_fixture(path: Path, rows: list[dict[str, str]], channels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_METADATA) + channels)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_checks(channels: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="mantis-egemaps-") as directory:
        root = Path(directory)
        good_rows = [
            fixture_row("1", "0", "1.01", f"{index / 100:.2f}", f"{index / 100 + 0.02:.2f}", channels)
            for index in range(100)
        ]
        good_path = root / "good.csv"
        write_fixture(good_path, good_rows, channels)
        good = inspect_artifact(good_path, channels, 0.01)
        assert good["status"] == "VALID"
        assert good["candidate_windows"] == 1
        assert good["valid_windows_80"] == 1
        assert good["missing_expected_frame_starts"] == 0
        assert good["density_min"] == 1.0

        permitted = list(good_rows)
        permitted[-1] = dict(permitted[-1], frame_end_sec="1.0100625")
        permitted_path = root / "permitted_terminal.csv"
        write_fixture(permitted_path, permitted, channels)
        assert inspect_artifact(
            permitted_path, channels, 0.01, "CARE", "strict"
        )["status"] == "INVALID_TEMPORAL_DATA"
        policy_result = inspect_artifact(
            permitted_path, channels, 0.01, "CARE", "terminal-exclude"
        )
        assert policy_result["status"] == "VALID"
        assert policy_result["excluded_frame_count"] == 1
        assert policy_result["original_out_of_turn_rows"] == 1
        assert policy_result["out_of_turn_rows"] == 0
        assert policy_result["retained_rows"] == 99
        assert policy_result["exclusion_reasons"][
            "excluded_terminal_out_of_turn_1sample"
        ] == 1
        assert policy_result["density_min"] == 0.99

        large_excess = list(good_rows)
        large_excess[-1] = dict(large_excess[-1], frame_end_sec="1.0101")
        large_path = root / "large_excess.csv"
        write_fixture(large_path, large_excess, channels)
        assert inspect_artifact(
            large_path, channels, 0.01, "CARE", "terminal-exclude"
        )["status"] == "INVALID_TEMPORAL_DATA"

        nonterminal = list(good_rows)
        nonterminal[50] = dict(nonterminal[50], frame_end_sec="1.02")
        nonterminal_path = root / "nonterminal.csv"
        write_fixture(nonterminal_path, nonterminal, channels)
        assert inspect_artifact(
            nonterminal_path, channels, 0.01, "CARE", "terminal-exclude"
        )["status"] == "INVALID_TEMPORAL_DATA"

        multiple = list(good_rows)
        multiple[-2] = dict(multiple[-2], frame_end_sec="1.0100625")
        multiple[-1] = dict(multiple[-1], frame_end_sec="1.0100625")
        multiple_path = root / "multiple_boundary.csv"
        write_fixture(multiple_path, multiple, channels)
        assert inspect_artifact(
            multiple_path, channels, 0.01, "CARE", "terminal-exclude"
        )["status"] == "INVALID_TEMPORAL_DATA"

        def check_invalid(name: str, rows: list[dict[str, str]], expected: str) -> None:
            path = root / f"{name}.csv"
            write_fixture(path, rows, channels)
            assert inspect_artifact(path, channels, 0.01)["status"] == expected
            assert inspect_artifact(
                path, channels, 0.01, "CARE", "terminal-exclude"
            )["status"] == expected

        missing = good_rows[:50] + good_rows[51:]
        check_invalid("missing_frame", missing, "INVALID_TEMPORAL_DATA")
        duplicate = good_rows[:50] + [good_rows[49]]
        check_invalid("duplicate", duplicate, "INVALID_TEMPORAL_DATA")
        nonfinite = list(good_rows)
        nonfinite[3] = dict(nonfinite[3], frame_start_sec="nan")
        check_invalid("nonfinite", nonfinite, "INVALID_TEMPORAL_DATA")
        out_of_order = list(good_rows)
        out_of_order[2], out_of_order[3] = out_of_order[3], out_of_order[2]
        check_invalid("out_of_order", out_of_order, "INVALID_TEMPORAL_DATA")
        inconsistent = list(good_rows)
        inconsistent[4] = dict(inconsistent[4], turn_end_sec="1.02")
        check_invalid("inconsistent_turn", inconsistent, "INVALID_TEMPORAL_DATA")
        short = [
            fixture_row("1", "0", "0.9", "0.0", "0.02", channels),
        ]
        short_path = root / "short.csv"
        write_fixture(short_path, short, channels)
        assert inspect_artifact(short_path, channels, 0.01)["candidate_windows"] == 0
        boundary = [
            fixture_row("1", "5.0", "6.01", f"{5 + index / 100:.2f}", f"{5 + index / 100 + 0.02:.2f}", channels)
            for index in range(100)
        ]
        boundary_path = root / "boundary.csv"
        write_fixture(boundary_path, boundary, channels)
        assert inspect_artifact(boundary_path, channels, 0.01)["candidate_windows"] == 1
        wrong_schema = root / "wrong_schema.csv"
        write_fixture(wrong_schema, good_rows, list(reversed(channels)))
        assert inspect_artifact(wrong_schema, channels, 0.01)["status"] == "SCHEMA_MISMATCH"
        cache = {field: [] for field in CACHE_FIELDS}
        assert tuple(cache) == CACHE_FIELDS
        assert exit_code_for_failures([]) == 0
        assert exit_code_for_failures(["invalid fixture"]) != 0


def exit_code_for_failures(failures: list[str]) -> int:
    return 1 if failures else 0


def source_rows(root: Path, fold: int, part: str) -> list[dict[str, str]]:
    return read_csv(root / "splits" / "folds" / f"fold_{fold}_{part}.csv")


def profile_overlap(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> int:
    return len({row["profile_id"] for row in train_rows} & {row["profile_id"] for row in val_rows})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--care-root", type=Path, default=Path("/media/maxim/Databases/CARE"))
    parser.add_argument("--wsm-root", type=Path, default=Path("/media/maxim/Databases/WSM_NEW"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/feature_audit/sample_feature_availability.csv"))
    parser.add_argument("--registry", type=Path, default=Path("/media/maxim/Databases/CARE/splits/features.json"))
    parser.add_argument(
        "--boundary-policy",
        choices=("strict", "terminal-exclude"),
        default="strict",
    )
    args = parser.parse_args()

    try:
        channels = registry_channels(args.registry)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    synthetic_checks(channels)
    manifest = {
        (row["corpus"], row["sample_id"]): row
        for row in read_csv(args.manifest)
    }
    roots = {"CARE": args.care_root, "WSM": args.wsm_root}
    populations: dict[str, tuple[list[dict[str, str]], Path]] = {}
    paths_by_corpus: dict[str, set[Path]] = defaultdict(set)
    profile_overlaps: dict[str, int] = {}
    for corpus, root in roots.items():
        for fold in range(5):
            train_rows = source_rows(root, fold, "train")
            val_rows = source_rows(root, fold, "val")
            profile_overlaps[f"{corpus}/fold_{fold}"] = profile_overlap(train_rows, val_rows)
            populations[f"{corpus}/fold_{fold}/train"] = (train_rows, root)
            populations[f"{corpus}/fold_{fold}/val"] = (val_rows, root)
            for row in train_rows + val_rows:
                if manifest.get((corpus, row["sample_id"]), {}).get("egemaps_lld_status") == "VALID":
                    paths_by_corpus[corpus].add(artifact_path(root, row))

    nominal_hops: dict[str, float | None] = {}
    hop_evidence: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for corpus, paths in paths_by_corpus.items():
        nominal, evidence = derive_nominal_hop(paths)
        nominal_hops[corpus] = nominal
        hop_evidence[corpus] = dict(evidence)
        if nominal is None:
            failures.append(f"{corpus}: ambiguous nominal frame-start hop")

    artifact_cache: dict[tuple[Path, float, str, str], dict[str, Any]] = {}
    population_summary: dict[str, dict[str, Any]] = {}
    duration_distributions: dict[str, Counter[str]] = defaultdict(Counter)
    hop_distributions: dict[str, Counter[str]] = defaultdict(Counter)
    for population, (rows, root) in populations.items():
        corpus = population.split("/", 1)[0]
        nominal_hop = nominal_hops[corpus]
        diagnostics: list[dict[str, Any]] = []
        manifest_valid = 0
        for row in rows:
            if manifest.get((corpus, row["sample_id"]), {}).get("egemaps_lld_status") != "VALID":
                diagnostics.append(empty_diagnostics("MANIFEST_INVALID"))
                continue
            manifest_valid += 1
            path = artifact_path(root, row)
            if nominal_hop is None:
                diagnostics.append(empty_diagnostics("AMBIGUOUS_TIME_COORDINATES"))
                continue
            key = (path, nominal_hop, corpus, args.boundary_policy)
            if key not in artifact_cache:
                artifact_cache[key] = inspect_artifact(
                    path,
                    channels,
                    nominal_hop,
                    corpus,
                    args.boundary_policy,
                )
            diagnostics.append(artifact_cache[key])

        for diagnostic in diagnostics:
            merge_counter(duration_distributions[corpus], diagnostic["frame_duration_counts"])
            merge_counter(hop_distributions[corpus], diagnostic["frame_hop_counts"])
        fully_valid = [
            diagnostic for diagnostic in diagnostics if diagnostic["status"] == "VALID"
        ]
        audited = [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic["status"]
            not in {
                "MANIFEST_INVALID",
                "UNREADABLE",
                "EMPTY",
                "SCHEMA_MISMATCH",
                "AMBIGUOUS_TIME_COORDINATES",
            }
        ]
        readable = [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic["status"] not in {"MANIFEST_INVALID", "UNREADABLE"}
        ]
        invalid = [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic["status"] not in {"VALID", "MANIFEST_INVALID"}
        ]
        if invalid:
            failures.append(f"{population}: {len(invalid)} invalid artifacts")
        candidate = sum(diagnostic["candidate_windows"] for diagnostic in fully_valid)
        valid_80 = sum(diagnostic["valid_windows_80"] for diagnostic in fully_valid)
        valid_95 = sum(diagnostic["valid_windows_95"] for diagnostic in fully_valid)
        candidate_before = sum(
            diagnostic["candidate_windows_before_exclusion"]
            for diagnostic in audited
        )
        valid_80_before = sum(
            diagnostic["valid_windows_80_before_exclusion"]
            for diagnostic in audited
        )
        valid_95_before = sum(
            diagnostic["valid_windows_95_before_exclusion"]
            for diagnostic in audited
        )
        density_count = sum(diagnostic["density_count"] for diagnostic in fully_valid)
        density_sum = sum(diagnostic["density_sum"] for diagnostic in fully_valid)
        density_min = min(
            (
                diagnostic["density_min"]
                for diagnostic in fully_valid
                if diagnostic["density_min"] is not None
            ),
            default=None,
        )
        density_max = max(
            (
                diagnostic["density_max"]
                for diagnostic in fully_valid
                if diagnostic["density_max"] is not None
            ),
            default=None,
        )
        density_before_count = sum(
            diagnostic["density_count_before_exclusion"] for diagnostic in audited
        )
        density_before_sum = sum(
            diagnostic["density_sum_before_exclusion"] for diagnostic in audited
        )
        density_before_min = min(
            (
                diagnostic["density_min_before_exclusion"]
                for diagnostic in audited
                if diagnostic["density_min_before_exclusion"] is not None
            ),
            default=None,
        )
        density_before_max = max(
            (
                diagnostic["density_max_before_exclusion"]
                for diagnostic in audited
                if diagnostic["density_max_before_exclusion"] is not None
            ),
            default=None,
        )
        exclusion_reasons: Counter[str] = Counter()
        for diagnostic in diagnostics:
            exclusion_reasons.update(diagnostic["exclusion_reasons"])
        population_summary[population] = {
            "source_samples": len(rows),
            "manifest_valid_artifacts": manifest_valid,
            "readable_files": len(readable),
            "fully_valid_artifacts": len(fully_valid),
            "usable_samples_80": sum(
                diagnostic["valid_windows_80"] > 0 for diagnostic in fully_valid
            ),
            "usable_samples_95": sum(
                diagnostic["valid_windows_95"] > 0 for diagnostic in fully_valid
            ),
            "candidate_windows": candidate,
            "valid_windows_80": valid_80,
            "valid_windows_95": valid_95,
            "rejected_windows_80": candidate - valid_80,
            "rejected_windows_95": candidate - valid_95,
            "candidate_windows_before_exclusion": candidate_before,
            "valid_windows_80_before_exclusion": valid_80_before,
            "valid_windows_95_before_exclusion": valid_95_before,
            "expected_frame_density": density_stats(
                density_sum, density_count, density_min, density_max
            ),
            "expected_frame_density_before_exclusion": density_stats(
                density_before_sum,
                density_before_count,
                density_before_min,
                density_before_max,
            ),
            "excluded_frame_count": sum(
                diagnostic["excluded_frame_count"] for diagnostic in diagnostics
            ),
            "excluded_terminal_frame_count": sum(
                diagnostic["excluded_terminal_frame_count"]
                for diagnostic in diagnostics
            ),
            "original_frame_rows": sum(
                diagnostic["original_rows"] for diagnostic in diagnostics
            ),
            "retained_frame_rows": sum(
                diagnostic["retained_rows"] for diagnostic in diagnostics
            ),
            "exclusion_reasons": dict(exclusion_reasons),
            "invalid_timestamp_rows": sum(
                diagnostic["invalid_timestamp_rows"] for diagnostic in diagnostics
            ),
            "nonfinite_feature_rows": sum(
                diagnostic["nonfinite_feature_rows"] for diagnostic in diagnostics
            ),
            "duplicate_timestamp_rows": sum(
                diagnostic["duplicate_timestamp_rows"] for diagnostic in diagnostics
            ),
            "out_of_order_rows": sum(
                diagnostic["out_of_order_rows"] for diagnostic in diagnostics
            ),
            "inconsistent_turn_boundaries": sum(
                diagnostic["inconsistent_turn_boundaries"]
                for diagnostic in diagnostics
            ),
            "original_out_of_turn_rows": sum(
                diagnostic["original_out_of_turn_rows"]
                for diagnostic in diagnostics
            ),
            "out_of_turn_rows": sum(
                diagnostic["out_of_turn_rows"] for diagnostic in diagnostics
            ),
            "missing_expected_frame_starts": sum(
                diagnostic["missing_expected_frame_starts"]
                for diagnostic in diagnostics
            ),
            "irregular_cadence_count": sum(
                diagnostic["irregular_cadence_count"] for diagnostic in diagnostics
            ),
            "policy_rejected_boundary_count": sum(
                diagnostic["policy_rejected_boundary_count"]
                for diagnostic in diagnostics
            ),
            "interval_gap_count": sum(
                diagnostic["interval_gap_count"] for diagnostic in diagnostics
            ),
            "interval_overlap_count": sum(
                diagnostic["interval_overlap_count"] for diagnostic in diagnostics
            ),
        }

    for population, overlap in profile_overlaps.items():
        if overlap:
            failures.append(f"{population}: TRAIN/VAL profile overlap={overlap}")

    summary = {
        "contract": {
            "channel_count": len(channels),
            "channels": channels,
            "window_seconds": WINDOW_SECONDS,
            "window_hop_seconds": WINDOW_HOP_SECONDS,
            "endpoint_rule": "half-open [start, end)",
            "frame_assignment": "interval overlap for observation coverage; frame starts for density",
            "quality_thresholds": QUALITY_THRESHOLDS,
            "cache_fields": CACHE_FIELDS,
            "boundary_policy": args.boundary_policy,
            "terminal_excess_limit_seconds": MAX_TERMINAL_EXCESS,
            "synthetic_checks": "PASS",
        },
        "nominal_hop_seconds": nominal_hops,
        "hop_evidence": hop_evidence,
        "frame_duration_distribution": {corpus: dict(counter) for corpus, counter in duration_distributions.items()},
        "frame_start_hop_distribution": {corpus: dict(counter) for corpus, counter in hop_distributions.items()},
        "profile_overlaps": profile_overlaps,
        "populations": population_summary,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code_for_failures(failures)


if __name__ == "__main__":
    raise SystemExit(main())
