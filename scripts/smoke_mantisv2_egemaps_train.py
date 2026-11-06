#!/usr/bin/env python3
"""Bounded, TRAIN-only real eGeMAPS inference with the pinned MantisV2 model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_REVISION = "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
EXPECTED_MODEL_SHA256 = "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
WINDOW_SECONDS = 1.0
WINDOW_HOP_SECONDS = 0.5
NOMINAL_HOP_SECONDS = 0.01
TIMESTAMP_TOLERANCE = 1e-6
MIN_CONSECUTIVE_FRAMES = 95
MAX_ARTIFACTS = 3


def load_validator() -> Any:
    path = Path(__file__).with_name("validate_mantis_egemaps_temporalization.py")
    spec = importlib.util.spec_from_file_location("temporal_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load temporal validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite artifact value")
    return result


def artifact_frames(
    path: Path, channels: list[str], validator: Any
) -> tuple[list[dict[str, Any]], int, list[str]]:
    rows = read_csv(path)
    required = set(validator.REQUIRED_METADATA) | set(channels)
    if not rows or set(rows[0]) != required:
        raise ValueError("unexpected eGeMAPS schema")

    parsed: list[dict[str, Any]] = []
    for row in rows:
        item = {name: parse_float(row[name]) for name in validator.REQUIRED_METADATA[1:]}
        item["turn_id"] = row["turn_id"]
        item["values"] = np.asarray([parse_float(row[channel]) for channel in channels], dtype=np.float32)
        parsed.append(item)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed:
        grouped[item["turn_id"]].append(item)

    retained: list[dict[str, Any]] = []
    excluded = 0
    exclusion_reasons: list[str] = []
    for frames in grouped.values():
        ordered = sorted(frames, key=lambda item: item["frame_start_sec"])
        boundary = [
            item
            for item in ordered
            if item["frame_start_sec"] < item["turn_start_sec"] - TIMESTAMP_TOLERANCE
            or item["frame_end_sec"] > item["turn_end_sec"] + TIMESTAMP_TOLERANCE
        ]
        for item in boundary:
            excess = item["frame_end_sec"] - item["turn_end_sec"]
            permitted = (
                len(boundary) == 1
                and item is ordered[-1]
                and item["frame_start_sec"] <= item["turn_end_sec"] + TIMESTAMP_TOLERANCE
                and excess > TIMESTAMP_TOLERANCE
                and excess <= 1 / 16000 + TIMESTAMP_TOLERANCE
            )
            if permitted:
                excluded += 1
                exclusion_reasons.append("excluded_terminal_out_of_turn_1sample")
            else:
                raise ValueError("unapproved boundary frame reached temporalization")
        boundary_ids = {id(item) for item in boundary}
        retained.extend(item for item in ordered if id(item) not in boundary_ids)
    return retained, excluded, exclusion_reasons


def consecutive_runs(frames: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for frame in frames:
        if not current:
            current = [frame]
            continue
        delta = frame["frame_start_sec"] - current[-1]["frame_start_sec"]
        if abs(delta - NOMINAL_HOP_SECONDS) <= TIMESTAMP_TOLERANCE:
            current.append(frame)
        else:
            runs.append(current)
            current = [frame]
    if current:
        runs.append(current)
    return runs


def temporalize(frames: list[dict[str, Any]], channels: int) -> list[dict[str, Any]]:
    by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_turn[frame["turn_id"]].append(frame)

    windows: list[dict[str, Any]] = []
    for turn_frames in by_turn.values():
        ordered = sorted(turn_frames, key=lambda item: item["frame_start_sec"])
        turn_start = ordered[0]["turn_start_sec"]
        turn_end = ordered[0]["turn_end_sec"]
        index = 0
        while turn_start + index * WINDOW_HOP_SECONDS + WINDOW_SECONDS <= turn_end + TIMESTAMP_TOLERANCE:
            start = turn_start + index * WINDOW_HOP_SECONDS
            end = start + WINDOW_SECONDS
            complete = [
                frame
                for frame in ordered
                if frame["frame_start_sec"] >= start - TIMESTAMP_TOLERANCE
                and frame["frame_end_sec"] <= end + TIMESTAMP_TOLERANCE
                and frame["frame_start_sec"] >= turn_start - TIMESTAMP_TOLERANCE
                and frame["frame_end_sec"] <= turn_end + TIMESTAMP_TOLERANCE
            ]
            runs = consecutive_runs(complete)
            run = max(runs, key=len, default=[])
            if len(run) >= MIN_CONSECUTIVE_FRAMES:
                source_times = np.asarray(
                    [frame["frame_start_sec"] for frame in run], dtype=np.float64
                )
                target_times = np.linspace(source_times[0], source_times[-1], 512)
                values = np.stack([frame["values"] for frame in run], axis=0)
                interpolated = np.vstack(
                    [np.interp(target_times, source_times, values[:, channel]) for channel in range(channels)]
                ).astype(np.float32)
                assert interpolated.shape == (channels, 512)
                assert target_times[0] >= source_times[0]
                assert target_times[-1] <= source_times[-1]
                windows.append(
                    {
                        "start_sec": start,
                        "end_sec": end,
                        "source_frame_count": len(run),
                        "source_start_sec": float(source_times[0]),
                        "source_end_sec": float(source_times[-1]),
                        "tensor": interpolated,
                    }
                )
            index += 1
    if len(windows) <= 2:
        return windows
    return [windows[0], windows[-1]]


def synthetic_checks(channels: int) -> None:
    frames = []
    for index in range(100):
        start = index * NOMINAL_HOP_SECONDS
        frames.append(
            {
                "turn_id": "synthetic",
                "turn_start_sec": 0.0,
                "turn_end_sec": 1.01,
                "frame_start_sec": start,
                "frame_end_sec": start + 0.02,
                "values": np.full(channels, index, dtype=np.float32),
            }
        )
    windows = temporalize(frames, channels)
    assert len(windows) == 1
    assert windows[0]["tensor"].shape == (channels, 512)
    assert np.isfinite(windows[0]["tensor"]).all()
    assert windows[0]["source_frame_count"] == 99


def select_artifacts(
    corpus: str,
    root: Path,
    manifest: dict[tuple[str, str], dict[str, str]],
    channels: list[str],
    validator: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    rows = read_csv(root / "splits" / "folds" / "fold_0_train.csv")
    selected: list[dict[str, Any]] = []
    regular: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        record = manifest.get((corpus, row["sample_id"]))
        if record is None or record.get("egemaps_lld_status") != "VALID":
            continue
        path = validator.artifact_path(root, row)
        diagnostic = validator.inspect_artifact(
            path, channels, NOMINAL_HOP_SECONDS, corpus, "terminal-exclude"
        )
        if diagnostic["status"] != "VALID":
            raise ValueError("selected TRAIN artifact failed Policy C validation")
        entry = {"path": path, "diagnostic": diagnostic}
        if diagnostic["excluded_frame_count"]:
            excluded.append(entry)
        else:
            regular.append(entry)
        if corpus == "WSM" and len(regular) >= MAX_ARTIFACTS:
            break
        if corpus == "CARE" and excluded and len(regular) >= MAX_ARTIFACTS - 1:
            break
    if corpus == "CARE":
        selected = excluded[:1] + regular[: MAX_ARTIFACTS - 1]
        if not excluded:
            selected = regular[:MAX_ARTIFACTS]
    else:
        selected = regular[:MAX_ARTIFACTS]
    if not selected:
        raise ValueError(f"no eligible {corpus} TRAIN artifacts")
    return selected, len(rows), len(regular) + len(excluded)


def run_inference(batch: np.ndarray, checkpoint: Path, device: str) -> dict[str, Any]:
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    model_file = checkpoint / "model.safetensors"
    assert sha256(model_file) == EXPECTED_MODEL_SHA256
    network = MantisV2(device=device, return_transf_layer=2, output_token="combined")
    network = network.from_pretrained(str(checkpoint))
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    before = [parameter.detach().clone() for parameter in network.parameters()]
    inputs = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
    assert inputs.shape[1:] == (25, 512)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer = MantisTrainer(device=device, network=network)
    with torch.no_grad():
        output = trainer.transform(inputs, three_dim=True, to_numpy=False)
        repeated = trainer.transform(inputs.clone(), three_dim=True, to_numpy=False)
    assert output.shape == (batch.shape[0], 25, 512)
    assert torch.isfinite(output).all().item()
    assert torch.equal(output, repeated)
    after = [parameter.detach() for parameter in network.parameters()]
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    assert not any(parameter.grad is not None for parameter in network.parameters())
    memory = {}
    if device == "cuda":
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    return {
        "device": device,
        "input_shape": list(inputs.shape),
        "output_shape": list(output.shape),
        "finite": True,
        "repeated_inference_equal": True,
        "parameters_unchanged": True,
        "gradients_absent": True,
        **memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("reports/feature_audit/sample_feature_availability.csv"))
    parser.add_argument("--care-root", type=Path, default=Path("/media/maxim/Databases/CARE"))
    parser.add_argument("--wsm-root", type=Path, default=Path("/media/maxim/Databases/WSM_NEW"))
    parser.add_argument("--registry", type=Path, default=Path("/media/maxim/Databases/CARE/splits/features.json"))
    args = parser.parse_args()

    validator = load_validator()
    channels = validator.registry_channels(args.registry)
    assert len(channels) == 25
    synthetic_checks(len(channels))
    manifest = {
        (row["corpus"], row["sample_id"]): row for row in read_csv(args.manifest)
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    population_counts = {}
    scanned_counts = {}
    for corpus, root in (("CARE", args.care_root), ("WSM", args.wsm_root)):
        selected[corpus], population_counts[corpus], scanned_counts[corpus] = select_artifacts(
            corpus, root, manifest, channels, validator
        )

    all_windows = []
    artifact_summary = {}
    for corpus, entries in selected.items():
        artifact_summary[corpus] = {
            "selected_artifacts": len(entries),
            "policy_c_excluded_frames": 0,
            "policy_c_exclusion_reasons": {},
            "source_frame_rows": 0,
            "accepted_windows": 0,
        }
        for entry in entries:
            frames, exclusions, exclusion_reasons = artifact_frames(
                entry["path"], channels, validator
            )
            windows = temporalize(frames, len(channels))
            artifact_summary[corpus]["policy_c_excluded_frames"] += exclusions
            for reason in exclusion_reasons:
                reasons = artifact_summary[corpus]["policy_c_exclusion_reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
            artifact_summary[corpus]["source_frame_rows"] += entry["diagnostic"]["original_rows"]
            artifact_summary[corpus]["accepted_windows"] += len(windows)
            all_windows.extend(windows)
    assert all_windows
    batch = np.stack([window["tensor"] for window in all_windows], axis=0)
    assert batch.shape[1:] == (25, 512)
    assert np.isfinite(batch).all()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = run_inference(batch, args.checkpoint, device)
    result = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "model_sha256": sha256(args.checkpoint / "model.safetensors"),
        "fold": 0,
        "split": "TRAIN",
        "channels": len(channels),
        "channel_order_verified": True,
        "population_counts": population_counts,
        "bounded_candidate_rows_scanned": scanned_counts,
        "artifact_summary": artifact_summary,
        "total_accepted_windows": len(all_windows),
        "batched_input_shape": list(batch.shape),
        "interpolation": "float32 linear interpolation over observed frame-start times; no extrapolation",
        "min_consecutive_valid_frames": MIN_CONSECUTIVE_FRAMES,
        "policy_c_excluded_frames_not_in_input": True,
        "inference": inference,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
