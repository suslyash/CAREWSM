"""Paired native and timestamp-respecting Mantis representation adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from audio.data.joint_datamodule import JointDataModule, NativeFeature, Sample


@dataclass(frozen=True)
class Representation:
    mode: str
    values: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    metadata: dict[str, Any]


def native_representation(dm: JointDataModule, sample: Sample, family: str) -> Representation:
    """Keep the audited sequence unchanged; static inputs are never given fake time."""
    feature: NativeFeature = dm.load_native(sample, family)
    return Representation("native", feature.values[None, ...],
                          np.ones((1, len(feature.values)), dtype=bool),
                          np.asarray([len(feature.values)], dtype=np.int64),
                          {"corpus": sample.corpus, "sample_id": sample.sample_id,
                           "profile_id": sample.profile_id, "family": family,
                           "source": str(feature.path), "source_metadata": feature.metadata})


def pooled_native_representation(dm: JointDataModule, sample: Sample, family: str,
                                 max_sequence_length: int) -> Representation:
    """Pool contiguous bins over the complete trajectory for bounded smoke runs."""
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    representation = native_representation(dm, sample, family)
    values = representation.values[0]
    if len(values) <= max_sequence_length:
        return representation
    edges = np.linspace(0, len(values), max_sequence_length + 1, dtype=np.int64)
    pooled = np.stack([values[edges[i]:edges[i + 1]].mean(axis=0)
                       for i in range(max_sequence_length)], axis=0)
    return Representation("native_binned", pooled[None, ...],
                          np.ones((1, len(pooled)), dtype=bool),
                          np.asarray([len(pooled)], dtype=np.int64),
                          {**representation.metadata, "full_length": len(values),
                           "pooling": "contiguous_full_trajectory_mean",
                           "max_sequence_length": max_sequence_length})


def mantis_windows(feature: NativeFeature, window_sec: float = 1.0,
                   hop_sec: float = 0.5) -> Representation:
    """Window a timestamped frame/segment sequence without crossing turn boundaries."""
    required = {"turn_id", "turn_start_sec", "turn_end_sec", "frame_start_sec", "frame_end_sec"}
    if not required.issubset(feature.metadata):
        raise ValueError("Mantis windowing requires real frame/segment and turn timestamps")
    starts = feature.metadata["frame_start_sec"].astype(float)
    ends = feature.metadata["frame_end_sec"].astype(float)
    turns = feature.metadata["turn_id"]
    if np.any(np.diff(starts) < 0) or np.any(ends < starts):
        raise ValueError("timestamps are not ordered")
    chunks, intervals, fractions, source_masks = [], [], [], []
    for turn in dict.fromkeys(turns.tolist()):
        idx = np.flatnonzero(turns == turn)
        turn_start = float(feature.metadata["turn_start_sec"][idx[0]])
        turn_end = float(feature.metadata["turn_end_sec"][idx[0]])
        if turn_end <= turn_start:
            continue
        t = turn_start
        while t + window_sec <= turn_end + 1e-7:
            observed = (starts[idx] < t + window_sec) & (ends[idx] > t)
            if observed.any():
                chunks.append(feature.values[idx[observed]].mean(axis=0))
                source_masks.append(observed)
                intervals.append((t, t + window_sec, turn))
                fractions.append(float(np.clip((np.minimum(ends[idx[observed]], t + window_sec) -
                                               np.maximum(starts[idx[observed]], t)).sum() / window_sec, 0, 1)))
            t += hop_sec
    values = np.asarray(chunks, dtype=np.float32).reshape((-1, feature.values.shape[1]))
    return Representation("mantis", values[None, ...], np.ones((1, len(values)), bool),
                          np.asarray([len(values)], np.int64),
                          {"windows": np.asarray(intervals, dtype=object),
                           "valid_observation_fraction": np.asarray(fractions, np.float32),
                           "source_family": feature.family, "window_sec": window_sec,
                           "hop_sec": hop_sec, "source_path": str(feature.path)})


def official_mantis_probe() -> dict[str, Any]:
    """Return executable compatibility facts without downloading or guessing a model."""
    try:
        import mantis  # type: ignore
    except ImportError:
        return {"compatible": False, "status": "BLOCKED", "reason": "mantis-tsfm is not installed"}
    return {"compatible": False, "status": "REVIEW_REQUIRED",
            "reason": "package is present; checkpoint/revision and model probe must be pinned" ,
            "module": getattr(mantis, "__file__", None)}
