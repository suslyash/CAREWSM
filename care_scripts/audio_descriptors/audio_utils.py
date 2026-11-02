"""Shared helpers for CARE/WSM audio feature extraction."""

import csv
import os
import sys
from pathlib import Path
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turn_utils import extract_participant_turns


@lru_cache(maxsize=8)
def _load_registry(metadata_csv):
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_sample(wav_path, metadata_csv=None):
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    parts = stem.split("__")
    if len(parts) == 4:
        return tuple(parts)
    profile_id = os.path.basename(os.path.dirname(os.path.dirname(wav_path)))
    video_id = stem
    if metadata_csv:
        rows = [row for row in _load_registry(os.path.abspath(metadata_csv))
                if row.get("profile_id") == profile_id and row.get("video_id") == video_id]
        if len(rows) == 1:
            row = rows[0]
            return row["label_id"], row["subgroup"], profile_id, video_id
        if len(rows) > 1:
            raise ValueError(f"Ambiguous registry match for {wav_path}")
        raise ValueError(f"No registry row for profile/video: {profile_id}/{video_id}")
    for component in os.path.normpath(wav_path).split(os.sep):
        if component.lower() in {"depression", "parkinson"}:
            subgroup = component.upper()
            return subgroup, subgroup, profile_id, video_id
    raise ValueError(f"Cannot infer identifiers from {wav_path}; pass --metadata-csv")


def sample_output_dir(wav_path, output_root, metadata_csv=None):
    group_id, subgroup_id, profile_id, video_id = resolve_sample(wav_path, metadata_csv)
    output_dir = os.path.join(output_root, group_id, subgroup_id,
                              f"PROFILE_{profile_id}", f"VIDEO_{video_id}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_turn_times(wav_path, output_dir, duration_sec, use_diarization=False):
    rttm_path = os.path.join(output_dir, "diarizen_diarization.rttm")
    if use_diarization and os.path.exists(rttm_path):
        return extract_participant_turns(rttm_path)
    return [(0.0, float(duration_sec))]
