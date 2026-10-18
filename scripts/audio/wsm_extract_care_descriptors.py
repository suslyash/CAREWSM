#!/usr/bin/env python3
"""Bounded, manifest-driven CARE eGeMAPS/DisVoice extraction for WSM.

This standalone script never imports ChimeraML. It uses only the seven
lightweight audio descriptor families authorized by TASK-003A. Dependency and
model provisioning are intentionally outside this script's scope.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Iterable

MAX_FILES = 3
SAMPLE_RATE = 16000
FEATURES = (
    "egemaps_func.csv",
    "egemaps_lld.csv",
    "disvoice_prosody_static.csv",
    "disvoice_prosody_dynamic.csv",
    "disvoice_phonation_static.csv",
    "disvoice_phonation_dynamic.csv",
    "disvoice_articulation_static.csv",
)
PREFERRED_STRATA = ("DEPRESSION", "CONTROL", "PARKINSON")
META_TURN = ["turn_id", "turn_start_sec", "turn_end_sec"]
META_FRAME = META_TURN + ["frame_start_sec", "frame_end_sec"]


def atomic_csv(df, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir="/tmp")
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        df.to_csv(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_rows(root: Path, split: str, limit: int) -> list[dict]:
    manifest = root / "splits" / f"{split}.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "video_path", "feature_dir", "split", "storage_type", "stratum"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"missing manifest columns: {missing}")
        rows = []
        for row in reader:
            if row.get("split") != split or row.get("storage_type") != "files":
                continue
            video_rel = row["video_path"]
            feature_rel = row["feature_dir"]
            video = (root / video_rel).resolve()
            destination = (root / feature_rel).resolve()
            if Path(video_rel).is_absolute() or Path(feature_rel).is_absolute():
                continue
            if root.resolve() not in video.parents or root.resolve() not in destination.parents:
                continue
            if not video.is_file():
                continue
            rows.append({**row, "video": video, "destination": destination})
    rows.sort(key=lambda row: (row["sample_id"], row["video_path"]))
    selected = []
    used = set()
    for stratum in PREFERRED_STRATA:
        row = next((item for item in rows if item.get("stratum") == stratum), None)
        if row and row["sample_id"] not in used:
            selected.append(row)
            used.add(row["sample_id"])
    return selected[: min(limit, MAX_FILES)]


def parse_rttm(path: Path) -> list[tuple[str, float, float]]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            continue
        start = float(fields[3]); end = start + float(fields[4])
        segments.append((fields[7], start, end))
    if not segments:
        raise RuntimeError(f"RTTM has no SPEAKER segments: {path}")
    durations: dict[str, float] = {}
    for speaker, start, end in segments:
        durations[speaker] = durations.get(speaker, 0.0) + max(0.0, end - start)
    ordered = sorted(durations, key=durations.get, reverse=True)
    mapping = {ordered[0]: "PARTICIPANT"}
    mapping.update({speaker: "INTERVIEWER" for speaker in ordered[1:]})
    return [(mapping[speaker], start, end) for speaker, start, end in segments]


def participant_turns(segments: Iterable[tuple[str, float, float]], max_gap: float = 3.0, min_duration: float = 0.3):
    ordered = sorted(segments, key=lambda item: item[1])
    turns = []
    current_start = current_end = None
    for speaker, start, end in ordered:
        if speaker == "INTERVIEWER":
            if current_start is not None:
                turns.append((current_start, current_end))
                current_start = current_end = None
            continue
        if current_start is None:
            current_start, current_end = start, end
        elif start - current_end <= max_gap:
            current_end = end
        else:
            turns.append((current_start, current_end))
            current_start, current_end = start, end
    if current_start is not None:
        turns.append((current_start, current_end))
    return [(start, end) for start, end in turns if end - start >= min_duration]


def create_turns(row: dict, temporary: Path, args) -> tuple[list[tuple[float, float]], dict]:
    destination = row["destination"]
    rttm = destination / "diarizen_diarization.rttm"
    if rttm.is_file():
        return [(start, end) for _, start, end in parse_rttm(rttm)], {"source": str(rttm), "policy": "CARE RTTM longest-speaker relabeling"}
    if not args.weights or not args.embedding_weights:
        raise RuntimeError(f"no RTTM at {rttm}; explicit local DiariZen weights are required")
    if not args.weights.is_dir() or not args.embedding_weights.is_file():
        raise RuntimeError("explicit local DiariZen weights/embedding file are missing")
    missing = [name for name in ("torch", "torchaudio", "diarizen") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"missing local DiariZen dependencies: {', '.join(missing)}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU DiariZen is prohibited")
    from diarizen.pipelines.inference import DiariZenPipeline
    mono = temporary / f"{row['sample_id']}.mono16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(row["video"]), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(mono)], check=True)
    pipeline = DiariZenPipeline(diarizen_hub=args.weights, embedding_model=str(args.embedding_weights), rttm_out_dir=None)
    diarization = pipeline(str(mono))
    segments = []
    durations: dict[str, float] = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start, end = float(segment.start), float(segment.end)
        segments.append((str(speaker), start, end))
        durations[str(speaker)] = durations.get(str(speaker), 0.0) + max(0.0, end - start)
    if not segments:
        raise RuntimeError("DiariZen returned no speech segments")
    ordered = sorted(durations, key=durations.get, reverse=True)
    mapping = {ordered[0]: "PARTICIPANT"}
    mapping.update({speaker: "INTERVIEWER" for speaker in ordered[1:]})
    labeled = [(mapping[speaker], start, end) for speaker, start, end in segments]
    return participant_turns(labeled), {"source": "local DiariZen inference", "policy": "provisional longest detected speaker = PARTICIPANT", "speaker_count": len(durations)}


def feature_frame(smile, signal, turn_id, start, end, level: str):
    features = smile.process_signal(signal.astype("float32"), SAMPLE_RATE).reset_index(drop=True)
    features["turn_id"] = turn_id
    features["turn_start_sec"] = start
    features["turn_end_sec"] = end
    meta = META_TURN
    if level == "lld":
        if not hasattr(smile, "process_signal"):
            raise RuntimeError("invalid openSMILE object")
        # openSMILE's frame index is not retained after reset_index; process again
        raw = smile.process_signal(signal.astype("float32"), SAMPLE_RATE)
        features["frame_start_sec"] = raw.index.get_level_values("start").total_seconds() + start
        features["frame_end_sec"] = raw.index.get_level_values("end").total_seconds() + start
        meta = META_FRAME
    return features[meta + [column for column in features.columns if column not in meta]]


def as_dataframe(value, expected_width: int):
    import numpy as np
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        array = np.asarray(value)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        frame = pd.DataFrame(array)
    if frame.shape[1] != expected_width:
        raise RuntimeError(f"DisVoice width {frame.shape[1]} != expected {expected_width}")
    return frame


def extract_row(row: dict, turns, temporary: Path, args):
    import librosa
    import numpy as np
    import opensmile
    import pandas as pd
    from disvoice.articulation import Articulation
    from disvoice.phonation import Phonation
    from disvoice.prosody import Prosody
    import soundfile as sf

    waveform, _ = librosa.load(str(row["video_wav"]), sr=SAMPLE_RATE, mono=True)
    smile_func = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    smile_lld = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    extractors = {"prosody": Prosody(), "phonation": Phonation(), "articulation": Articulation()}
    accum = {name: [] for name in FEATURES}
    widths = {"prosody_static": 103, "prosody_dynamic": 13, "phonation_static": 28, "phonation_dynamic": 7, "articulation_static": 488}
    for turn_id, (start, end) in enumerate(turns, start=1):
        chunk = waveform[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)].astype(np.float32)
        if len(chunk) == 0:
            continue
        accum["egemaps_func.csv"].append(feature_frame(smile_func, chunk, turn_id, start, end, "func"))
        accum["egemaps_lld.csv"].append(feature_frame(smile_lld, chunk, turn_id, start, end, "lld"))
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temporary, delete=True) as handle:
            sf.write(handle.name, chunk, SAMPLE_RATE)
            for name, extractor in extractors.items():
                static_options = (True,) if name == "articulation" else (True, False)
                for static in static_options:
                    kind = "static" if static else "dynamic"
                    value = extractor.extract_features_file(handle.name, static=static, fmt="csv")
                    prefix = f"disvoice_{name}_{kind}.csv"
                    frame = as_dataframe(value, widths[f"{name}_{kind}"])
                    frame["turn_id"] = turn_id
                    frame["turn_start_sec"] = start
                    frame["turn_end_sec"] = end
                    if not static:
                        hop = {"prosody": 0.01, "phonation": 0.02, "articulation": 0.04}[name]
                        offset = {"prosody": 0, "phonation": 11, "articulation": 2}[name]
                        frame["frame_start_sec"] = start + (np.arange(len(frame)) + offset) * hop
                        frame["frame_end_sec"] = frame["frame_start_sec"] + hop
                    meta = META_TURN if static else META_FRAME
                    accum[prefix].append(frame[meta + [c for c in frame.columns if c not in meta]])
    for filename, frames in accum.items():
        if not frames:
            raise RuntimeError(f"no rows produced for {filename}")
        atomic_csv(pd.concat(frames, ignore_index=True), row["destination"] / filename)
    return {name: len(pd.concat(frames, ignore_index=True)) for name, frames in accum.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsm-root", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "dev", "test"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--embedding-weights", type=Path)
    args = parser.parse_args()
    root = args.wsm_root.resolve()
    rows = read_rows(root, args.split, args.limit)
    result = {"task": "TASK-003A", "status": "dry_run" if args.dry_run else "blocked", "split": args.split, "limit": min(args.limit, MAX_FILES), "selected": []}
    for row in rows:
        result["selected"].append({"sample_id": row["sample_id"], "video_path": row["video_path"], "feature_dir": row["feature_dir"], "destination": str(row["destination"]), "stratum": row.get("stratum"), "task": row.get("task")})
    invalid_destinations = [row["feature_dir"] for row in rows if not row["feature_dir"].startswith("extracted/")]
    if invalid_destinations:
        result["blocker"] = {"type": "manifest_feature_dir_contract", "paths": invalid_destinations, "message": "feature_dir must be an existing WSM manifest destination under extracted/; no output was written."}
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    if invalid_destinations:
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    missing = [name for name in ("numpy", "pandas", "librosa", "soundfile", "opensmile", "disvoice") if importlib.util.find_spec(name) is None]
    if missing:
        result["blocker"] = {"type": "missing_local_dependency", "packages": missing, "message": "No installation or download was attempted."}
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    result["runs"] = []
    with tempfile.TemporaryDirectory(prefix="carewsm-task003a-") as temp_name:
        temporary = Path(temp_name)
        for row in rows:
            started = time.perf_counter()
            try:
                wav = temporary / f"{row['sample_id']}.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(row["video"]), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
                row["video_wav"] = wav
                turns, provenance = create_turns(row, temporary, args)
                counts = extract_row(row, turns, temporary, args)
                result["runs"].append({"sample_id": row["sample_id"], "status": "ok", "turn_count": len(turns), "turn_provenance": provenance, "rows_per_file": counts, "elapsed_s": round(time.perf_counter() - started, 3)})
            except Exception as exc:
                result["runs"].append({"sample_id": row["sample_id"], "status": "blocked", "error": str(exc)})
        result["temporary_artifacts_cleaned"] = True
    if result.get("runs") and all(run["status"] == "ok" for run in result["runs"]):
        result["status"] = "ok"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
