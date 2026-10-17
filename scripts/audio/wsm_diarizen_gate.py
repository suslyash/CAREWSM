#!/usr/bin/env python3
"""Bounded, local-only DiariZen feasibility gate for existing WSM TRAIN media."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import wave

MAX_FILES = 3
MAX_SECONDS = 120.0
PREFERRED_STRATA = ("DEPRESSION", "CONTROL", "PARKINSON")


def run_json(command: list[str]) -> dict:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def select_rows(root: Path) -> list[dict]:
    manifest = root / "splits" / "train.csv"
    rows: list[dict] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "video_path", "split", "storage_type", "stratum"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"missing manifest columns: {missing}")
        for row in reader:
            if row.get("split") != "train":
                continue
            relative = row["video_path"]
            candidate = (root / relative).resolve()
            if (
                row.get("storage_type") != "files"
                or Path(relative).is_absolute()
                or root.resolve() not in candidate.parents
                or not candidate.is_file()
            ):
                continue
            rows.append({**row, "resolved_path": str(candidate)})

    rows.sort(key=lambda row: (row["sample_id"], row["video_path"]))
    selected: list[dict] = []
    used: set[str] = set()
    for wanted in PREFERRED_STRATA:
        match = next((row for row in rows if row.get("stratum") == wanted), None)
        if match and match["sample_id"] not in used:
            selected.append(match)
            used.add(match["sample_id"])
    return selected[:MAX_FILES]


def probe(path: Path) -> dict:
    payload = run_json([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ])
    audio = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"), None)
    if audio is None:
        raise RuntimeError("no audio stream")
    return {
        "duration_s": float(payload.get("format", {}).get("duration", 0.0)),
        "codec": audio.get("codec_name"),
        "sample_rate_hz": int(audio.get("sample_rate", 0)),
        "channels": int(audio.get("channels", 0)),
    }


def stereo_check(source: Path, temporary_dir: Path, seconds: float, stem: str) -> dict:
    stereo_wav = temporary_dir / f"{stem}.stereo.wav"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(source), "-t", str(min(seconds, MAX_SECONDS)),
        "-ar", "16000", "-ac", "2", "-c:a", "pcm_s16le", str(stereo_wav)
    ], check=True)
    with wave.open(str(stereo_wav), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
        width = handle.getsampwidth()
    if channels != 2 or width != 2:
        return {"status": "unmeasured", "reason": "temporary stereo decode did not produce 2x16-bit PCM"}
    values = [int.from_bytes(frames[i:i + 2], "little", signed=True) for i in range(0, len(frames), 2)]
    left, right = values[::2], values[1::2]
    if not left or not right:
        return {"status": "unmeasured", "reason": "temporary stereo decode was empty"}
    left_rms = math.sqrt(sum(value * value for value in left) / len(left))
    right_rms = math.sqrt(sum(value * value for value in right) / len(right))
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denom = math.sqrt(sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right))
    correlation = covariance / denom if denom else None
    return {
        "status": "measured",
        "left_rms": round(left_rms, 3),
        "right_rms": round(right_rms, 3),
        "absolute_rms_ratio": round(min(left_rms, right_rms) / max(left_rms, right_rms), 6) if max(left_rms, right_rms) else None,
        "channel_correlation": round(correlation, 6) if correlation is not None else None,
        "interpretation": "not respondent/interviewer separated by evidence available here",
    }


def make_mono(source: Path, temporary_dir: Path, seconds: float, stem: str) -> Path:
    mono_wav = temporary_dir / f"{stem}.mono16k.wav"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(source), "-t", str(min(seconds, MAX_SECONDS)),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(mono_wav)
    ], check=True)
    return mono_wav


def summarize_diarization(diarization, audio_duration: float) -> dict:
    segments = []
    speaker_durations: dict[str, float] = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        duration = max(0.0, float(segment.end) - float(segment.start))
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
        segments.append({"speaker": str(speaker), "start_s": float(segment.start), "end_s": float(segment.end), "duration_s": duration})
    ordered = sorted(speaker_durations, key=speaker_durations.get, reverse=True)
    participant = ordered[0] if ordered else None
    return {
        "speaker_count": len(speaker_durations),
        "segment_count": len(segments),
        "speaker_durations_s": {str(k): round(v, 6) for k, v in speaker_durations.items()},
        "participant_speaker_by_longest_speech": participant,
        "participant_turn_coverage": round(speaker_durations.get(participant, 0.0) / audio_duration, 6) if participant and audio_duration else None,
        "segments": segments,
        "respondent_interviewer_sanity": "unverified",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsm-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path, help="existing local DiariZen model directory; never downloaded")
    args = parser.parse_args()
    root = args.wsm_root.resolve()
    weights = args.weights.expanduser().resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    result: dict = {
        "status": "blocked",
        "task": "TASK-002B",
        "wsm_root": str(root),
        "split": "train",
        "selection_cap": MAX_FILES,
        "duration_cap_s": MAX_SECONDS,
        "weights": str(weights),
        "offline_model_policy": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "downloads": False},
    }
    try:
        if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
            raise RuntimeError("ffprobe and ffmpeg are both required")
        selected = select_rows(root)
        result["selected"] = []
        with tempfile.TemporaryDirectory(prefix="carewsm-diarizen-") as temp:
            temp_dir = Path(temp)
            for row in selected:
                metadata = probe(Path(row["resolved_path"]))
                stem = row["sample_id"].replace("/", "_")
                metadata.update({"sample_id": row["sample_id"], "stratum": row.get("stratum"), "task": row.get("task"), "source_path": row["video_path"]})
                metadata["stereo_channel_check"] = stereo_check(Path(row["resolved_path"]), temp_dir, metadata["duration_s"], stem)
                result["selected"].append(metadata)

            missing = [name for name in ("diarizen", "torchaudio", "tqdm") if importlib.util.find_spec(name) is None]
            if missing:
                result["blocker"] = {"type": "missing_local_dependency", "packages": missing, "message": "DiariZen was not invoked; no installation or model download was attempted."}
            elif not weights.is_dir():
                result["blocker"] = {"type": "missing_local_weights", "path": str(weights), "message": "--weights must name an existing local model directory; no download was attempted."}
            else:
                import torch
                if not torch.cuda.is_available():
                    result["blocker"] = {"type": "unavailable_gpu", "message": "DiariZen was not invoked because CUDA is unavailable locally."}
                else:
                    from diarizen.pipelines.inference import DiariZenPipeline
                    pipeline = DiariZenPipeline.from_pretrained(str(weights))
                    result["runs"] = []
                    for row, metadata in zip(selected, result["selected"]):
                        stem = row["sample_id"].replace("/", "_")
                        mono_wav = make_mono(Path(row["resolved_path"]), temp_dir, metadata["duration_s"], stem)
                        audio_duration = min(metadata["duration_s"], MAX_SECONDS)
                        started = time.perf_counter()
                        diarization = pipeline(str(mono_wav))
                        elapsed = time.perf_counter() - started
                        summary = summarize_diarization(diarization, audio_duration)
                        rttm_path = temp_dir / f"{stem}.rttm"
                        with rttm_path.open("w", encoding="utf-8") as handle:
                            diarization.write_rttm(handle)
                        result["runs"].append({"sample_id": row["sample_id"], "audio_duration_s": audio_duration, "elapsed_s": round(elapsed, 6), "rtf": round(elapsed / audio_duration, 6) if audio_duration else None, **summary})
                    result["status"] = "ok"
            result["temporary_artifacts_cleaned"] = False
        result["temporary_artifacts_cleaned"] = True
    except Exception as exc:
        result["blocker"] = {"type": "preflight_or_runtime_failure", "message": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
