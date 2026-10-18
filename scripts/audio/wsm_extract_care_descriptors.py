#!/usr/bin/env python3
"""Bounded CARE-compatible seven-family audio extraction for WSM."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from typing import Iterable

MAX_FILES = 3
SAMPLE_RATE = 16000
PREFERRED_STRATA = ("DEPRESSION", "CONTROL", "PARKINSON")
FILES = (
    "egemaps_func.csv", "egemaps_lld.csv",
    "disvoice_prosody_static.csv", "disvoice_prosody_dynamic.csv",
    "disvoice_phonation_static.csv", "disvoice_phonation_dynamic.csv",
    "disvoice_articulation_static.csv",
)
META_TURN = ["turn_id", "turn_start_sec", "turn_end_sec"]
META_FRAME = META_TURN + ["frame_start_sec", "frame_end_sec"]


@contextmanager
def safe_disvoice_temp_names():
    """Give DisVoice and NamedTemporaryFile collision-free per-call names."""
    previous = tempfile._get_candidate_names

    def candidates():
        while True:
            yield "carewsm-" + uuid.uuid4().hex

    tempfile._get_candidate_names = candidates
    try:
        yield
    finally:
        tempfile._get_candidate_names = previous


def reference_headers(care_root: Path, filename: str) -> list[str]:
    matches = sorted(care_root.glob(f"extracted/**/{filename}"))
    if not matches:
        raise RuntimeError(f"CARE reference file not found for {filename}")
    with matches[0].open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def numeric_width(headers: list[str], filename: str) -> int:
    return len(headers) - len(META_FRAME if "dynamic" in filename or filename == "egemaps_lld.csv" else META_TURN)


def validate_existing(path: Path, expected: list[str]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    metadata = set(META_FRAME if "dynamic" in path.name or path.name == "egemaps_lld.csv" else META_TURN)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise RuntimeError(f"existing feature has invalid CARE header: {path}")
        row_count = 0
        for row_number, row in enumerate(reader, 2):
            row_count += 1
            if None in row:
                raise RuntimeError(f"existing feature has extra columns: {path}:{row_number}")
            for column in expected:
                value = row.get(column)
                if value is None or (value.strip() == "" and column in metadata):
                    raise RuntimeError(f"existing feature has missing metadata: {path}:{row_number}:{column}")
                if value.strip() == "":
                    continue
                try:
                    number = float(value)
                except ValueError as exc:
                    raise RuntimeError(f"existing feature has nonnumeric value: {path}:{row_number}:{column}") from exc
                if not math.isfinite(number):
                    raise RuntimeError(f"existing feature has NaN/Inf: {path}:{row_number}:{column}")
            if "turn_id" in metadata and float(row["turn_id"]) != int(float(row["turn_id"])):
                raise RuntimeError(f"existing feature has noninteger turn_id: {path}:{row_number}")
        if row_count == 0:
            raise RuntimeError(f"existing feature is empty: {path}")
    return True


def atomic_csv(frame, destination: Path) -> None:
    """Write and rename within destination's filesystem; never use /tmp here."""
    destination.parent.mkdir(parents=False, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing feature: {destination}")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_destination(root: Path, row: dict) -> Path:
    required = ("label_id", "subgroup", "profile_id", "video_id")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise RuntimeError(f"manifest row missing output fields: {missing}")
    destination = (root / "extracted" / row["label_id"] / row["subgroup"] / f"PROFILE_{row['profile_id']}" / f"VIDEO_{row['video_id']}").resolve()
    profile_dir = destination.parent
    if root.resolve() not in destination.parents or not profile_dir.is_dir():
        raise RuntimeError(f"expected existing WSM extracted profile directory is missing: {profile_dir}")
    return destination


def read_rows(root: Path, split: str, limit: int) -> list[dict]:
    manifest = root / "splits" / f"{split}.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "video_path", "feature_dir", "split", "storage_type", "stratum", "label_id", "subgroup", "profile_id", "video_id"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"missing manifest columns: {missing}")
        rows = []
        for row in reader:
            if row.get("split") != split or row.get("storage_type") != "files":
                continue
            video_rel = row["video_path"]
            video = (root / video_rel).resolve()
            if Path(video_rel).is_absolute() or root.resolve() not in video.parents or not video.is_file():
                continue
            rows.append({**row, "video": video, "destination": resolve_destination(root, row)})
    rows.sort(key=lambda row: (row["sample_id"], row["video_path"]))
    selected = []
    used = set()
    for stratum in PREFERRED_STRATA:
        row = next((item for item in rows if item.get("stratum") == stratum), None)
        if row and row["sample_id"] not in used:
            selected.append(row)
            used.add(row["sample_id"])
    destinations = [str(row["destination"]) for row in selected[:min(limit, MAX_FILES)]]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("selected rows have duplicate output destinations")
    return selected[:min(limit, MAX_FILES)]


def parse_rttm(path: Path) -> list[tuple[str, float, float]]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0] == "SPEAKER":
            start = float(fields[3]); end = start + float(fields[4])
            segments.append((fields[7], start, end))
    if not segments:
        raise RuntimeError(f"RTTM has no SPEAKER segments: {path}")
    durations = {}
    for speaker, start, end in segments:
        durations[speaker] = durations.get(speaker, 0.0) + max(0.0, end - start)
    ordered = sorted(durations, key=durations.get, reverse=True)
    mapping = {ordered[0]: "PARTICIPANT"}
    mapping.update({speaker: "INTERVIEWER" for speaker in ordered[1:]})
    return [(mapping[speaker], start, end) for speaker, start, end in segments]


def participant_turns(segments: Iterable[tuple[str, float, float]], max_gap=3.0, min_duration=0.3):
    turns = []
    current_start = current_end = None
    for speaker, start, end in sorted(segments, key=lambda item: item[1]):
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


def create_turns(row: dict, scratch: Path, args) -> tuple[list[tuple[float, float]], dict]:
    rttm = row["destination"] / "diarizen_diarization.rttm"
    if rttm.is_file():
        return participant_turns(parse_rttm(rttm)), {"source": str(rttm), "policy": "CARE longest-speaker relabeling"}
    if not args.weights or not args.embedding_weights:
        raise RuntimeError(f"no existing RTTM at {rttm}; explicit local DiariZen weights are required")
    if not args.weights.is_dir() or not args.embedding_weights.is_file():
        raise RuntimeError("explicit local DiariZen weights or embedding file is missing")
    missing = [name for name in ("torch", "torchaudio", "diarizen") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"missing local DiariZen dependencies: {missing}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU DiariZen is prohibited")
    from diarizen.pipelines.inference import DiariZenPipeline
    mono = scratch / f"{row['sample_id']}.mono16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(row["video"]), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(mono)], check=True)
    pipeline = DiariZenPipeline(diarizen_hub=args.weights, embedding_model=str(args.embedding_weights), rttm_out_dir=None)
    diarization = pipeline(str(mono))
    segments = []; durations = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start, end, label = float(segment.start), float(segment.end), str(speaker)
        segments.append((label, start, end))
        durations[label] = durations.get(label, 0.0) + max(0.0, end - start)
    if not segments:
        raise RuntimeError("DiariZen returned no speech segments")
    ordered = sorted(durations, key=durations.get, reverse=True)
    mapping = {ordered[0]: "PARTICIPANT"}
    mapping.update({speaker: "INTERVIEWER" for speaker in ordered[1:]})
    return participant_turns([(mapping[speaker], start, end) for speaker, start, end in segments]), {"source": "local DiariZen inference", "policy": "provisional longest detected speaker = PARTICIPANT", "speaker_count": len(durations)}


def configure_headless_praat(scratch: Path) -> None:
    praat = shutil.which("praat")
    if praat is None:
        raise RuntimeError("Praat executable is unavailable")
    wrapper = scratch / "praat"
    wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(praat) + " --run \"" + chr(36) + "@\"\n", encoding="utf-8")
    wrapper.chmod(0o700)
    os.environ["PATH"] = f"{scratch}{os.pathsep}{os.environ.get('PATH', '')}"


def exact_frame(frame, expected, filename):
    import pandas as pd
    actual = list(frame.columns)
    if actual != expected:
        raise RuntimeError(f"{filename} columns differ from CARE: actual={len(actual)} expected={len(expected)}")
    metadata_count = len(META_FRAME if "dynamic" in filename or filename == "egemaps_lld.csv" else META_TURN)
    for column in expected[metadata_count:]:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = converted.isna() & frame[column].notna()
        if invalid.any():
            raise RuntimeError(f"{filename} has nonnumeric feature values in {column}")
    return frame


def build_disvoice_models():
    """Create fresh DisVoice objects; their legacy temp-file state is per recording."""
    import numpy as np
    if not hasattr(np, "int"):
        np.int = int
    from disvoice.articulation import Articulation
    from disvoice.phonation import Phonation
    from disvoice.prosody import Prosody
    return {"prosody": Prosody(), "phonation": Phonation(), "articulation": Articulation()}


def build_descriptor_models():
    """Load the seven descriptor objects once for a batch worker."""
    import opensmile
    import numpy as np
    if not hasattr(np, "int"):
        np.int = int
    from disvoice.articulation import Articulation
    from disvoice.phonation import Phonation
    from disvoice.prosody import Prosody
    smiles = {
        "egemaps_func.csv": opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals),
        "egemaps_lld.csv": opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.LowLevelDescriptors),
    }
    return smiles, build_disvoice_models()


def normalize_disvoice_praat_paths():
    """Normalize DisVoice's `..` temp paths before Praat receives them."""
    for module_name in ("praat.praat_functions", "disvoice.praat.praat_functions"):
        try:
            module = __import__(module_name, fromlist=["praat_vuv"])
        except ImportError:
            continue
        if getattr(module, "_carewsm_paths_normalized", False):
            continue
        original_vuv = module.praat_vuv
        original_formants = module.praat_formants

        def praat_vuv(audio, resultsp, resultst, *args, _original=original_vuv, **kwargs):
            return _original(audio, os.path.realpath(resultsp), os.path.realpath(resultst), *args, **kwargs)

        def praat_formants(audio, results, *args, _original=original_formants, **kwargs):
            return _original(audio, os.path.realpath(results), *args, **kwargs)

        module.praat_vuv = praat_vuv
        module.praat_formants = praat_formants
        module._carewsm_paths_normalized = True


def extract_row(row: dict, turns, scratch: Path, expected: dict[str, list[str]], models=None):
    import librosa
    import numpy as np
    if not hasattr(np, "int"):
        np.int = int
    import opensmile
    import pandas as pd
    import soundfile as sf
    from disvoice.articulation import Articulation
    from disvoice.phonation import Phonation
    from disvoice.prosody import Prosody
    waveform = row.get("waveform")
    if waveform is None:
        waveform, _ = librosa.load(str(row["audio_wav"]), sr=SAMPLE_RATE, mono=True)
    if models is None:
        smiles, extractors = build_descriptor_models()
    else:
        smiles, extractors = models
    rows = {name: [] for name in FILES}
    normalize_disvoice_praat_paths()
    disvoice_root = scratch / "disvoice"
    for name, extractor in extractors.items():
        module_path = disvoice_root / name / "impl"
        module_path.mkdir(parents=True, exist_ok=True)
        module = __import__(extractor.__class__.__module__, fromlist=["PATH"])
        module.PATH = str(module_path)
        if hasattr(extractor, "PATH"):
            extractor.PATH = str(module_path)
    with safe_disvoice_temp_names():
        for turn_id, (start, end) in enumerate(turns, 1):
            chunk = waveform[int(start*SAMPLE_RATE):int(end*SAMPLE_RATE)].astype(np.float32)
            if not len(chunk):
                continue
            for filename, smile in smiles.items():
                raw = smile.process_signal(chunk, SAMPLE_RATE)
                frame = raw.reset_index(drop=True)
                frame["turn_id"] = turn_id; frame["turn_start_sec"] = start; frame["turn_end_sec"] = end
                if filename == "egemaps_lld.csv":
                    frame["frame_start_sec"] = raw.index.get_level_values("start").total_seconds() + start
                    frame["frame_end_sec"] = raw.index.get_level_values("end").total_seconds() + start
                metadata = META_FRAME if filename == "egemaps_lld.csv" else META_TURN
                frame = frame[metadata + [c for c in frame.columns if c not in metadata]]
                rows[filename].append(exact_frame(frame, expected[filename], filename))
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=scratch, delete=True) as handle:
                sf.write(handle.name, chunk, SAMPLE_RATE)
                for name, extractor in extractors.items():
                    static_options = (True,) if name == "articulation" else (True, False)
                    for static in static_options:
                        kind = "static" if static else "dynamic"
                        filename = f"disvoice_{name}_{kind}.csv"
                        value = extractor.extract_features_file(handle.name, static=static, fmt="dataframe")
                        frame = value.copy()
                        if not hasattr(frame, "columns"):
                            raise RuntimeError(f"DisVoice did not return a dataframe for {filename}")
                        frame["turn_id"] = turn_id; frame["turn_start_sec"] = start; frame["turn_end_sec"] = end
                        if not static:
                            hop = {"prosody": 0.01, "phonation": 0.02}[name]
                            offset = {"prosody": 0, "phonation": 11}[name]
                            frame["frame_start_sec"] = start + (np.arange(len(frame)) + offset) * hop
                            frame["frame_end_sec"] = frame["frame_start_sec"] + hop
                        metadata = META_TURN if static else META_FRAME
                        frame = frame[metadata + [c for c in frame.columns if c not in metadata]]
                        rows[filename].append(exact_frame(frame, expected[filename], filename))
    written = {}; skipped = {}
    for filename in FILES:
        destination = row["destination"] / filename
        if validate_existing(destination, expected[filename]):
            skipped[filename] = str(destination); continue
        if not rows[filename]:
            raise RuntimeError(f"no feature rows produced for {filename}")
        frame = pd.concat(rows[filename], ignore_index=True)
        exact_frame(frame, expected[filename], filename)
        atomic_csv(frame, destination)
        written[filename] = {"path": str(destination), "bytes": destination.stat().st_size, "rows": len(frame)}
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsm-root", required=True, type=Path)
    parser.add_argument("--care-root", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "dev", "test"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--embedding-weights", type=Path)
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    result = {"task": "TASK-003A", "split": args.split, "limit": min(args.limit, MAX_FILES), "status": "blocked"}
    try:
        rows = read_rows(args.wsm_root.resolve(), args.split, args.limit)
        expected = {filename: reference_headers(args.care_root.resolve(), filename) for filename in FILES}
        result["selected"] = [{"sample_id": r["sample_id"], "stratum": r["stratum"], "video_path": r["video_path"], "source_feature_dir_ignored": r["feature_dir"], "destination": str(r["destination"])} for r in rows]
        if args.dry_run:
            result["status"] = "dry_run"; print(json.dumps(result, indent=2, sort_keys=True)); return 0
    except Exception as exc:
        result["blocker"] = {"type": "selection_or_schema_preflight", "message": str(exc)}
        print(json.dumps(result, indent=2, sort_keys=True)); return 1
    missing = [name for name in ("numpy", "pandas", "librosa", "soundfile", "opensmile", "disvoice") if importlib.util.find_spec(name) is None]
    if missing:
        result["blocker"] = {"type": "missing_local_dependency", "packages": missing, "message": "No installation or download was attempted."}
        print(json.dumps(result, indent=2, sort_keys=True)); return 1
    if shutil.which("praat") is None:
        result["blocker"] = {"type": "missing_external_dependency", "executable": "praat", "message": "DisVoice requires the Praat executable on PATH; no installation or download was attempted."}
        print(json.dumps(result, indent=2, sort_keys=True)); return 1
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        result["blocker"] = {"type": "missing_runtime_display", "message": "Praat is installed but this headless session has no DISPLAY; run under an existing display or xvfb-run."}
        print(json.dumps(result, indent=2, sort_keys=True)); return 1
    disvoice_spec = importlib.util.find_spec("disvoice")
    disvoice_root = Path(disvoice_spec.origin).parent if disvoice_spec and disvoice_spec.origin else None
    missing_scripts = [] if disvoice_root is None else [str(disvoice_root / "praat" / name) for name in ("vuv_praat.praat", "FormantsPraat.praat") if not (disvoice_root / "praat" / name).is_file()]
    if missing_scripts:
        result["blocker"] = {"type": "missing_disvoice_resources", "paths": missing_scripts, "message": "Installed DisVoice code is missing its required Praat scripts; no download was attempted."}
        print(json.dumps(result, indent=2, sort_keys=True)); return 1
    result["runs"] = []
    with tempfile.TemporaryDirectory(prefix="carewsm-task003a-") as scratch_name:
        scratch = Path(scratch_name)
        configure_headless_praat(scratch)
        for row in rows:
            started = time.perf_counter()
            try:
                row["destination"].mkdir(parents=False, exist_ok=True)
                wav = scratch / f"{row['sample_id']}.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(row["video"]), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
                row["audio_wav"] = wav
                turns, provenance = create_turns(row, scratch, args)
                written, skipped = extract_row(row, turns, scratch, expected)
                result["runs"].append({"sample_id": row["sample_id"], "status": "ok", "turn_count": len(turns), "turn_provenance": provenance, "written": written, "skipped": skipped, "elapsed_s": round(time.perf_counter()-started, 3)})
            except Exception as exc:
                result["runs"].append({"sample_id": row["sample_id"], "status": "blocked", "error": str(exc)})
        result["temporary_artifacts_cleaned"] = True
    if result["runs"] and all(item["status"] == "ok" for item in result["runs"]):
        result["status"] = "ok"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "ok" else 1

if __name__ == "__main__":
    raise SystemExit(main())
