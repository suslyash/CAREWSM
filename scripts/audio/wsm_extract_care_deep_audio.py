#!/usr/bin/env python3
"""Bounded, offline CARE-compatible extraction for the four deep audio families.

This script never calls a loader that can download weights.  Every model path is
explicit and every output is validated before a no-clobber publish.
"""
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

import numpy as np

MAX_FILES = 3
SAMPLE_RATE = 16000
STRATA = ("DEPRESSION", "CONTROL", "PARKINSON")
NPZ_FAMILIES = {
    "wav2vec_xlsr300m": "wav2vec_xlsr300m.npz",
    "trillsson4_1": "trillsson4_1.npz",
    "emotionmsp_embeds": "emotionmsp_embeds.npz",
}
TRAJ_FILE = "emotionmsp_trajectories.csv"
TRAJ_HEADER = ("turn_id", "turn_start_sec", "turn_end_sec", "arousal", "dominance", "valence")


def read_rows(root: Path, split: str, limit: int) -> list[dict]:
    with (root / "splits" / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "video_path", "split", "storage_type", "stratum", "label_id", "subgroup", "profile_id", "video_id"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise RuntimeError(f"manifest missing columns: {missing}")
        candidates = []
        for row in reader:
            if row.get("split") != split or row.get("storage_type") != "files":
                continue
            rel = row["video_path"]
            video = (root / rel).resolve()
            if Path(rel).is_absolute() or root.resolve() not in video.parents or not video.is_file():
                continue
            destination = (root / "extracted" / row["label_id"] / row["subgroup"] /
                           f"PROFILE_{row['profile_id']}" / f"VIDEO_{row['video_id']}").resolve()
            if root.resolve() not in destination.parents or not destination.parent.is_dir():
                raise RuntimeError(f"canonical destination/profile directory missing: {destination}")
            candidates.append({**row, "video": video, "destination": destination})
    candidates.sort(key=lambda row: (row["sample_id"], row["video_path"]))
    chosen = []
    used = set()
    for stratum in STRATA:
        row = next((item for item in candidates if item.get("stratum") == stratum), None)
        if row and row["sample_id"] not in used:
            chosen.append(row)
            used.add(row["sample_id"])
    return chosen[: min(MAX_FILES, limit)]


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
    labels = {ordered[0]: "PARTICIPANT"}
    labels.update({speaker: "INTERVIEWER" for speaker in ordered[1:]})
    return [(labels[speaker], start, end) for speaker, start, end in segments]


def participant_turns(segments: list[tuple[str, float, float]]) -> list[tuple[float, float]]:
    turns: list[tuple[float, float]] = []
    start = end = None
    for speaker, begin, finish in sorted(segments, key=lambda item: item[1]):
        if speaker == "INTERVIEWER":
            if start is not None:
                turns.append((start, end))
                start = end = None
            continue
        if start is None:
            start, end = begin, finish
        elif begin - end <= 3.0:
            end = finish
        else:
            turns.append((start, end)); start, end = begin, finish
    if start is not None:
        turns.append((start, end))
    return [(a, b) for a, b in turns if b - a >= 0.3]


def turns_for(row: dict, args) -> list[tuple[float, float]]:
    rttm = row["destination"] / "diarizen_diarization.rttm"
    if rttm.is_file():
        turns = participant_turns(parse_rttm(rttm))
    else:
        if not args.weights or not args.embedding_weights:
            raise RuntimeError(f"missing local RTTM: {rttm}; provide explicit local DiariZen weights for temporary turns")
        if not args.weights.is_dir() or not args.embedding_weights.is_file():
            raise RuntimeError("explicit local DiariZen weights or embedding file is missing")
        missing = package_missing(("torch", "torchaudio", "diarizen"))
        if missing:
            raise RuntimeError(f"missing local DiariZen dependencies: {missing}")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU DiariZen is prohibited")
        from diarizen.pipelines.inference import DiariZenPipeline
        scratch = Path(tempfile.mkdtemp(prefix=".deep-diarization-", dir=row["destination"]))
        try:
            wav = local_wav(row["video"], scratch)
            pipeline = DiariZenPipeline(diarizen_hub=args.weights, embedding_model=str(args.embedding_weights), rttm_out_dir=None)
            diarization = pipeline(str(wav))
            segments = []; durations = {}
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                start, end, label = float(segment.start), float(segment.end), str(speaker)
                segments.append((label, start, end)); durations[label] = durations.get(label, 0.0) + max(0.0, end - start)
            if not segments:
                raise RuntimeError("DiariZen returned no speech segments")
            ordered = sorted(durations, key=durations.get, reverse=True)
            labels = {ordered[0]: "PARTICIPANT"}
            labels.update({speaker: "INTERVIEWER" for speaker in ordered[1:]})
            turns = participant_turns([(labels[speaker], start, end) for speaker, start, end in segments])
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    if not turns:
        raise RuntimeError("no participant turns available")
    return turns


def validate_npz(path: Path, family: str, turn_count: int) -> dict:
    expected = {"wav2vec_xlsr300m": ("mean", "std", "turn_ids"),
                "trillsson4_1": ("mean", "std", "turn_ids"),
                "emotionmsp_embeds": ("embed", "turn_ids")}[family]
    with np.load(path, allow_pickle=False) as data:
        if tuple(data.files) != expected:
            raise RuntimeError(f"{path.name}: keys {data.files!r}, expected {expected!r}")
        turns = data["turn_ids"]
        if turns.dtype != np.dtype("int64") or turns.shape != (turn_count,) or not np.array_equal(turns, np.arange(1, turn_count + 1, dtype=np.int64)):
            raise RuntimeError(f"{path.name}: invalid turn_ids shape/dtype/order")
        key = "embed" if family == "emotionmsp_embeds" else "mean"
        shape = list(data[key].shape)
        for name in expected:
            value = data[name]
            if name != "turn_ids" and (value.dtype != np.dtype("float32") or value.shape != (turn_count, 1024)):
                raise RuntimeError(f"{path.name}: invalid {name} dtype/shape {value.dtype}/{value.shape}")
            if not np.isfinite(value).all():
                raise RuntimeError(f"{path.name}: nonfinite {name}")
    return {"bytes": path.stat().st_size, "shape": shape, "turns": turn_count}


def validate_trajectories(path: Path, turns: list[tuple[float, float]]) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TRAJ_HEADER:
            raise RuntimeError(f"{path.name}: header mismatch")
        rows = list(reader)
    if len(rows) != len(turns):
        raise RuntimeError(f"{path.name}: row count {len(rows)} != turns {len(turns)}")
    for index, (row, (start, end)) in enumerate(zip(rows, turns), 1):
        if int(row["turn_id"]) != index:
            raise RuntimeError(f"{path.name}: invalid turn_id at row {index}")
        for name, expected in (("turn_start_sec", start), ("turn_end_sec", end)):
            value = float(row[name])
            if not math.isfinite(value) or abs(value - expected) > 1e-5:
                raise RuntimeError(f"{path.name}: invalid {name} at row {index}")
        for name in ("arousal", "dominance", "valence"):
            if not math.isfinite(float(row[name])):
                raise RuntimeError(f"{path.name}: nonfinite {name} at row {index}")
    return {"bytes": path.stat().st_size, "rows": len(rows), "columns": list(TRAJ_HEADER)}


def no_clobber_publish(staged: dict[str, Path], destination: Path) -> None:
    for name in staged:
        if (destination / name).exists():
            raise FileExistsError(f"refusing to overwrite existing output: {destination / name}")
    linked = []
    try:
        for name, temp in staged.items():
            os.link(temp, destination / name)
            linked.append(destination / name)
    except Exception:
        for path in linked:
            path.unlink(missing_ok=True)
        raise


def package_missing(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def model_status(args) -> dict[str, dict]:
    return {
        "wav2vec_xlsr300m": {"path": str(args.xlsr_model) if args.xlsr_model else None, "missing_packages": package_missing(("torch", "torchaudio")), "missing_path": not args.xlsr_model or not args.xlsr_model.is_file()},
        "trillsson4_1": {"path": str(args.trillsson_model) if args.trillsson_model else None, "missing_packages": package_missing(("tensorflow", "tensorflow_hub")), "missing_path": not args.trillsson_model or not args.trillsson_model.exists()},
        "emotionmsp_trajectories": {"path": str(args.emotion_model) if args.emotion_model else None, "missing_packages": package_missing(("torch", "torchaudio", "transformers")), "missing_path": not args.emotion_model or not args.emotion_model.is_dir()},
        "emotionmsp_embeds": {"path": str(args.emotion_model) if args.emotion_model else None, "missing_packages": package_missing(("torch", "torchaudio", "transformers")), "missing_path": not args.emotion_model or not args.emotion_model.is_dir()},
    }


def local_wav(video: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f".{video.stem}.mono16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(output)], check=True)
    return output


def load_xlsr(checkpoint: Path, device: str):
    import torch
    import torchaudio
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
    model = torchaudio.models.wav2vec2_model(**bundle._params)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    if getattr(bundle, "_normalize_waveform", False):
        from torchaudio.pipelines._wav2vec2 import utils
        model = utils._extend_model(model, normalize_waveform=True)
    return model.eval().to(device), torch


def load_emotion(model_path: Path, device: str):
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__(); self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout); self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        def forward(self, features):
            x = self.dropout(features); x = torch.tanh(self.dense(x)); return self.out_proj(self.dropout(x))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config); self.wav2vec2 = Wav2Vec2Model(config); self.classifier = RegressionHead(config); self.post_init()
        def forward(self, input_values):
            hidden = torch.mean(self.wav2vec2(input_values).last_hidden_state, dim=1)
            return hidden, self.classifier(hidden)

    processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = EmotionModel.from_pretrained(str(model_path), local_files_only=True).eval().to(device)
    return processor, model, torch


def infer_outputs(row: dict, turns: list[tuple[float, float]], args, ready: set[str]) -> dict[str, object]:
    """Run only explicitly provisioned local stacks and return validated arrays/rows."""
    import librosa
    scratch = Path(tempfile.mkdtemp(prefix=".deep-audio-", dir=row["destination"]))
    try:
        wav = local_wav(row["video"], scratch)
        waveform, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
        output: dict[str, object] = {}
        if "wav2vec_xlsr300m" in ready:
            model, torch = load_xlsr(args.xlsr_model, args.device)
            tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(args.device)
            means, stds, ids = [], [], []
            with torch.no_grad():
                for turn_id, (start, end) in enumerate(turns, 1):
                    chunk = tensor[:, round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)]
                    feats = model.extract_features(chunk)[0][-1]
                    means.append(feats.mean(dim=1).squeeze(0).cpu().numpy())
                    stds.append(feats.std(dim=1, unbiased=False).squeeze(0).cpu().numpy()); ids.append(turn_id)
            output["wav2vec_xlsr300m.npz"] = {"mean": np.asarray(means, dtype=np.float32), "std": np.asarray(stds, dtype=np.float32), "turn_ids": np.asarray(ids, dtype=np.int64)}
        if "trillsson4_1" in ready:
            import tensorflow as tf
            import tensorflow_hub as hub
            model = hub.KerasLayer(str(args.trillsson_model), trainable=False)
            means, stds, ids = [], [], []
            for turn_id, (start, end) in enumerate(turns, 1):
                chunk = tf.convert_to_tensor(waveform[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)], dtype=tf.float32)
                feats = model(tf.expand_dims(chunk, 0))["embedding"]
                means.append(tf.reduce_mean(feats, axis=0).numpy()); stds.append(tf.math.reduce_std(feats, axis=0).numpy()); ids.append(turn_id)
            output["trillsson4_1.npz"] = {"mean": np.asarray(means, dtype=np.float32), "std": np.asarray(stds, dtype=np.float32), "turn_ids": np.asarray(ids, dtype=np.int64)}
        if "emotionmsp_trajectories" in ready or "emotionmsp_embeds" in ready:
            processor, model, torch = load_emotion(args.emotion_model, args.device)
            embeds, rows, ids = [], [], []
            with torch.no_grad():
                for turn_id, (start, end) in enumerate(turns, 1):
                    chunk = waveform[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)]
                    values = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"].to(args.device)
                    embed, trajectory = model(values)
                    embeds.append(embed.squeeze(0).cpu().numpy()); ids.append(turn_id)
                    scores = trajectory.squeeze(0).cpu().numpy()
                    rows.append((turn_id, start, end, float(scores[0]), float(scores[1]), float(scores[2])))
            output["emotionmsp_embeds.npz"] = {"embed": np.asarray(embeds, dtype=np.float32), "turn_ids": np.asarray(ids, dtype=np.int64)}
            output[TRAJ_FILE] = rows
        return output
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def stage_outputs(outputs: dict[str, object], destination: Path, turns: list[tuple[float, float]]) -> dict[str, dict]:
    staged = {}
    temp_dir = Path(tempfile.mkdtemp(prefix=".deep-stage-", dir=destination))
    try:
        for name, value in outputs.items():
            path = temp_dir / name
            if name.endswith(".npz"):
                np.savez(path, **value)
                validate_npz(path, next(key for key, filename in NPZ_FAMILIES.items() if filename == name), len(turns))
            else:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle); writer.writerow(TRAJ_HEADER); writer.writerows(value)
                validate_trajectories(path, turns)
            staged[name] = path
        no_clobber_publish(staged, destination)
        return {name: {"status": "created", "bytes": (destination / name).stat().st_size} for name in staged}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsm-root", required=True, type=Path)
    parser.add_argument("--care-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--xlsr-model", type=Path, help="local torchaudio XLS-R checkpoint; never downloaded")
    parser.add_argument("--trillsson-model", type=Path, help="local TensorFlow Hub trillsson4/1 SavedModel; never downloaded")
    parser.add_argument("--emotion-model", type=Path, help="local audeering Emotion MSP HF snapshot; never downloaded")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", type=Path, help="explicit local DiariZen snapshot; never downloaded")
    parser.add_argument("--embedding-weights", type=Path, help="explicit local DiariZen embedding file; never downloaded")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    selected = read_rows(args.wsm_root, args.split, args.limit)
    statuses = model_status(args)
    result = {"task": "TASK-003B", "status": "ok", "split": args.split, "limit": min(MAX_FILES, args.limit), "selected": [], "families": {}, "downloads": False}
    for family, status in statuses.items():
        if status["missing_packages"] or status["missing_path"]:
            result["families"][family] = {"status": "blocked", **status}
        else:
            result["families"][family] = {"status": "ready", **status}
    for row in selected:
        sample = {"sample_id": row["sample_id"], "destination": str(row["destination"]), "files": {}}
        try:
            turns = turns_for(row, args)
            sample["turn_count"] = len(turns)
        except Exception as exc:
            sample["blocker"] = {"type": "missing_local_rttm", "message": str(exc)}
            result["status"] = "blocked"
            result["selected"].append(sample)
            continue
        for name in (*NPZ_FAMILIES.values(), TRAJ_FILE):
            path = row["destination"] / name
            if path.is_file():
                try:
                    check = validate_trajectories(path, turns) if name == TRAJ_FILE else validate_npz(path, next(key for key, value in NPZ_FAMILIES.items() if value == name), len(turns))
                    sample["files"][name] = {"status": "valid_existing", **check}
                except Exception as exc:
                    sample["files"][name] = {"status": "blocked_existing_invalid", "message": str(exc)}
                    result["status"] = "blocked"
            else:
                sample["files"][name] = {"status": "missing"}
        if not args.dry_run:
            ready = {family for family, status in result["families"].items() if status.get("status") == "ready"}
            missing_outputs = {name for name, info in sample["files"].items() if info["status"] == "missing"}
            if ready and missing_outputs:
                started = time.perf_counter()
                try:
                    outputs = infer_outputs(row, turns, args, ready)
                    outputs = {name: value for name, value in outputs.items() if name in missing_outputs}
                    sample["files"].update(stage_outputs(outputs, row["destination"], turns))
                    sample["elapsed_s"] = round(time.perf_counter() - started, 3)
                    sample["device"] = args.device
                except Exception as exc:
                    result["status"] = "blocked"
                    sample["blocker"] = {"type": "local_inference_failure", "message": str(exc)}
            elif missing_outputs:
                result["status"] = "blocked"
                sample["blocker"] = {"type": "missing_local_models_or_dependencies", "message": "No eligible local model stack was available; no inference was attempted."}
        result["selected"].append(sample)
    if any(item.get("status") == "blocked" for item in result["families"].values()):
        result["status"] = "blocked"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
