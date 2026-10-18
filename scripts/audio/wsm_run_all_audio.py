#!/usr/bin/env python3
"""TASK-004A: one-pass full-VIDEO WSM audio extraction, without DiariZen."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import wsm_extract_care_descriptors as descriptors

AUDIO = Path(__file__).resolve().parent
RESET = AUDIO / "wsm_reset_audio_outputs.py"
SPLITS = ("train", "dev", "test")
FILES = (
    "egemaps_func.csv", "egemaps_lld.csv", "disvoice_prosody_static.csv",
    "disvoice_prosody_dynamic.csv", "disvoice_phonation_static.csv",
    "disvoice_phonation_dynamic.csv", "disvoice_articulation_static.csv",
    "wav2vec_xlsr300m.npz", "trillsson4_1.npz",
    "emotionmsp_trajectories.csv", "emotionmsp_embeds.npz",
)
STOP = False


def stop(_signum, _frame):
    global STOP
    STOP = True


def safe(value: str, field: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or value in {".", ".."} or path.name != value or ".." in path.parts:
        raise RuntimeError(f"unsafe manifest {field}: {value!r}")


def rows_for(root: Path, split: str) -> tuple[list[dict], str]:
    manifest = root / "splits" / f"{split}.csv"
    raw = manifest.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "video_path", "split", "storage_type", "label_id", "subgroup", "profile_id", "video_id", "stratum"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing: raise RuntimeError(f"{manifest}: missing columns {missing}")
        rows = []
        for row in reader:
            if row.get("split") != split or row.get("storage_type") != "files": continue
            safe(row["sample_id"], "sample_id")
            rel = row.get("video_path", "")
            if not rel or Path(rel).is_absolute(): raise RuntimeError(f"unsafe video_path: {rel!r}")
            video = (root / rel).resolve()
            if root not in video.parents: raise RuntimeError(f"video escapes WSM root: {row['sample_id']}")
            for field in ("label_id", "subgroup", "profile_id", "video_id", "stratum"): safe(row.get(field, ""), field)
            destination = (root / "extracted" / row["label_id"] / row["subgroup"] / f"PROFILE_{row['profile_id']}" / f"VIDEO_{row['video_id']}").resolve()
            if root not in destination.parents or destination.is_symlink(): raise RuntimeError(f"unsafe destination: {destination}")
            rows.append({**row, "video": video, "destination": destination})
    rows.sort(key=lambda item: (item["sample_id"], item["video_path"]))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)): raise RuntimeError(f"duplicate sample_id in {manifest}")
    return rows, digest


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def decode(video: Path, scratch: Path):
    import soundfile as sf
    wav = scratch / "full.mono16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:a:0", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
    waveform, rate = sf.read(wav, dtype="float32", always_2d=False)
    if rate != 16000 or getattr(waveform, "ndim", 1) != 1 or len(waveform) == 0:
        raise RuntimeError(f"decoded audio is not nonempty mono 16 kHz: rate={rate}, shape={getattr(waveform, 'shape', None)}")
    return waveform, len(waveform) / 16000.0


def configure_headless_praat(runtime: Path) -> None:
    """Make DisVoice's hard-coded `praat` call use the non-GUI binary."""
    candidate = shutil.which("praat_nogui") or "/usr/bin/praat_nogui"
    if not Path(candidate).is_file():
        raise RuntimeError("missing local praat_nogui executable")
    bindir = runtime / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    wrapper = bindir / "praat"
    wrapper.write_text(f"#!/bin/sh\nexec {candidate} --run \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")


def model_preflight(args) -> list[str]:
    errors = []
    for executable in ("ffmpeg", "ffprobe", "praat"):
        if shutil.which(executable) is None: errors.append(f"missing executable: {executable}")
    if shutil.which("xvfb-run") is None and not os.environ.get("DISPLAY"): errors.append("missing headless display: xvfb-run or DISPLAY")
    for package in ("numpy", "pandas", "librosa", "soundfile", "opensmile", "disvoice", "torch", "torchaudio", "tensorflow", "tensorflow_hub", "transformers"):
        if importlib.util.find_spec(package) is None: errors.append(f"missing local package: {package}")
    for label, path, kind in (("XLS-R", args.xlsr_model, "file"), ("TRILLsson", args.trillsson_model, "path"), ("Emotion MSP", args.emotion_model, "dir")):
        if (kind == "file" and not path.is_file()) or (kind == "dir" and not path.is_dir()) or (kind == "path" and not path.exists()): errors.append(f"missing local {label} model: {path}")
    try:
        import torch
        if not torch.cuda.is_available(): errors.append("CUDA unavailable; CPU PyTorch is prohibited")
    except Exception as exc: errors.append(f"CUDA check failed: {exc}")
    return errors


def install_models(args):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_hub as hub
    import wsm_extract_care_deep_audio as deep
    import wsm_extract_care_descriptors as descriptors
    xlsr = deep.load_xlsr(args.xlsr_model, args.device)
    trill = (hub.KerasLayer(str(args.trillsson_model), trainable=False), tf)
    emotion = deep.load_emotion(args.emotion_model, args.device)
    smiles, disvoice = descriptors.build_descriptor_models()
    return {"xlsr": xlsr, "trill": trill, "emotion": emotion, "descriptors": (smiles, disvoice)}


def expected_headers(care_root: Path) -> dict[str, list[str]]:
    import wsm_extract_care_descriptors as descriptors
    from wsm_extract_care_deep_audio import TRAJ_HEADER
    return {**{name: descriptors.reference_headers(care_root, name) for name in descriptors.FILES}, "emotionmsp_trajectories.csv": list(TRAJ_HEADER)}


def inspect_existing(destination: Path, expected: dict[str, list[str]], duration: float) -> tuple[bool, list[str], dict[str, str]]:
    import numpy as np
    import validate_care_audio_parity as validator
    errors = []; statuses = {}; present = 0
    for name in FILES:
        path = destination / name
        if not path.is_file(): statuses[name] = "missing"; continue
        present += 1
        try:
            if name.endswith(".npz"):
                check = validator.validate_npz(path, name)
                with np.load(path, allow_pickle=False) as data:
                    if not np.array_equal(data["turn_ids"], np.array([1], dtype=np.int64)): raise RuntimeError("turn_ids must be [1]")
            else:
                check = validator.validate_csv(path, expected[name])
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        if int(float(row["turn_id"])) != 1 or abs(float(row["turn_start_sec"])) > 1e-5 or abs(float(row["turn_end_sec"]) - duration) > 1e-2:
                            raise RuntimeError("not a single full-clip turn")
            if check.get("status") != "ok": raise RuntimeError(str(check))
            statuses[name] = "valid_existing"
        except Exception as exc:
            statuses[name] = "invalid_existing"; errors.append(f"{name}: {exc}")
    return present == len(FILES) and not errors, errors, statuses


def deep_outputs(waveform, duration, models, missing: set[str], args):
    import numpy as np
    output = {}
    xlsr_model, torch = models["xlsr"]
    tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(args.device)
    if "wav2vec_xlsr300m.npz" in missing:
        with torch.no_grad(): feats = xlsr_model.extract_features(tensor)[0][-1]
        output["wav2vec_xlsr300m.npz"] = {"mean": feats.mean(dim=1).cpu().numpy().astype("float32"), "std": feats.std(dim=1, unbiased=False).cpu().numpy().astype("float32"), "turn_ids": np.array([1], dtype=np.int64)}
    if "trillsson4_1.npz" in missing:
        model, tf = models["trill"]
        feats = model(tf.expand_dims(tf.convert_to_tensor(waveform, dtype=tf.float32), 0))["embedding"]
        output["trillsson4_1.npz"] = {"mean": tf.reduce_mean(feats, axis=0).numpy().astype("float32"), "std": tf.math.reduce_std(feats, axis=0).numpy().astype("float32"), "turn_ids": np.array([1], dtype=np.int64)}
    if {"emotionmsp_embeds.npz", "emotionmsp_trajectories.csv"} & missing:
        processor, model, torch = models["emotion"]
        with torch.no_grad(): embed, trajectory = model(processor(waveform, sampling_rate=16000, return_tensors="pt")["input_values"].to(args.device))
        score = trajectory.squeeze(0).cpu().numpy()
        output["emotionmsp_embeds.npz"] = {"embed": embed.cpu().numpy().astype("float32"), "turn_ids": np.array([1], dtype=np.int64)}
        output["emotionmsp_trajectories.csv"] = [(1, 0.0, duration, float(score[0]), float(score[1]), float(score[2]))]
    return output


def publish_deep(outputs, destination: Path, duration: float) -> None:
    import numpy as np
    import validate_care_audio_parity as validator
    from wsm_extract_care_deep_audio import TRAJ_HEADER
    temp = Path(tempfile.mkdtemp(prefix=".full-segment-stage-", dir=destination))
    try:
        for name, value in outputs.items():
            path = temp / name
            if name.endswith(".npz"):
                np.savez(path, **value); check = validator.validate_npz(path, name)
            else:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle); writer.writerow(TRAJ_HEADER); writer.writerows(value)
                check = validator.validate_csv(path, list(TRAJ_HEADER))
            if check.get("status") != "ok": raise RuntimeError(f"invalid staged {name}: {check}")
        for name in outputs:
            target = destination / name
            if target.exists(): raise FileExistsError(f"refusing overwrite: {target}")
        for name in outputs: os.link(temp / name, destination / name)
    finally: shutil.rmtree(temp, ignore_errors=True)


def process(row, args, headers, models):
    started = time.perf_counter(); row["destination"].mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{row['sample_id']}-", dir=args.runtime / "scratch") as scratch_name:
        waveform, duration = decode(row["video"], Path(scratch_name))
        complete, errors, statuses = inspect_existing(row["destination"], headers, duration)
        if complete: return {"sample_id": row["sample_id"], "status": "already_complete", "duration_s": duration, "files": len(FILES), "elapsed_s": round(time.perf_counter()-started, 3)}
        if errors: return {"sample_id": row["sample_id"], "status": "invalid_existing", "errors": errors, "statuses": statuses}
        turns = [(0.0, duration)]
        row_for_descriptor = {**row, "waveform": waveform}
        descriptor_models = models["descriptors"]
        written, skipped = descriptors.extract_row(row_for_descriptor, turns, Path(scratch_name), headers, models=descriptor_models)
        missing = {name for name in FILES if not (row["destination"] / name).is_file()}
        if missing:
            outputs = deep_outputs(waveform, duration, models, missing, args)
            publish_deep({name: value for name, value in outputs.items() if name in missing}, row["destination"], duration)
        complete, errors, _ = inspect_existing(row["destination"], headers, duration)
        if not complete: return {"sample_id": row["sample_id"], "status": "blocked", "errors": errors, "written": written, "skipped": skipped}
        return {"sample_id": row["sample_id"], "status": "complete", "duration_s": duration, "files": len(FILES), "elapsed_s": round(time.perf_counter()-started, 3), "written": written, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsm-root", type=Path, required=True); parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--xlsr-model", type=Path, required=True); parser.add_argument("--trillsson-model", type=Path, required=True); parser.add_argument("--emotion-model", type=Path, required=True)
    parser.add_argument("--production", action="store_true"); parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--reset-audio", action="store_true"); parser.add_argument("--confirm-replace-diarizen-outputs", action="store_true")
    parser.add_argument("--preflight", action="store_true"); parser.add_argument("--preflight-limit", type=int, default=3)
    args = parser.parse_args(); args.wsm_root=args.wsm_root.resolve(); args.care_root=args.care_root.resolve(); args.runtime=args.runtime.resolve()
    if args.runtime == args.wsm_root or args.runtime in args.wsm_root.parents: raise SystemExit("runtime must be outside WSM root")
    os.environ.update({"HF_HUB_OFFLINE":"1", "HF_DATASETS_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"})
    if args.preflight or args.production:
        configure_headless_praat(args.runtime)
    errors = model_preflight(args)
    if args.preflight:
        rows, digest = rows_for(args.wsm_root, "train"); selected=rows[:args.preflight_limit]
        result={"task":"TASK-004A", "status":"blocked" if errors else "ready", "mode":"read_only_train_preflight", "manifest_sha256":digest, "row_count":len(rows), "selection":[{"position":i,"sample_id":r["sample_id"],"video":str(r["video"]),"video_exists":r["video"].is_file(),"destination":str(r["destination"])} for i,r in enumerate(selected)], "errors":errors, "diarizen":"not used"}
        print(json.dumps(result, indent=2, sort_keys=True)); return 0 if not errors else 1
    if not args.production or not args.all_splits or not args.reset_audio or not args.confirm_replace_diarizen_outputs:
        raise SystemExit("owner run requires --production --all-splits --reset-audio --confirm-replace-diarizen-outputs")
    if os.environ.get("CAREWSM_ALLOW_PRODUCTION") != "1":
        raise SystemExit("production is fail-closed; set CAREWSM_ALLOW_PRODUCTION=1 in the owner command")
    if errors: print(json.dumps({"status":"blocked","errors":errors}, indent=2)); return 1
    args.runtime.mkdir(parents=True, exist_ok=True); (args.runtime / "scratch").mkdir(exist_ok=True)
    reset = subprocess.run([sys.executable, str(RESET), "--wsm-root", str(args.wsm_root), "--runtime", str(args.runtime), "--delete-audio-only", "--confirm-replace-diarizen-outputs"], check=False)
    if reset.returncode != 0: return reset.returncode
    headers = expected_headers(args.care_root); models = install_models(args); summaries={}
    lock_path=args.runtime / "full_segment.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            import fcntl; fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError): return 2
        for split in SPLITS:
            rows,digest=rows_for(args.wsm_root,split); state_path=args.runtime / "state" / f"{split}.json"
            state=json.loads(state_path.read_text()) if state_path.exists() else {"manifest_sha256":digest,"row_count":len(rows),"next_index":0,"records":{}}
            if state.get("manifest_sha256") != digest or state.get("row_count") != len(rows): raise SystemExit(f"manifest changed: {split}")
            for index in range(state["next_index"], len(rows)):
                if STOP: break
                row=rows[index]
                try: record=process(row,args,headers,models)
                except Exception as exc: record={"sample_id":row["sample_id"],"status":"blocked","error":str(exc)}
                record["position"]=index; state["records"][row["sample_id"]]=record; state["next_index"]=index+1; state["high_water_sample_id"]=row["sample_id"]
                atomic_json(state_path,state); summaries[split]=state; atomic_json(args.runtime / "summary.json",summaries)
                print(json.dumps({"split":split,"position":index,"status":record["status"],"sample_id":row["sample_id"]}), flush=True)
            summaries[split]=state; atomic_json(args.runtime / "summary.json",summaries)
    return 1 if any(record.get("status") not in {"complete","already_complete"} for state in summaries.values() for record in state.get("records",{}).values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
