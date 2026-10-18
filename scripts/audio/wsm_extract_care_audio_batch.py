#!/usr/bin/env python3
"""Offline, resumable WSM extraction of the eleven owned audio families."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import signal
import sys
import tempfile
import time
from typing import Any

import numpy as np

AUDIO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIO_DIR))
import wsm_extract_care_descriptors as descriptors  # noqa: E402
import wsm_extract_care_deep_audio as deep  # noqa: E402
import validate_care_audio_parity as validator  # noqa: E402

SAMPLE_RATE = 16000
MAX_PILOT = 30
GLOBAL_CAP = 30
STRATA = ("DEPRESSION", "CONTROL", "PARKINSON")
ALL_FILES = (
    *descriptors.FILES,
    "wav2vec_xlsr300m.npz",
    "trillsson4_1.npz",
    "emotionmsp_trajectories.csv",
    "emotionmsp_embeds.npz",
)
DIARIZATION_POLICY = "CARE longest-speaker provisional participant; max_gap=3.0; min_turn_duration=0.3"
DIARIZATION_CONFIG_ID = "diarizen-wavlm-large-s80-md-v1-seg16-step0.1-vbx-ahc0.6"
KNOWN_BLOCKED_SAMPLE_IDS = (
    "DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_014",
    "DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_015",
    "DEPRESSION__DEPRESSION__0FbNUlG6xpg__0FbNUlG6xpg_001",
    "CONTROL__DEPRESSION__0xj8RWesq5c__0xj8RWesq5c_005",
    "PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_003",
    "PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_006",
    "PARKINSON__PARKINSON__06K1GIoQ7sE__06K1GIoQ7sE_001",
)
ORIGINAL_ACCEPTED_SAMPLE_IDS = {
    "DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_001",
    "CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_001",
    "PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_001",
}
STOP_REQUESTED = False


def on_signal(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def safe_component(value: str, field: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or value in {".", ".."} or path.name != value or ".." in path.parts:
        raise RuntimeError(f"unsafe manifest {field}: {value!r}")


def load_rows(root: Path, split: str) -> list[dict]:
    manifest = root / "splits" / f"{split}.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "video_path", "split", "storage_type", "stratum", "label_id", "subgroup", "profile_id", "video_id"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise RuntimeError(f"manifest missing columns: {missing}")
        rows = []
        for row in reader:
            if row.get("split") != split or row.get("storage_type") != "files":
                continue
            for field in ("sample_id", "label_id", "subgroup", "profile_id", "video_id", "stratum"):
                safe_component(row.get(field, ""), field)
            rel = row.get("video_path", "")
            video = (root / rel).resolve()
            if Path(rel).is_absolute() or root not in video.parents or not video.is_file() or (video.is_symlink() and root not in video.parents):
                continue
            destination = (root / "extracted" / row["label_id"] / row["subgroup"] / f"PROFILE_{row['profile_id']}" / f"VIDEO_{row['video_id']}").resolve()
            if root not in destination.parents or destination.is_symlink():
                raise RuntimeError(f"unsafe canonical destination: {destination}")
            rows.append({**row, "video": video, "destination": destination})
    rows.sort(key=lambda item: (item["sample_id"], item["video_path"]))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample_id in manifest")
    destinations = [str(row["destination"]) for row in rows]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("duplicate canonical destination in manifest")
    return rows


def read_turn_metadata(path: Path) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            turn = int(float(row["turn_id"]))
            times = (float(row["turn_start_sec"]), float(row["turn_end_sec"]))
            if turn in result and any(abs(a - b) > 1e-5 for a, b in zip(result[turn], times)):
                raise RuntimeError(f"inconsistent timing in {path}")
            result[turn] = times
    return result


def read_sidecar(row: dict, args) -> tuple[list[tuple[float, float]], dict] | None:
    path = row["destination"] / "_carewsm_audio_turns.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        stat = row["video"].stat()
        required = {"sample_id": row["sample_id"], "video_path": row["video_path"], "video_size": stat.st_size, "video_mtime_ns": stat.st_mtime_ns}
        if any(data.get(k) != value for k, value in required.items()):
            return None
        if data.get("weights") != str(args.weights) or data.get("embedding_weights") != str(args.embedding_weights):
            return None
        if data.get("policy") != DIARIZATION_POLICY:
            return None
        turns = [(float(a), float(b)) for a, b in data["turns"]]
        if not turns or any(b - a < 0.3 for a, b in turns):
            return None
        return turns, data
    except Exception:
        return None


def npz_turn_ids(path: Path) -> list[int]:
    with np.load(path, allow_pickle=False) as data:
        return [int(value) for value in data["turn_ids"].tolist()]


def inspect_record(destination: Path, expected: dict[str, list[str]], sidecar_turns: list[tuple[float, float]] | None = None) -> dict:
    statuses: dict[str, str] = {}
    errors: list[str] = []
    csv_maps: list[dict[int, tuple[float, float]]] = []
    npz_ids: dict[str, list[int]] = {}
    for filename in ALL_FILES:
        path = destination / filename
        if not path.is_file():
            statuses[filename] = "missing"
            continue
        try:
            if filename in descriptors.FILES or filename == "emotionmsp_trajectories.csv":
                checked = validator.validate_csv(path, expected[filename])
                if checked["status"] != "ok":
                    raise RuntimeError(str(checked))
                csv_maps.append(read_turn_metadata(path))
            else:
                checked = validator.validate_npz(path, filename)
                if checked["status"] != "ok":
                    raise RuntimeError(str(checked))
                npz_ids[filename] = npz_turn_ids(path)
            statuses[filename] = "valid_existing"
        except Exception as exc:
            statuses[filename] = "blocked_existing_invalid"
            errors.append(f"{filename}: {exc}")
    maps = [item for item in csv_maps if item]
    if maps:
        first = maps[0]
        for item in maps[1:]:
            if item != first:
                errors.append("cross-family CSV turn metadata mismatch")
        expected_ids = list(first)
        for filename, ids in npz_ids.items():
            if ids != expected_ids:
                errors.append(f"{filename}: NPZ turn_ids do not match CSV turn_ids")
        if sidecar_turns is not None:
            sidecar_map = {index: value for index, value in enumerate(sidecar_turns, 1)}
            if sidecar_map != first:
                errors.append("cached sidecar turn boundaries do not match feature metadata")
    elif npz_ids:
        expected_ids = next(iter(npz_ids.values()))
        for filename, ids in npz_ids.items():
            if ids != expected_ids:
                errors.append(f"{filename}: NPZ turn_ids mismatch")
    complete = all(statuses.get(name) == "valid_existing" for name in ALL_FILES) and not errors
    return {"statuses": statuses, "errors": errors, "complete": complete, "turn_ids": list(maps[0]) if maps else (next(iter(npz_ids.values())) if npz_ids else [])}


def stable_selection(rows: list[dict], args) -> list[dict]:
    ordered = rows
    if args.start_after:
        ordered = [row for row in ordered if row["sample_id"] > args.start_after]
    if args.per_stratum and (args.offset or args.start_after):
        raise RuntimeError("--per-stratum cannot be combined with --offset/--start-after; use immutable manifest windows")
    if args.per_stratum:
        quota = min(args.per_stratum, args.limit)
        selected = []
        for stratum in STRATA:
            selected.extend([row for row in ordered if row["stratum"] == stratum][:quota])
        used = {row["sample_id"] for row in selected}
        selected.extend(row for row in ordered if row["sample_id"] not in used)
        return selected[:args.limit]
    # Pagination is applied to immutable manifest positions before file filtering.
    return ordered[args.offset:args.offset + args.limit]


def load_allowlist(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(ids) != len(set(ids)) or set(ids) != set(KNOWN_BLOCKED_SAMPLE_IDS):
        raise RuntimeError("sample-id allowlist must exactly equal the seven manager-approved blocked IDs")
    return list(KNOWN_BLOCKED_SAMPLE_IDS)


def budget_receipt(path: Path, rows: list[dict], dry_run: bool, retry: bool) -> dict:
    complete = [row["sample_id"] for row in rows if row.get("existing_state", {}).get("complete")]
    new_complete = sorted(set(complete) - ORIGINAL_ACCEPTED_SAMPLE_IDS)
    scope_state = "within_cap" if len(new_complete) <= GLOBAL_CAP else "grandfathered_scope_violation_frozen"
    if len(new_complete) > GLOBAL_CAP and not retry:
        raise RuntimeError(f"persistent TASK-003C cap exceeded: {len(new_complete)} new complete records > {GLOBAL_CAP}; only the manager-approved seven-ID retry is allowed")
    receipt = {"task": "TASK-003C", "global_cap": GLOBAL_CAP, "new_complete_sample_ids": new_complete, "grandfathered_complete_sample_ids": sorted(ORIGINAL_ACCEPTED_SAMPLE_IDS), "blocked_allowlist": list(KNOWN_BLOCKED_SAMPLE_IDS), "scope_state": scope_state, "frozen": True, "retry_used": False, "retry_results": {}}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if retry and old.get("retry_used"):
            raise RuntimeError("the manager-approved seven-ID retry has already been used; refusing a second retry")
        for key in ("task", "global_cap", "new_complete_sample_ids", "grandfathered_complete_sample_ids", "blocked_allowlist", "scope_state", "frozen"):
            if old.get(key) != receipt[key]:
                raise RuntimeError("budget ledger does not match read-only corpus audit; refusing to proceed")
        receipt.update({"retry_used": bool(old.get("retry_used", False)), "retry_results": old.get("retry_results", {})})
    elif not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "state": "validated_existing" if path.exists() else "derived_read_only", "new_complete_count": len(new_complete), "global_cap": GLOBAL_CAP}


def write_sidecar(row: dict, turns: list[tuple[float, float]], args) -> None:
    destination = row["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "_carewsm_audio_turns.json"
    payload = {"sample_id": row["sample_id"], "video_path": row["video_path"], "video_size": row["video"].stat().st_size, "video_mtime_ns": row["video"].stat().st_mtime_ns, "weights": str(args.weights), "embedding_weights": str(args.embedding_weights), "policy": DIARIZATION_POLICY, "config_id": DIARIZATION_CONFIG_ID, "turns": [[float(a), float(b)] for a, b in turns]}
    if path.exists():
        return
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination, prefix="._carewsm_audio_turns.", suffix=".partial", delete=False) as handle:
        temp = Path(handle.name); json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def decode_audio(row: dict, scratch: Path) -> Path:
    import subprocess
    wav = scratch / f"{row['sample_id']}.mono16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(row["video"]), "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
    return wav


def diarizer_pipeline(args):
    if not args.weights or not args.embedding_weights or not args.weights.is_dir() or not args.embedding_weights.is_file():
        raise RuntimeError("explicit local DiariZen weights and embedding weights are required")
    missing = deep.package_missing(("torch", "torchaudio", "diarizen"))
    if missing:
        raise RuntimeError(f"missing local DiariZen dependencies: {missing}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU DiariZen is prohibited")
    from diarizen.pipelines.inference import DiariZenPipeline
    return DiariZenPipeline(
        diarizen_hub=args.weights, embedding_model=str(args.embedding_weights),
        config_parse={"model": {"args": {"wavlm_src": "wavlm_large_s80_md", "wavlm_layer_num": 25, "wavlm_feat_dim": 1024, "attention_in": 256, "ffn_hidden": 1024, "num_head": 4, "num_layer": 4, "dropout": 0.1, "chunk_size": 16, "use_posi": False, "output_activate_function": False, "selected_channel": 0}}, "inference": {"args": {"seg_duration": 16, "segmentation_step": 0.1, "batch_size": 1, "apply_median_filtering": True}}, "clustering": {"args": {"method": "VBxClustering", "min_speakers": 1, "max_speakers": 20, "ahc_criterion": "distance", "ahc_threshold": 0.6, "Fa": 0.07, "Fb": 0.8, "lda_dim": 128, "max_iters": 20}}}, rttm_out_dir=None)


def diarize_once(wav: Path, row: dict, args, pipeline, praat_log: Path) -> tuple[list[tuple[float, float]], dict]:
    cached = read_sidecar(row, args)
    if cached is not None:
        return cached[0], {"source": "validated sidecar", "diarization_runs": 0}
    if pipeline is None:
        pipeline = diarizer_pipeline(args)
    diarization = pipeline(str(wav)); segments = []; durations = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start, end, label = float(segment.start), float(segment.end), str(speaker)
        segments.append((label, start, end)); durations[label] = durations.get(label, 0.0) + max(0.0, end - start)
    if not segments:
        raise RuntimeError("DiariZen returned no speech segments")
    ordered = sorted(durations, key=durations.get, reverse=True)
    labels = {ordered[0]: "PARTICIPANT"}; labels.update({speaker: "INTERVIEWER" for speaker in ordered[1:]})
    turns = deep.participant_turns([(labels[speaker], start, end) for speaker, start, end in segments])
    if not turns:
        raise RuntimeError("no participant turns after CARE policy")
    write_sidecar(row, turns, args)
    return turns, {"source": "local DiariZen", "diarization_runs": 1, "speaker_count": len(durations), "segment_count": len(segments), "praat_log": str(praat_log)}


def load_deep_models(args) -> dict[str, Any]:
    models = {}
    if args.xlsr_model and args.xlsr_model.is_file():
        models["xlsr"] = deep.load_xlsr(args.xlsr_model, args.device)
    if args.trillsson_model and args.trillsson_model.exists():
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        import tensorflow_hub as hub
        models["trill"] = (hub.KerasLayer(str(args.trillsson_model), trainable=False), tf)
    if args.emotion_model and args.emotion_model.is_dir():
        models["emotion"] = deep.load_emotion(args.emotion_model, args.device)
    return models


def configure_praat_wrapper(scratch: Path, log: Path) -> None:
    real = shutil.which("praat")
    if real is None:
        raise RuntimeError("Praat executable is unavailable")
    wrapper = scratch / "praat"
    wrapper.write_text("#!/bin/sh\nlog=" + shlex.quote(str(log)) + "\nprintf '%s\\n' \"PRAAT_START $*\" >> \"$log\"\ntimeout --kill-after=5 90 " + shlex.quote(real) + " --run \"$@\" >> \"$log\" 2>&1\nrc=$?\nprintf '%s\\n' \"PRAAT_EXIT $rc\" >> \"$log\"\nexit $rc\n", encoding="utf-8")
    wrapper.chmod(0o700)
    os.environ["PATH"] = f"{scratch}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["CAREWSM_PRAAT_LOG"] = str(log)


def diagnostic_tail(path: Path) -> str:
    if not path.is_file():
        return "Praat log was not created"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return " | ".join(lines[-8:])[-2000:]


def deep_outputs(wav: Path, turns, missing: set[str], models: dict[str, Any], device: str) -> dict[str, object]:
    import librosa
    waveform, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True); output = {}
    if "wav2vec_xlsr300m.npz" in missing:
        model, torch = models["xlsr"]; tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(device); means=[]; stds=[]; ids=[]
        with torch.no_grad():
            for turn_id, (start, end) in enumerate(turns, 1):
                feats = model.extract_features(tensor[:, round(start*SAMPLE_RATE):round(end*SAMPLE_RATE)])[0][-1]
                means.append(feats.mean(dim=1).squeeze(0).cpu().numpy()); stds.append(feats.std(dim=1, unbiased=False).squeeze(0).cpu().numpy()); ids.append(turn_id)
        output["wav2vec_xlsr300m.npz"] = {"mean": np.asarray(means, dtype=np.float32), "std": np.asarray(stds, dtype=np.float32), "turn_ids": np.asarray(ids, dtype=np.int64)}
    if "trillsson4_1.npz" in missing:
        model, tf = models["trill"]; means=[]; stds=[]; ids=[]
        for turn_id, (start, end) in enumerate(turns, 1):
            chunk=tf.convert_to_tensor(waveform[round(start*SAMPLE_RATE):round(end*SAMPLE_RATE)], dtype=tf.float32); feats=model(tf.expand_dims(chunk,0))["embedding"]
            means.append(tf.reduce_mean(feats,axis=0).numpy()); stds.append(tf.math.reduce_std(feats,axis=0).numpy()); ids.append(turn_id)
        output["trillsson4_1.npz"]={"mean":np.asarray(means,dtype=np.float32),"std":np.asarray(stds,dtype=np.float32),"turn_ids":np.asarray(ids,dtype=np.int64)}
    if {"emotionmsp_embeds.npz", "emotionmsp_trajectories.csv"} & missing:
        processor, model, torch = models["emotion"]; embeds=[]; rows=[]; ids=[]
        with torch.no_grad():
            for turn_id,(start,end) in enumerate(turns,1):
                values=processor(waveform[round(start*SAMPLE_RATE):round(end*SAMPLE_RATE)],sampling_rate=SAMPLE_RATE,return_tensors="pt")["input_values"].to(device); embed, trajectory=model(values); embeds.append(embed.squeeze(0).cpu().numpy()); ids.append(turn_id); score=trajectory.squeeze(0).cpu().numpy(); rows.append((turn_id,start,end,float(score[0]),float(score[1]),float(score[2])))
        output["emotionmsp_embeds.npz"]={"embed":np.asarray(embeds,dtype=np.float32),"turn_ids":np.asarray(ids,dtype=np.int64)}; output["emotionmsp_trajectories.csv"]=rows
    return {name:value for name,value in output.items() if name in missing}


def stage_deep(outputs: dict[str, object], destination: Path) -> dict:
    staged={}; temp_dir=Path(tempfile.mkdtemp(prefix=".carewsm-deep-stage-",dir=destination))
    try:
        for name,value in outputs.items():
            path=temp_dir/name
            if name.endswith(".npz"):
                np.savez(path,**value); checked=validator.validate_npz(path,name)
            else:
                with path.open("w",newline="",encoding="utf-8") as handle: writer=csv.writer(handle); writer.writerow(deep.TRAJ_HEADER); writer.writerows(value)
                checked=validator.validate_csv(path,list(deep.TRAJ_HEADER))
            if checked["status"] != "ok": raise RuntimeError(f"invalid staged {name}: {checked}")
            staged[name]=path
        for name in staged:
            if (destination/name).exists(): raise FileExistsError(f"refusing overwrite: {destination/name}")
        linked=[]
        try:
            for name,path in staged.items(): os.link(path,destination/name); linked.append(destination/name)
        except Exception:
            for path in linked: path.unlink(missing_ok=True)
            raise
        return {name:{"status":"created","bytes":(destination/name).stat().st_size} for name in staged}
    finally: shutil.rmtree(temp_dir,ignore_errors=True)


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--wsm-root",required=True,type=Path); parser.add_argument("--care-root",required=True,type=Path); parser.add_argument("--split",choices=("train","dev","test"),default="train")
    parser.add_argument("--limit",type=int,default=MAX_PILOT); parser.add_argument("--offset",type=int,default=0); parser.add_argument("--start-after"); parser.add_argument("--per-stratum",type=int,default=10); parser.add_argument("--all",action="store_true"); parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--sample-ids-file",type=Path); parser.add_argument("--retry-blocked-only",action="store_true"); parser.add_argument("--budget-ledger",type=Path)
    parser.add_argument("--report-dir",type=Path,default=Path("/tmp/carewsm-task003c")); parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--weights",type=Path); parser.add_argument("--embedding-weights",type=Path); parser.add_argument("--xlsr-model",type=Path); parser.add_argument("--trillsson-model",type=Path); parser.add_argument("--emotion-model",type=Path)
    args=parser.parse_args()
    if args.limit < 0 or args.offset < 0 or args.per_stratum < 0: parser.error("limit, offset and per-stratum must be non-negative")
    if args.split != "train" or args.all or args.offset or args.start_after or args.per_stratum not in (0,10):
        if not args.retry_blocked_only and args.split != "train": parser.error("only TRAIN is permitted in this corrective task")
    if args.retry_blocked_only:
        if args.split != "train" or args.all or not args.sample_ids_file or args.offset or args.start_after or args.per_stratum: parser.error("blocked retry requires TRAIN, no pagination, no --all and --sample-ids-file")
        allowlist=load_allowlist(args.sample_ids_file); args.limit=len(allowlist)
    elif args.sample_ids_file:
        parser.error("--sample-ids-file requires --retry-blocked-only")
    else:
        parser.error("new sample selection is frozen; only the exact seven blocked-ID retry is authorized")
    os.environ.update({"HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,on_signal)
    args.wsm_root=args.wsm_root.resolve(); args.care_root=args.care_root.resolve(); args.report_dir=args.report_dir.resolve(); args.report_dir.mkdir(parents=True,exist_ok=True)
    args.budget_ledger=(args.budget_ledger or args.report_dir/"task003c_budget.json").resolve()
    expected={name:descriptors.reference_headers(args.care_root,name) for name in descriptors.FILES}; expected["emotionmsp_trajectories.csv"]=list(deep.TRAJ_HEADER)
    rows=load_rows(args.wsm_root,args.split); by_id={row["sample_id"]:row for row in rows}
    if args.retry_blocked_only:
        if any(sample_id not in by_id for sample_id in allowlist): raise RuntimeError("allowlist ID missing from TRAIN manifest")
        selected=[by_id[sample_id] for sample_id in allowlist]
    else:
        # Frozen in this corrective branch; kept unreachable as a fail-closed guard.
        selected=stable_selection(rows,args)
    for row in rows:
        sidecar=read_sidecar(row,args)
        row["existing_state"]=inspect_record(row["destination"],expected,sidecar[0] if sidecar else None)
    if args.retry_blocked_only:
        selected=[by_id[sample_id] for sample_id in allowlist]
        if any(row["existing_state"]["complete"] for row in selected):
            raise RuntimeError("approved blocked allowlist contains a now-complete record; refusing ambiguous retry")
    ledger=budget_receipt(args.budget_ledger,rows,args.dry_run,args.retry_blocked_only)
    result={"task":"TASK-003C","split":"train","status":"dry_run" if args.dry_run else "running","selection":{"selected":len(selected),"allowlist_exact":args.retry_blocked_only},"budget":ledger,"selected":[{"sample_id":r["sample_id"],"stratum":r["stratum"],"video_path":r["video_path"],"destination":str(r["destination"]),"existing":r["existing_state"]} for r in selected],"downloads":False}
    if args.dry_run:
        print(json.dumps(result,indent=2,sort_keys=True)); return 0
    missing_deps=[name for name in ("numpy","pandas","librosa","soundfile","opensmile","disvoice","torch","torchaudio","diarizen","transformers","tensorflow","tensorflow_hub") if importlib.util.find_spec(name) is None]
    if missing_deps: result.update(status="blocked",blocker={"type":"missing_local_dependency","packages":missing_deps}); print(json.dumps(result,indent=2)); return 1
    checkpoint=args.report_dir/f"{args.split}.jsonl"; processed=[]; descriptor_smiles=descriptors.build_descriptor_models()[0]; models=load_deep_models(args); pipeline=None
    with tempfile.TemporaryDirectory(prefix="carewsm-task003c-",dir=args.report_dir) as worker_scratch:
        worker=Path(worker_scratch); praat_log=worker/"praat.log"; configure_praat_wrapper(worker,praat_log)
        if any(read_sidecar(row,args) is None and not row["existing_state"]["complete"] for row in selected):
            pipeline=diarizer_pipeline(args)
        for index,row in enumerate(selected,1):
            if STOP_REQUESTED: break
            started=time.perf_counter(); destination=row["destination"]
            if row["existing_state"]["complete"]:
                processed.append({"sample_id":row["sample_id"],"status":"skipped_complete"}); continue
            destination.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f"{row['sample_id']}-",dir=worker) as record_scratch_name:
                record_scratch=Path(record_scratch_name); record_log=record_scratch/"praat.log"; configure_praat_wrapper(record_scratch,record_log)
                try:
                    wav=decode_audio(row,record_scratch); turns,provenance=diarize_once(wav,row,args,pipeline,record_log)
                    descriptor_models=(descriptor_smiles,descriptors.build_disvoice_models())
                    written,skipped=descriptors.extract_row({**row,"audio_wav":wav},turns,record_scratch,expected,models=descriptor_models)
                    record={"sample_id":row["sample_id"],"status":"running","turn_count":len(turns),"turn_provenance":provenance,"written":{**written},"skipped":{**skipped}}
                    state=inspect_record(destination,expected,turns); missing={name for name,status in state["statuses"].items() if status=="missing"}; deep_ready={"wav2vec_xlsr300m.npz":"xlsr","trillsson4_1.npz":"trill","emotionmsp_embeds.npz":"emotion","emotionmsp_trajectories.csv":"emotion"}; required={name for name in missing if name in deep_ready and deep_ready[name] in models}
                    outputs=deep_outputs(wav,turns,required,models,args.device) if required else {}
                    if outputs: record["written"].update(stage_deep(outputs,destination))
                    final=inspect_record(destination,expected,turns)
                    if not final["complete"]: raise RuntimeError(str(final))
                    record.update(status="complete",validation={"status":"complete","files":len(ALL_FILES)},elapsed_s=round(time.perf_counter()-started,3),rss_max_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,praat_log_tail=diagnostic_tail(record_log)); processed.append(record)
                except Exception as exc:
                    processed.append({"sample_id":row["sample_id"],"status":"blocked_disvoice_vuv" if "tempVUV" in str(exc) else "blocked","error":str(exc),"elapsed_s":round(time.perf_counter()-started,3),"praat_log_tail":diagnostic_tail(record_log)})
                with checkpoint.open("a",encoding="utf-8") as handle: handle.write(json.dumps(processed[-1],sort_keys=True)+"\n"); handle.flush()
                print(f"[{index}/{len(selected)}] {processed[-1]['status']} {row['sample_id']}",flush=True)
    if args.retry_blocked_only:
        ledger_data=json.loads(args.budget_ledger.read_text(encoding="utf-8"))
        ledger_data["retry_used"]=True
        ledger_data["retry_results"]={item["sample_id"]: item["status"] for item in processed}
        ledger_data["new_complete_sample_ids"]=sorted(set(ledger_data.get("new_complete_sample_ids", [])) | {item["sample_id"] for item in processed if item["status"] == "complete"})
        args.budget_ledger.write_text(json.dumps(ledger_data,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    result["runs"]=processed; result["checkpoint"]=str(checkpoint); result["scratch_cleaned"]=True; result["status"]="ok" if processed and all(item["status"]=="complete" for item in processed) else ("interrupted" if STOP_REQUESTED else "blocked")
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"]=="ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
