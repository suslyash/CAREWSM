#!/usr/bin/env python3
"""Resumable offline WSM extraction of all eleven CARE audio families.

The batch worker decodes and diarizes each recording once, reuses one turn list
for every family, loads feature models once, and publishes only validated files.
It never downloads models and never overwrites an existing file.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
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
STRATA = ("DEPRESSION", "CONTROL", "PARKINSON")
ALL_FILES = (
    *descriptors.FILES,
    "wav2vec_xlsr300m.npz",
    "trillsson4_1.npz",
    "emotionmsp_trajectories.csv",
    "emotionmsp_embeds.npz",
)
DEEP_NAMES = {"wav2vec_xlsr300m.npz", "trillsson4_1.npz", "emotionmsp_trajectories.csv", "emotionmsp_embeds.npz"}

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
            if Path(rel).is_absolute() or root not in video.parents or not video.is_file() or video.is_symlink() and root not in video.parents:
                continue
            destination = (root / "extracted" / row["label_id"] / row["subgroup"] / f"PROFILE_{row['profile_id']}" / f"VIDEO_{row['video_id']}").resolve()
            if root not in destination.parents:
                raise RuntimeError(f"destination escapes WSM root: {destination}")
            rows.append({**row, "video": video, "destination": destination})
    rows.sort(key=lambda item: (item["sample_id"], item["video_path"]))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample_id in manifest")
    destinations = [str(row["destination"]) for row in rows]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("duplicate canonical destination in manifest")
    return rows


def npz_check(path: Path, filename: str) -> dict:
    return validator.validate_npz(path, filename)


def read_turn_metadata(path: Path) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            turn = int(float(row["turn_id"]))
            times = (float(row["turn_start_sec"]), float(row["turn_end_sec"]))
            if turn in result and any(abs(a - b) > 1e-5 for a, b in zip(result[turn], times)):
                raise RuntimeError(f"inconsistent timing in {path}")
            result[turn] = times
    return result


def inspect_record(destination: Path, expected: dict[str, list[str]]) -> dict:
    """Validate existing files and return missing/invalid/cross-family state."""
    statuses = {}; errors = []; timing_maps = []
    for filename in ALL_FILES:
        path = destination / filename
        if not path.is_file():
            statuses[filename] = "missing"
            continue
        try:
            if filename in descriptors.FILES:
                checked = validator.validate_csv(path, expected[filename])
                if checked["status"] != "ok":
                    raise RuntimeError(str(checked))
                timing_maps.append(read_turn_metadata(path))
            elif filename.endswith(".npz"):
                checked = npz_check(path, filename)
                if checked["status"] != "ok":
                    raise RuntimeError(str(checked))
            else:
                checked = validator.validate_csv(path, expected[filename])
                if checked["status"] != "ok":
                    raise RuntimeError(str(checked))
                timing_maps.append(read_turn_metadata(path))
            statuses[filename] = "valid_existing"
        except Exception as exc:
            statuses[filename] = "blocked_existing_invalid"
            errors.append(f"{filename}: {exc}")
    valid_maps = [item for item in timing_maps if item]
    if valid_maps:
        first = valid_maps[0]
        for item in valid_maps[1:]:
            if item != first:
                errors.append("cross-family turn metadata mismatch")
    complete = all(statuses.get(name) == "valid_existing" for name in ALL_FILES) and not errors
    return {"statuses": statuses, "errors": errors, "complete": complete}


def choose_rows(rows: list[dict], args, expected: dict[str, list[str]]) -> tuple[list[dict], dict]:
    eligible = []
    complete_count = 0
    for row in rows:
        state = inspect_record(row["destination"], expected)
        row["existing_state"] = state
        if state["complete"]:
            complete_count += 1
        else:
            eligible.append(row)
    if args.start_after:
        eligible = [row for row in eligible if row["sample_id"] > args.start_after]
    if args.offset:
        eligible = eligible[args.offset:]
    limit = args.limit
    if not args.all:
        limit = min(limit, MAX_PILOT)
    if limit < 0 or args.offset < 0:
        raise RuntimeError("limit and offset must be non-negative")
    selected = []
    if args.per_stratum:
        quota = min(args.per_stratum, limit)
        for stratum in STRATA:
            selected.extend([row for row in eligible if row["stratum"] == stratum][:quota])
        selected_ids = {row["sample_id"] for row in selected}
        selected.extend(row for row in eligible if row["sample_id"] not in selected_ids)
    else:
        selected = eligible
    selected = selected[:limit]
    return selected, {"complete_existing": complete_count, "eligible_new": len(eligible), "selected": len(selected)}


def write_sidecar(row: dict, turns: list[tuple[float, float]], args) -> None:
    destination = row["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "_carewsm_audio_turns.json"
    payload = {"sample_id": row["sample_id"], "video_path": row["video_path"], "video_size": row["video"].stat().st_size, "video_mtime_ns": row["video"].stat().st_mtime_ns, "weights": str(args.weights), "embedding_weights": str(args.embedding_weights), "policy": "CARE longest-speaker provisional participant; max_gap=3.0; min_turn_duration=0.3", "turns": [[float(a), float(b)] for a, b in turns]}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old == payload:
                return
        except Exception:
            return
        return
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination, prefix="._carewsm_audio_turns.", suffix=".partial", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_sidecar(row: dict, args) -> list[tuple[float, float]] | None:
    path = row["destination"] / "_carewsm_audio_turns.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("sample_id") != row["sample_id"] or data.get("video_path") != row["video_path"]:
            return None
        stat = row["video"].stat()
        if data.get("video_size") != stat.st_size or data.get("video_mtime_ns") != stat.st_mtime_ns:
            return None
        turns = [(float(a), float(b)) for a, b in data["turns"]]
        if not turns or any(b - a < 0.3 for a, b in turns):
            return None
        return turns
    except Exception:
        return None


def decode_audio(row: dict, scratch: Path) -> Path:
    import subprocess
    wav = scratch / f"{row['sample_id']}.mono16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(row["video"]), "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
    return wav


def diarize_once(wav: Path, row: dict, args) -> tuple[list[tuple[float, float]], dict]:
    cached = load_sidecar(row, args)
    if cached is not None:
        return cached, {"source": "validated sidecar", "diarization_runs": 0}
    if not args.weights or not args.embedding_weights or not args.weights.is_dir() or not args.embedding_weights.is_file():
        raise RuntimeError("explicit local DiariZen weights and embedding weights are required")
    missing = deep.package_missing(("torch", "torchaudio", "diarizen"))
    if missing:
        raise RuntimeError(f"missing local DiariZen dependencies: {missing}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU DiariZen is prohibited")
    from diarizen.pipelines.inference import DiariZenPipeline
    pipeline = DiariZenPipeline(
        diarizen_hub=args.weights, embedding_model=str(args.embedding_weights),
        config_parse={
            "model": {"args": {"wavlm_src": "wavlm_large_s80_md", "wavlm_layer_num": 25, "wavlm_feat_dim": 1024, "attention_in": 256, "ffn_hidden": 1024, "num_head": 4, "num_layer": 4, "dropout": 0.1, "chunk_size": 16, "use_posi": False, "output_activate_function": False, "selected_channel": 0}},
            "inference": {"args": {"seg_duration": 16, "segmentation_step": 0.1, "batch_size": 1, "apply_median_filtering": True}},
            "clustering": {"args": {"method": "VBxClustering", "min_speakers": 1, "max_speakers": 20, "ahc_criterion": "distance", "ahc_threshold": 0.6, "Fa": 0.07, "Fb": 0.8, "lda_dim": 128, "max_iters": 20}},
        }, rttm_out_dir=None)
    diarization = pipeline(str(wav))
    segments = []; durations = {}
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
    del pipeline, diarization
    gc.collect(); torch.cuda.empty_cache()
    return turns, {"source": "local DiariZen", "diarization_runs": 1, "speaker_count": len(durations), "segment_count": len(segments)}


def load_deep_models(args, need_deep: bool) -> dict[str, Any]:
    if not need_deep:
        return {}
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


def deep_outputs(wav: Path, turns, missing: set[str], models: dict[str, Any], device: str) -> dict[str, object]:
    import librosa
    waveform, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
    output = {}
    if "wav2vec_xlsr300m.npz" in missing:
        model, torch = models["xlsr"]; tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(device)
        means=[]; stds=[]; ids=[]
        with torch.no_grad():
            for turn_id, (start, end) in enumerate(turns, 1):
                feats = model.extract_features(tensor[:, round(start*SAMPLE_RATE):round(end*SAMPLE_RATE)])[0][-1]
                means.append(feats.mean(dim=1).squeeze(0).cpu().numpy()); stds.append(feats.std(dim=1, unbiased=False).squeeze(0).cpu().numpy()); ids.append(turn_id)
        output["wav2vec_xlsr300m.npz"] = {"mean": np.asarray(means, dtype=np.float32), "std": np.asarray(stds, dtype=np.float32), "turn_ids": np.asarray(ids, dtype=np.int64)}
    if "trillsson4_1.npz" in missing:
        model, tf = models["trill"]; means=[]; stds=[]; ids=[]
        for turn_id, (start, end) in enumerate(turns, 1):
            chunk=tf.convert_to_tensor(waveform[round(start*SAMPLE_RATE):round(end*SAMPLE_RATE)], dtype=tf.float32)
            feats=model(tf.expand_dims(chunk,0))["embedding"]
            means.append(tf.reduce_mean(feats,axis=0).numpy()); stds.append(tf.math.reduce_std(feats,axis=0).numpy()); ids.append(turn_id)
        output["trillsson4_1.npz"]={"mean":np.asarray(means,dtype=np.float32),"std":np.asarray(stds,dtype=np.float32),"turn_ids":np.asarray(ids,dtype=np.int64)}
    if {"emotionmsp_embeds.npz", "emotionmsp_trajectories.csv"} & missing:
        processor, model, torch = models["emotion"]; embeds=[]; rows=[]; ids=[]
        with torch.no_grad():
            for turn_id,(start,end) in enumerate(turns,1):
                values=processor(waveform[round(start*SAMPLE_RATE):round(end*SAMPLE_RATE)],sampling_rate=SAMPLE_RATE,return_tensors="pt")["input_values"].to(device)
                embed, trajectory=model(values); embeds.append(embed.squeeze(0).cpu().numpy()); ids.append(turn_id); score=trajectory.squeeze(0).cpu().numpy()
                rows.append((turn_id,start,end,float(score[0]),float(score[1]),float(score[2])))
        output["emotionmsp_embeds.npz"]={"embed":np.asarray(embeds,dtype=np.float32),"turn_ids":np.asarray(ids,dtype=np.int64)}
        output["emotionmsp_trajectories.csv"]=rows
    return {name:value for name,value in output.items() if name in missing}


def stage_deep(outputs: dict[str, object], destination: Path, turns) -> dict:
    staged={}; temp_dir=Path(tempfile.mkdtemp(prefix=".carewsm-deep-stage-",dir=destination))
    try:
        for name,value in outputs.items():
            path=temp_dir/name
            if name.endswith(".npz"):
                np.savez(path,**value); checked=validator.validate_npz(path,name)
                if checked["status"] != "ok": raise RuntimeError(f"invalid staged {name}: {checked}")
            else:
                with path.open("w",newline="",encoding="utf-8") as handle:
                    writer=csv.writer(handle); writer.writerow(deep.TRAJ_HEADER); writer.writerows(value)
                checked=validator.validate_csv(path,list(deep.TRAJ_HEADER))
                if checked["status"] != "ok": raise RuntimeError(f"invalid staged {name}: {checked}")
            staged[name]=path
        for name,path in staged.items():
            target=destination/name
            if target.exists(): raise FileExistsError(f"refusing overwrite: {target}")
        linked=[]
        try:
            for name,path in staged.items(): os.link(path,destination/name); linked.append(destination/name)
        except Exception:
            for path in linked: path.unlink(missing_ok=True)
            raise
        return {name:{"status":"created","bytes":(destination/name).stat().st_size} for name in staged}
    finally:
        shutil.rmtree(temp_dir,ignore_errors=True)


def cross_validate(destination: Path, expected: dict[str,list[str]]) -> dict:
    state=inspect_record(destination,expected)
    if not state["complete"]: raise RuntimeError(str(state))
    return {"status":"complete","files":len(ALL_FILES)}


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--wsm-root",required=True,type=Path); parser.add_argument("--care-root",required=True,type=Path)
    parser.add_argument("--split",choices=("train","dev","test"),default="train")
    parser.add_argument("--limit",type=int,default=MAX_PILOT); parser.add_argument("--offset",type=int,default=0); parser.add_argument("--start-after")
    parser.add_argument("--per-stratum",type=int,default=10); parser.add_argument("--all",action="store_true"); parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--report-dir",type=Path,default=Path("/tmp/carewsm-task003c")); parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--weights",type=Path); parser.add_argument("--embedding-weights",type=Path); parser.add_argument("--xlsr-model",type=Path); parser.add_argument("--trillsson-model",type=Path); parser.add_argument("--emotion-model",type=Path)
    args=parser.parse_args()
    if args.limit < 0 or args.offset < 0 or args.per_stratum < 0: parser.error("limit, offset and per-stratum must be non-negative")
    if args.split != "train" and not args.all: parser.error("--all is required for DEV/TEST")
    if args.all and args.limit == 0: parser.error("provide a positive chunk --limit")
    os.environ.update({"HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig,on_signal)
    args.wsm_root=args.wsm_root.resolve(); args.care_root=args.care_root.resolve(); args.report_dir=args.report_dir.resolve(); args.report_dir.mkdir(parents=True,exist_ok=True)
    expected={name:descriptors.reference_headers(args.care_root,name) for name in descriptors.FILES}; expected["emotionmsp_trajectories.csv"]=list(deep.TRAJ_HEADER)
    rows=load_rows(args.wsm_root,args.split); selected,summary=choose_rows(rows,args,expected)
    result={"task":"TASK-003C","split":args.split,"status":"dry_run" if args.dry_run else "running","selection":summary,"selected":[{"sample_id":r["sample_id"],"stratum":r["stratum"],"video_path":r["video_path"],"destination":str(r["destination"]),"existing":r["existing_state"]} for r in selected],"downloads":False}
    if args.dry_run:
        print(json.dumps(result,indent=2,sort_keys=True)); return 0
    checkpoint=args.report_dir/f"{args.split}.jsonl"
    need_any=any(not row["existing_state"]["complete"] for row in selected)
    missing_deps=[name for name in ("numpy","pandas","librosa","soundfile","opensmile","disvoice","torch","torchaudio","diarizen","transformers","tensorflow","tensorflow_hub") if importlib.util.find_spec(name) is None]
    if missing_deps: result.update(status="blocked",blocker={"type":"missing_local_dependency","packages":missing_deps}); print(json.dumps(result,indent=2)); return 1
    models={}; descriptor_smiles=None; processed=[]
    with tempfile.TemporaryDirectory(prefix="carewsm-task003c-",dir=args.report_dir) as scratch_name:
        scratch=Path(scratch_name)
        real_praat=shutil.which("praat")
        if real_praat is None:
            raise RuntimeError("Praat executable is unavailable")
        wrapper=scratch / "praat"
        wrapper.write_text("#!/bin/sh\nexec timeout --kill-after=5 90 " + shlex.quote(real_praat) + " --run \"$@\"\n", encoding="utf-8")
        wrapper.chmod(0o700)
        os.environ["PATH"] = f"{scratch}{os.pathsep}{os.environ.get('PATH','')}"
        prepared=[]
        for index,row in enumerate(selected,1):
            if STOP_REQUESTED: break
            if row["existing_state"]["complete"]:
                processed.append({"sample_id":row["sample_id"],"status":"skipped_complete","diarization_runs":0}); continue
            if row["existing_state"]["errors"]:
                processed.append({"sample_id":row["sample_id"],"status":"blocked_existing_invalid","errors":row["existing_state"]["errors"]}); continue
            started=time.perf_counter(); destination=row["destination"]; destination.mkdir(parents=True,exist_ok=True)
            try:
                wav=decode_audio(row,scratch); turns,provenance=diarize_once(wav,row,args); prepared.append((row,wav,turns,provenance,started))
                print(f"[{index}/{len(selected)}] prepared {row['sample_id']} turns={len(turns)}",flush=True)
            except Exception as exc:
                processed.append({"sample_id":row["sample_id"],"status":"blocked","error":str(exc),"elapsed_s":round(time.perf_counter()-started,3)})
                print(f"[{index}/{len(selected)}] blocked {row['sample_id']}: {exc}",flush=True)
        if prepared:
            descriptor_smiles=descriptors.build_descriptor_models()[0]
            models=load_deep_models(args,True)
            for index,(row,wav,turns,provenance,started) in enumerate(prepared,1):
                if STOP_REQUESTED: break
                record={"sample_id":row["sample_id"],"status":"running","turn_count":len(turns),"turn_provenance":provenance}
                try:
                    destination=row["destination"]
                    descriptor_models=(descriptor_smiles, descriptors.build_disvoice_models())
                    written,skipped=descriptors.extract_row({**row,"audio_wav":wav},turns,scratch,expected,models=descriptor_models)
                    record["written"]={**written}; record["skipped"]={**skipped}
                    state=inspect_record(destination,expected); missing={name for name,status in state["statuses"].items() if status=="missing"}
                    deep_ready={"wav2vec_xlsr300m.npz":"xlsr","trillsson4_1.npz":"trill","emotionmsp_embeds.npz":"emotion","emotionmsp_trajectories.csv":"emotion"}
                    required={name for name in missing if name in deep_ready and deep_ready[name] in models}
                    outputs=deep_outputs(wav,turns,required,models,args.device) if required else {}
                    if outputs: record["written"].update(stage_deep(outputs,destination,turns))
                    record["validation"]=cross_validate(destination,expected); record["status"]="complete"; record["elapsed_s"]=round(time.perf_counter()-started,3); record["rss_max_kb"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if importlib.util.find_spec("torch"):
                        import torch
                        if torch.cuda.is_available(): record["cuda_max_allocated_mb"]=round(torch.cuda.max_memory_allocated()/1024/1024,1)
                except Exception as exc:
                    record.update(status="blocked",error=str(exc),elapsed_s=round(time.perf_counter()-started,3))
                processed.append(record); print(f"[{index}/{len(prepared)}] {record['status']} {row['sample_id']} elapsed={record.get('elapsed_s')}",flush=True)
                with checkpoint.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record,sort_keys=True)+"\n"); handle.flush()
    result["runs"]=processed; result["status"]="ok" if processed and all(item["status"] in {"complete","skipped_complete"} for item in processed) and not STOP_REQUESTED else ("interrupted" if STOP_REQUESTED else "blocked"); result["checkpoint"]=str(checkpoint); result["scratch_cleaned"]=True
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"]=="ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
