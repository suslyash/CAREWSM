#!/usr/bin/env python3
"""Owner-operated resumable launcher for the eleven WSM audio families.

Default mode is a bounded TRAIN-only read-only preflight. Full extraction requires
both --production --all-splits and CAREWSM_ALLOW_PRODUCTION=1.
"""
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

AUDIO = Path(__file__).resolve().parent
WORKER = AUDIO / "wsm_extract_care_audio_batch.py"
SPLITS = ("train", "dev", "test")
ALL_FILES = (
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


def safe_component(value: str, field: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or value in {".", ".."} or path.name != value or ".." in path.parts:
        raise RuntimeError(f"unsafe manifest {field}: {value!r}")


def destination(root: Path, row: dict) -> Path:
    for field in ("label_id", "subgroup", "profile_id", "video_id", "stratum"):
        safe_component(row.get(field, ""), field)
    path = (root / "extracted" / row["label_id"] / row["subgroup"] / f"PROFILE_{row['profile_id']}" / f"VIDEO_{row['video_id']}").resolve()
    if root.resolve() not in path.parents or path.is_symlink():
        raise RuntimeError(f"unsafe canonical destination for {row['sample_id']}: {path}")
    return path


def manifest_rows(root: Path, split: str) -> tuple[list[dict], str]:
    path = root / "splits" / f"{split}.csv"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "video_path", "split", "storage_type", "label_id", "subgroup", "profile_id", "video_id", "stratum"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise RuntimeError(f"{path}: missing columns {missing}")
        rows = []
        for row in reader:
            if row.get("split") != split or row.get("storage_type") != "files":
                continue
            safe_component(row.get("sample_id", ""), "sample_id")
            video_rel = row.get("video_path", "")
            if not video_rel or Path(video_rel).is_absolute():
                raise RuntimeError(f"unsafe manifest video_path for {row.get('sample_id')}: {video_rel!r}")
            video = (root / video_rel).resolve()
            if root.resolve() not in video.parents:
                raise RuntimeError(f"video escapes WSM root for {row['sample_id']}: {video}")
            rows.append({**row, "video": video, "destination": destination(root, row)})
    rows.sort(key=lambda row: (row["sample_id"], row["video_path"]))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path}: duplicate sample_id")
    destinations = [str(row["destination"]) for row in rows]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError(f"{path}: duplicate canonical destination")
    return rows, digest


def state_path(runtime: Path, split: str) -> Path:
    return runtime / "state" / f"{split}.json"


def load_state(path: Path, digest: str, count: int) -> dict:
    if not path.exists():
        return {"manifest_sha256": digest, "row_count": count, "next_index": 0, "records": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("manifest_sha256") != digest or state.get("row_count") != count:
        raise RuntimeError(f"manifest changed since checkpoint: {path}")
    if not 0 <= int(state.get("next_index", 0)) <= count:
        raise RuntimeError(f"invalid checkpoint cursor: {path}")
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_counts(state: dict) -> None:
    counts = {"complete": 0, "already_complete": 0, "blocked": 0, "invalid_existing": 0, "missing_media": 0, "excluded": 0}
    for item in state.get("records", {}).values():
        status = item.get("status", "")
        if status == "complete":
            counts["complete"] += 1
        elif status == "skipped_complete":
            counts["already_complete"] += 1
        elif status == "invalid_existing":
            counts["invalid_existing"] += 1
        elif status == "missing_media":
            counts["missing_media"] += 1
        elif status.startswith("blocked") or status == "blocked_unreported":
            counts["blocked"] += 1
    state["counts"] = counts


def write_summary(runtime: Path, summaries: dict) -> None:
    temporary = runtime / "summary.json.partial"
    temporary.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, runtime / "summary.json")


def last_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    found = None
    found_end = -1
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and index + end > found_end:
            found, found_end = value, index + end
    if found is None:
        raise RuntimeError("worker produced no JSON summary")
    return found


def preflight(args) -> int:
    errors = []
    if shutil.which("praat") is None:
        errors.append("missing executable: praat")
    if shutil.which("xvfb-run") is None and not os.environ.get("DISPLAY"):
        errors.append("missing headless display: xvfb-run or DISPLAY")
    for package in ("numpy", "pandas", "librosa", "soundfile", "opensmile", "disvoice", "torch", "torchaudio", "diarizen", "transformers", "tensorflow", "tensorflow_hub"):
        if importlib.util.find_spec(package) is None:
            errors.append(f"missing local package: {package}")
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA unavailable; CPU PyTorch is prohibited")
    except Exception as exc:
        errors.append(f"CUDA preflight failed: {exc}")
    for label, path, kind in (("DiariZen weights", args.weights, "dir"), ("embedding weights", args.embedding_weights, "file"), ("XLS-R weights", args.xlsr_model, "file"), ("TRILLsson model", args.trillsson_model, "path"), ("Emotion model", args.emotion_model, "dir")):
        if (kind == "dir" and not path.is_dir()) or (kind == "file" and not path.is_file()) or (kind == "path" and not path.exists()):
            errors.append(f"missing local {label}: {path}")
    root = args.wsm_root.resolve()
    runtime = args.runtime.resolve()
    if root in runtime.parents or runtime == root:
        errors.append("runtime directory must be outside WSM root")
    rows, digest = manifest_rows(root, "train")
    selected = rows[:args.preflight_limit]
    expected = {}
    if args.care_root.exists():
        import wsm_extract_care_descriptors as descriptors
        import validate_care_audio_parity as validator
        from wsm_extract_care_deep_audio import TRAJ_HEADER
        for name in descriptors.FILES:
            expected[name] = descriptors.reference_headers(args.care_root, name)
        expected["emotionmsp_trajectories.csv"] = list(TRAJ_HEADER)
    samples = []
    for position, row in enumerate(selected):
        files = {}
        for name in ALL_FILES:
            path = row["destination"] / name
            if not path.is_file():
                files[name] = {"status": "missing_output"}
                continue
            if name.endswith(".npz"):
                checked = validator.validate_npz(path, name)
            else:
                checked = validator.validate_csv(path, expected.get(name, []))
            files[name] = {"status": checked.get("status"), "bytes": path.stat().st_size}
            if checked.get("status") != "ok":
                errors.append(f"invalid existing {row['sample_id']}:{name}")
        samples.append({"position": position, "sample_id": row["sample_id"], "video_path": row["video_path"], "video_exists": row["video"].is_file(), "destination": str(row["destination"]), "files": files})
        if not row["video"].is_file():
            errors.append(f"missing source media: {row['sample_id']}")
    result = {"status": "ok" if not errors else "blocked", "mode": "read_only_preflight", "split": "train", "manifest_sha256": digest, "row_count": len(rows), "selection": {"start": 0, "end_exclusive": len(selected), "limit": args.preflight_limit}, "samples": samples, "errors": errors, "downloads": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


def production(args) -> int:
    if os.environ.get("CAREWSM_ALLOW_PRODUCTION") != "1":
        raise RuntimeError("production requires CAREWSM_ALLOW_PRODUCTION=1")
    args.runtime.mkdir(parents=True, exist_ok=True)
    lock_path = args.runtime / "production.lock"
    try:
        import fcntl
        lock = lock_path.open("w", encoding="utf-8")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, BlockingIOError) as exc:
        raise RuntimeError("another CAREWSM production launcher is already running") from exc
    summaries = {}
    for split in SPLITS:
        if STOP:
            break
        rows, digest = manifest_rows(args.wsm_root.resolve(), split)
        path = state_path(args.runtime, split)
        state = load_state(path, digest, len(rows))
        update_counts(state)
        summaries[split] = state
        write_summary(args.runtime, summaries)
        while state["next_index"] < len(rows) and not STOP:
            start = state["next_index"]
            end = min(start + args.batch_size, len(rows))
            chunk = rows[start:end]
            existing_ids = []
            for position, row in enumerate(chunk, start):
                if not row["video"].is_file():
                    state["records"][row["sample_id"]] = {"split_position": position, "status": "missing_media", "video_path": row["video_path"]}
                else:
                    existing_ids.append(row["sample_id"])
            ids_file = args.runtime / "chunks" / f"{split}_{start:08d}_{end:08d}.ids"
            ids_file.parent.mkdir(parents=True, exist_ok=True)
            ids_file.write_text("\n".join(existing_ids) + ("\n" if existing_ids else ""), encoding="utf-8")
            worker_summary = {"runs": [], "status": "no_media"}
            log = args.runtime / "logs" / f"{split}_{start:08d}_{end:08d}.log"
            if existing_ids:
                command = [sys.executable, str(WORKER), "--production", "--wsm-root", str(args.wsm_root), "--care-root", str(args.care_root), "--split", split, "--sample-ids-file", str(ids_file), "--limit", str(len(existing_ids)), "--report-dir", str(args.runtime / "worker"), "--weights", str(args.weights), "--embedding-weights", str(args.embedding_weights), "--xlsr-model", str(args.xlsr_model), "--trillsson-model", str(args.trillsson_model), "--emotion-model", str(args.emotion_model), "--device", args.device]
                env = os.environ.copy()
                env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CAREWSM_ALLOW_PRODUCTION": "1"})
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a", encoding="utf-8") as handle:
                    completed = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
                text = log.read_text(encoding="utf-8", errors="replace")
                worker_summary = last_json(text)
                returned = {item.get("sample_id"): item for item in worker_summary.get("runs", [])}
                if set(returned) != set(existing_ids):
                    raise RuntimeError(f"worker did not report every selected ID for {split} positions {start}:{end}; cursor not advanced; see {log}")
                if completed.returncode not in (0, 1):
                    raise RuntimeError(f"worker failed structurally for {split} positions {start}:{end}; cursor not advanced; see {log}")
            returned = {item.get("sample_id"): item for item in worker_summary.get("runs", [])}
            for position, row in enumerate(chunk, start):
                item = state["records"].setdefault(row["sample_id"], {"split_position": position})
                item.update(returned.get(row["sample_id"], {}))
                item["split_position"] = position
                if row["sample_id"] not in returned and item.get("status") != "missing_media":
                    raise RuntimeError(f"unreported eligible ID {row['sample_id']}; cursor not advanced")
            state["next_index"] = end
            state["high_water_sample_id"] = chunk[-1]["sample_id"]
            update_counts(state)
            save_state(path, state)
            summaries[split] = state
            write_summary(args.runtime, summaries)
            print(json.dumps({"split": split, "positions": [start, end], "next_index": end, "worker_status": worker_summary.get("status"), "log": str(log)}, sort_keys=True), flush=True)
        update_counts(state)
        summaries[split] = state
        write_summary(args.runtime, summaries)
    unresolved = [item for state in summaries.values() for item in state.get("records", {}).values() if item.get("status") not in {"complete", "skipped_complete"}]
    return 1 if unresolved or STOP else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsm-root", type=Path, required=True)
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/media/maxim/Programs/CAREWSM-runtime/task003d"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--preflight-limit", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--embedding-weights", type=Path, required=True)
    parser.add_argument("--xlsr-model", type=Path, required=True)
    parser.add_argument("--trillsson-model", type=Path, required=True)
    parser.add_argument("--emotion-model", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--all-splits", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 100:
        parser.error("--batch-size must be between 1 and 100")
    if args.preflight_limit < 1 or args.preflight_limit > 30:
        parser.error("--preflight-limit must be between 1 and 30")
    if args.preflight:
        return preflight(args)
    if not args.production or not args.all_splits:
        print(json.dumps({"status": "dry_run", "message": "use --preflight for TRAIN inspection; use --production --all-splits with CAREWSM_ALLOW_PRODUCTION=1 for owner execution", "worker": str(WORKER)}, indent=2))
        return 0
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    return production(args)


if __name__ == "__main__":
    raise SystemExit(main())
