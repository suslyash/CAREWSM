#!/usr/bin/env python3
"""Owner-operated, resumable production launcher for all 11 WSM audio families.

The default is read-only. Actual extraction requires both --production and
CAREWSM_ALLOW_PRODUCTION=1. This launcher never downloads or installs assets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

AUDIO = Path(__file__).resolve().parent
WORKER = AUDIO / "wsm_extract_care_audio_batch.py"
SPLITS = ("train", "dev", "test")
STOP = False


def stop(_signum, _frame):
    global STOP
    STOP = True


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
        rows = [row for row in reader if row.get("split") == split and row.get("storage_type") == "files"]
    rows.sort(key=lambda row: (row["sample_id"], row["video_path"]))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path}: duplicate sample_id")
    return rows, digest


def state_path(runtime: Path, split: str) -> Path:
    return runtime / "state" / f"{split}.json"


def load_state(path: Path, digest: str, count: int) -> dict:
    if not path.exists():
        return {"manifest_sha256": digest, "row_count": count, "next_index": 0, "records": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("manifest_sha256") != digest or state.get("row_count") != count:
        raise RuntimeError(f"manifest changed since checkpoint: {path}")
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
            found = value
            found_end = index + end
    if found is None:
        raise RuntimeError("worker produced no JSON summary")
    return found


def production(args) -> int:
    if os.environ.get("CAREWSM_ALLOW_PRODUCTION") != "1":
        raise RuntimeError("production requires CAREWSM_ALLOW_PRODUCTION=1")
    if not args.weights.is_dir() or not args.embedding_weights.is_file():
        raise RuntimeError("explicit local DiariZen weights are missing")
    for path in (args.xlsr_model, args.trillsson_model, args.emotion_model):
        if not path.exists():
            raise RuntimeError(f"explicit local feature weight is missing: {path}")
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
        rows, digest = manifest_rows(args.wsm_root, split)
        path = state_path(args.runtime, split)
        state = load_state(path, digest, len(rows))
        while state["next_index"] < len(rows) and not STOP:
            start = state["next_index"]
            end = min(start + args.batch_size, len(rows))
            chunk = rows[start:end]
            existing_ids = []
            for position, row in enumerate(chunk, start):
                video = (args.wsm_root / row["video_path"]).resolve()
                if not video.is_file() or video.is_symlink() and args.wsm_root.resolve() not in video.parents:
                    state["records"][row["sample_id"]] = {"split_position": position, "status": "missing_media"}
                else:
                    existing_ids.append(row["sample_id"])
            ids_file = args.runtime / "chunks" / f"{split}_{start:08d}_{end:08d}.ids"
            ids_file.parent.mkdir(parents=True, exist_ok=True)
            ids_file.write_text("\n".join(existing_ids) + ("\n" if existing_ids else ""), encoding="utf-8")
            worker_summary = {"runs": []}
            if existing_ids:
                command = [sys.executable, str(WORKER), "--production", "--wsm-root", str(args.wsm_root), "--care-root", str(args.care_root), "--split", split, "--sample-ids-file", str(ids_file), "--limit", str(len(existing_ids)), "--report-dir", str(args.runtime / "worker"), "--weights", str(args.weights), "--embedding-weights", str(args.embedding_weights), "--xlsr-model", str(args.xlsr_model), "--trillsson-model", str(args.trillsson_model), "--emotion-model", str(args.emotion_model), "--device", args.device]
                env = os.environ.copy()
                env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CAREWSM_ALLOW_PRODUCTION": "1"})
                log = args.runtime / "logs" / f"{split}_{start:08d}_{end:08d}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a", encoding="utf-8") as handle:
                    completed = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
                text = log.read_text(encoding="utf-8", errors="replace")
                try:
                    worker_summary = last_json(text)
                except RuntimeError:
                    worker_summary = {"status": "blocked", "error": "worker produced no JSON summary", "runs": []}
                if completed.returncode not in (0, 1):
                    raise RuntimeError(f"worker failed structurally for {split} positions {start}:{end}; see {log}")
            returned = {item.get("sample_id"): item for item in worker_summary.get("runs", [])}
            for position, row in enumerate(chunk, start):
                item = state["records"].setdefault(row["sample_id"], {"split_position": position})
                item.update(returned.get(row["sample_id"], {}))
                item["split_position"] = position
                if row["sample_id"] not in returned and item.get("status") != "missing_media":
                    item["status"] = "blocked_unreported"
            state["next_index"] = end
            state["high_water_sample_id"] = chunk[-1]["sample_id"] if chunk else None
            save_state(path, state)
            print(json.dumps({"split": split, "positions": [start, end], "next_index": end, "worker_status": worker_summary.get("status"), "log": str(args.runtime / "logs" / f"{split}_{start:08d}_{end:08d}.log")}, sort_keys=True), flush=True)
        summaries[split] = state
    (args.runtime / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    unresolved = [item for state in summaries.values() for item in state.get("records", {}).values() if item.get("status") not in {"complete", "skipped_complete"}]
    return 1 if unresolved or STOP else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsm-root", type=Path, required=True)
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/media/maxim/Programs/CAREWSM-runtime/task003d"))
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--embedding-weights", type=Path, required=True)
    parser.add_argument("--xlsr-model", type=Path, required=True)
    parser.add_argument("--trillsson-model", type=Path, required=True)
    parser.add_argument("--emotion-model", type=Path, required=True)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--all-splits", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 100:
        parser.error("--batch-size must be between 1 and 100")
    if not args.production or not args.all_splits:
        print(json.dumps({"status": "dry_run", "message": "read-only by default; use --production --all-splits with CAREWSM_ALLOW_PRODUCTION=1 for owner execution", "worker": str(WORKER)}, indent=2))
        return 0
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    return production(args)


if __name__ == "__main__":
    raise SystemExit(main())
