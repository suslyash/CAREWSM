#!/usr/bin/env python3
"""Inventory or safely remove only the obsolete DiariZen-era WSM audio outputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

AUDIO_FILES = frozenset({
    "egemaps_func.csv", "egemaps_lld.csv", "disvoice_prosody_static.csv",
    "disvoice_prosody_dynamic.csv", "disvoice_phonation_static.csv",
    "disvoice_phonation_dynamic.csv", "disvoice_articulation_static.csv",
    "wav2vec_xlsr300m.npz", "trillsson4_1.npz",
    "emotionmsp_trajectories.csv", "emotionmsp_embeds.npz",
    "_carewsm_audio_turns.json",
})


def old_job_running(runtime: Path) -> list[str]:
    blockers = []
    try:
        session = subprocess.run(["tmux", "has-session", "-t", "carewsm-audio-prod"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if session.returncode == 0:
            blockers.append("tmux session carewsm-audio-prod still exists")
    except OSError:
        pass
    ignored_pids = {os.getpid(), os.getppid()}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ignored_pids:
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(name in command for name in ("wsm_run_all_audio.py", "wsm_extract_care_audio_batch.py", "wsm_extract_full_segment_audio.py")):
            blockers.append(f"active extractor process {proc.name}: {command.strip()}")
    lock = runtime / "production.lock"
    if lock.exists():
        # An empty/stale lock is not sufficient evidence of a live process, but is
        # retained as a visible warning; the tmux/process checks remain authoritative.
        blockers.append(f"old runtime lock exists: {lock}")
    return blockers


def inventory(root: Path) -> list[dict]:
    root = root.resolve()
    extracted = root / "extracted"
    if root not in extracted.parents or not extracted.is_dir() or extracted.is_symlink():
        raise RuntimeError(f"unsafe or missing extracted root: {extracted}")
    found = []
    stack = [extracted]
    while stack:
        directory = stack.pop()
        if directory.is_symlink():
            raise RuntimeError(f"symlink directory refused: {directory}")
        for entry in os.scandir(directory):
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                stack.append(path)
            elif entry.is_symlink():
                if path.name in AUDIO_FILES:
                    raise RuntimeError(f"symlink eligible file refused: {path}")
            elif entry.is_file(follow_symlinks=False) and path.name in AUDIO_FILES:
                resolved = path.resolve()
                if root not in resolved.parents:
                    raise RuntimeError(f"eligible file escapes WSM root: {path}")
                found.append({"path": str(path), "name": path.name, "bytes": path.stat().st_size})
    return sorted(found, key=lambda item: item["path"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsm-root", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--delete-audio-only", action="store_true")
    parser.add_argument("--confirm-replace-diarizen-outputs", action="store_true")
    args = parser.parse_args()
    root = args.wsm_root.resolve(); runtime = args.runtime.resolve()
    if root == runtime or root in runtime.parents:
        raise SystemExit("runtime must be outside WSM root")
    targets = inventory(root)
    blockers = old_job_running(runtime)
    result = {"status": "blocked" if blockers else "ready", "mode": "delete" if args.delete_audio_only else "inventory", "targets": targets, "target_count": len(targets), "target_bytes": sum(item["bytes"] for item in targets), "blockers": blockers, "allowlist": sorted(AUDIO_FILES)}
    if args.delete_audio_only:
        if not args.confirm_replace_diarizen_outputs:
            raise SystemExit("deletion requires --confirm-replace-diarizen-outputs")
        if blockers:
            print(json.dumps(result, indent=2, sort_keys=True)); return 2
        receipt = runtime / "audio_reset.receipt.json"
        if receipt.exists():
            result.update(status="already_reset", receipt=str(receipt), deleted_count=0)
        else:
            runtime.mkdir(parents=True, exist_ok=True)
            deleted = []
            for item in targets:
                path = Path(item["path"])
                path.unlink()
                deleted.append(item)
            payload = {"task": "TASK-004A", "policy": "full_video_audio_no_diarization", "deleted_at": time.time(), "deleted": deleted, "count": len(deleted), "bytes": sum(x["bytes"] for x in deleted)}
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result.update(status="deleted", deleted_count=len(deleted), receipt=str(receipt))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"ready", "deleted", "already_reset"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
