#!/usr/bin/env python3
"""Bounded, read-only WSM TRAIN audio/video input preflight.

This script deliberately lives outside ChimeraML. It reads only the fixed
TRAIN manifest and uses ffprobe for metadata; it never writes corpus data.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


MAX_LIMIT = 5
REQUIRED_COLUMNS = {"sample_id", "split", "video_path", "storage_type"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsm-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=MAX_LIMIT)
    args = parser.parse_args()
    if not 1 <= args.limit <= MAX_LIMIT:
        parser.error(f"--limit must be between 1 and {MAX_LIMIT}")
    return args


def resolve_video_path(root: Path, row: dict[str, str], row_number: int) -> Path:
    raw_path = row.get("video_path", "").strip()
    if not raw_path:
        raise ValueError(f"row {row_number}: video_path is empty")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError(f"row {row_number}: video_path must be relative: {raw_path}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"row {row_number}: video_path escapes --wsm-root: {raw_path}") from exc
    return resolved


def read_train_rows(manifest_path: Path, root: Path) -> list[dict[str, Any]]:
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"TRAIN manifest missing required columns: {', '.join(missing)}")
        if any(column in columns for column in ("archive_relative_path", "tar_video_prefix")):
            # These columns are allowed to exist, but are not fallback path sources.
            pass

        candidates: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            if row.get("split", "").strip().lower() != "train":
                raise ValueError(f"row {row_number}: expected split=train")
            if row.get("storage_type", "").strip() != "files":
                raise ValueError(f"row {row_number}: unsupported storage_type; expected files")
            video_path = resolve_video_path(root, row, row_number)
            if video_path.is_file():
                candidates.append(
                    {
                        "sample_id": row["sample_id"],
                        "video_path": row["video_path"],
                        "resolved_path": str(video_path),
                        "storage_type": row["storage_type"],
                    }
                )
    return sorted(candidates, key=lambda item: (item["sample_id"], item["video_path"]))


def probe_media(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"probe_error": completed.stderr.strip() or f"ffprobe exit {completed.returncode}"}

    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = audio_streams[0] if audio_streams else {}
    format_info = payload.get("format", {})
    return {
        "audio_stream_present": bool(audio_streams),
        "video_stream_present": bool(video_streams),
        "duration_sec": format_info.get("duration") or audio.get("duration"),
        "sample_rate_hz": audio.get("sample_rate"),
        "channels": audio.get("channels"),
        "audio_codec": audio.get("codec_name"),
    }


def main() -> int:
    args = parse_args()
    root = args.wsm_root.resolve()
    manifest_path = root / "splits" / "train.csv"
    summary: dict[str, Any] = {
        "status": "ok",
        "wsm_root": str(root),
        "manifest": str(manifest_path),
        "split": "train",
        "limit": args.limit,
        "selected_count": 0,
        "ffprobe_available": False,
        "samples": [],
    }

    try:
        if not root.is_dir():
            raise ValueError(f"--wsm-root is not a directory: {root}")
        if not manifest_path.is_file():
            raise ValueError(f"TRAIN manifest not found: {manifest_path}")
        candidates = read_train_rows(manifest_path, root)
        ffprobe = shutil.which("ffprobe")
        summary["ffprobe_available"] = ffprobe is not None
        selected = candidates[: args.limit]
        summary["selected_count"] = len(selected)
        if not selected:
            summary["status"] = "blocked"
            summary["error"] = "no existing TRAIN video_path files resolved from the manifest"
        elif ffprobe is None:
            summary["status"] = "blocked"
            summary["error"] = "ffprobe is unavailable; media metadata was not measured"
        else:
            for sample in selected:
                sample["media"] = probe_media(Path(sample["resolved_path"]), ffprobe)
            summary["samples"] = selected
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        summary["status"] = "blocked"
        summary["error"] = str(exc)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
