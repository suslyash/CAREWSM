#!/usr/bin/env python3
"""Audit authoritative speaker-map coverage and split leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data.wsm_manifest import build_manifest
from common.data.wsm_speaker_map import SpeakerMapError, audit_speaker_map


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _check_output(path: Path, data_root: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SpeakerMapError(f"refusing to overwrite existing output without --overwrite: {path}")
    if path.is_relative_to(data_root):
        raise SpeakerMapError(f"refusing to write output inside dataset root: {path}")
    protected = {_repo_root() / "src", _repo_root() / "configs", _repo_root() / "docs"}
    if any(path == item or item in path.parents for item in protected):
        raise SpeakerMapError(f"refusing to write output inside protected path: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speaker-map", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    _check_output(output, data_root, args.overwrite)
    rows, manifest_audit = build_manifest(data_root)
    report = audit_speaker_map(rows, args.speaker_map, manifest_audit=manifest_audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote speaker split gate report ({report['status']}): {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpeakerMapError as exc:
        raise SystemExit(f"speaker-map audit failed: {exc}")
