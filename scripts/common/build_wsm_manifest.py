#!/usr/bin/env python3
"""Build the canonical WSM partial-label segment manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

from common.data.wsm_manifest import ManifestError, build_manifest, write_audit, write_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _check_output(path: Path, data_root: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ManifestError(f"refusing to overwrite existing output without --overwrite: {path}")
    if path.is_relative_to(data_root):
        raise ManifestError(f"refusing to write output inside dataset root: {path}")
    forbidden = {_repo_root() / "src", _repo_root() / "configs", _repo_root() / "docs"}
    if any(path == root or root in path.parents for root in forbidden):
        raise ManifestError(f"refusing to write output inside protected repository path: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    manifest_output = args.manifest_output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    if manifest_output == audit_output:
        raise ManifestError("manifest and audit outputs must be different")
    _check_output(manifest_output, data_root, args.overwrite)
    _check_output(audit_output, data_root, args.overwrite)
    rows, audit = build_manifest(data_root)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, manifest_output)
    write_audit(audit, audit_output)
    print(f"wrote {len(rows)} manifest rows to {manifest_output}")
    print(f"wrote manifest audit to {audit_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as exc:
        raise SystemExit(f"manifest build failed: {exc}")
