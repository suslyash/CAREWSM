#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from common.data.wsm_manifest import build_manifest
from text.features.xlmr_video_transcript import EXPECTED_MODEL_SHA256, MODEL_NAME, MODEL_REVISION, XLMRVideoTranscriptEncoder, sha256_file


AUDIT_SHA = "4254d81281522015d0633d6534e1616e925a2bf0130f67a37b541a2aacc2ed78"
VERSION = "t1_xlmr_video_transcript_v1"


def _decode(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            text = data.decode(encoding)
            if not text.strip():
                raise ValueError(f"empty transcript: {path}")
            return text
        except UnicodeDecodeError:
            continue
    raise ValueError(f"transcript is not UTF-8: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen T1 video-level XLM-R transcript feature cache.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    if root.exists():
        raise SystemExit(f"refusing to overwrite existing cache root: {root}")
    if not args.audit_report.is_file():
        raise SystemExit(f"audit report is missing: {args.audit_report}")
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if hashlib.sha256(args.audit_report.read_bytes()).hexdigest() != AUDIT_SHA:
        raise SystemExit("TASK-003A audit SHA mismatch")
    rows, manifest_audit = build_manifest(args.data_root)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["corpus"], row["split"], row["video_id"])].append(row)
    encoder = XLMRVideoTranscriptEncoder(device="cpu")
    model_file = hf_hub_download(MODEL_NAME, filename="model.safetensors", revision=MODEL_REVISION)
    model_sha = sha256_file(model_file)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise SystemExit(f"model weight SHA mismatch: {model_sha}")
    root.mkdir(parents=True)
    entries = []
    for key in sorted(grouped):
        corpus, split, video_id = key
        transcript = args.data_root / corpus / f"{split}_labels" / video_id / f"{video_id}.txt"
        entry = {"corpus": corpus, "split": split, "video_id": video_id, "source_path": str(transcript), "available": False, "transcript_sha256": None, "chunk_count": 0, "feature_dim": 768, "artifact_path": None, "artifact_sha256": None, "encoder_name": MODEL_NAME, "requested_revision": MODEL_REVISION, "resolved_commit": encoder.resolved_commit, "model_weight_sha256": model_sha, "tokenizer_fingerprint": encoder.fingerprint()["tokenizer_init_fingerprint"], "max_model_length": 512, "content_tokens_per_chunk": 510, "overlap_stride": 0, "pooling_rule": "mean final hidden state over valid non-special tokens", "feature_dtype": "float32", "canonical_manifest_fingerprint": manifest_audit["manifest_fingerprint"]["value"], "task003a_report_sha256": AUDIT_SHA}
        if transcript.is_file() and transcript.stat().st_size > 0:
            text = _decode(transcript)
            features = encoder.encode(text)
            rel = Path("features") / corpus / split / f"{video_id}.pt"
            destination = root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": features, "dtype": "float32", "feature_dim": 768, "chunk_count": int(features.shape[0]), "pooling": "mean final hidden state over valid non-special tokens", "model_name": MODEL_NAME, "model_revision": MODEL_REVISION}, destination)
            entry.update({"available": True, "transcript_sha256": sha256_file(transcript), "chunk_count": int(features.shape[0]), "artifact_path": str(rel), "artifact_sha256": sha256_file(destination)})
        entries.append(entry)
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    metadata = {"version": VERSION, "cache_fingerprint": fingerprint, "encoder": encoder.fingerprint() | {"model_weight_sha256": model_sha}, "canonical_manifest_fingerprint": manifest_audit["manifest_fingerprint"]["value"], "task003a_report_sha256": AUDIT_SHA, "package_versions": {"python": platform.python_version(), "torch": torch.__version__}}
    index = {**metadata, "entries": entries}
    (root / "cache_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"entries": len(entries), "available": sum(x["available"] for x in entries), "index": str(root / "cache_index.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
