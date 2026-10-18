#!/usr/bin/env python3
"""Cache the official DiariZen model and WeSpeaker embedding weights.

This script intentionally performs network access only when the user runs it.
It never reads or writes CARE/WSM corpus paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_REPO = "BUT-FIT/diarizen-wavlm-large-s80-md"
EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBEDDING_FILENAME = "pytorch_model.bin"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face cache root; defaults to HF_HOME/HF cache resolution",
    )
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    cache_dir = str(args.cache_dir.expanduser().resolve()) if args.cache_dir else None
    model_path = snapshot_download(
        repo_id=MODEL_REPO,
        revision=args.revision,
        cache_dir=cache_dir,
    )
    embedding_path = hf_hub_download(
        repo_id=EMBEDDING_REPO,
        filename=EMBEDDING_FILENAME,
        revision=args.revision,
        cache_dir=cache_dir,
    )
    print(
        json.dumps(
            {
                "model_repo": MODEL_REPO,
                "model_revision": args.revision,
                "model_dir": str(Path(model_path).resolve()),
                "embedding_repo": EMBEDDING_REPO,
                "embedding_filename": EMBEDDING_FILENAME,
                "embedding_revision": args.revision,
                "embedding_path": str(Path(embedding_path).resolve()),
                "cache_dir": cache_dir or "HF_HOME/default Hugging Face cache",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
