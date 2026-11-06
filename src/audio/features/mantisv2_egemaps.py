"""Build fingerprinted frozen MantisV2 features from audited eGeMAPS LLD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_REVISION = "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
CHECKPOINT_SHA256 = "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
FEATURE_FAMILY = "mantisv2_egemaps"
FEATURE_DIMENSION = 12_800

def _load_temporalizer() -> Any:
    path = Path(__file__).parents[3] / "scripts" / "smoke_mantisv2_egemaps_train.py"
    spec = importlib.util.spec_from_file_location("mantis_temporalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load accepted temporalizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _cache_name(corpus: str, sample_id: str) -> str:
    return f"{corpus}__{hashlib.sha256(sample_id.encode()).hexdigest()[:20]}.npz"

def _prepare_sample(
    corpus: str,
    root: Path,
    row: dict[str, str],
    temporalizer: Any,
    validator: Any,
    channels: list[str],
) -> dict[str, Any] | None:
    path = validator.artifact_path(root, row)
    frames, excluded, reasons = temporalizer.artifact_frames(path, channels, validator)
    windows = temporalizer.temporalize(frames, len(channels))
    if not windows:
        return None
    return {
        "corpus": corpus,
        "sample_id": row["sample_id"],
        "split": row.get("split", ""),
        "windows": windows,
        "policy_c_excluded_frames": excluded,
        "policy_c_exclusion_reasons": reasons,
        "source_frame_rows": len(frames) + excluded,
    }

def _infer_batch(
    pending: list[dict[str, Any]],
    cache_dir: Path,
    device: str,
    trainer: Any,
) -> list[dict[str, Any]]:
    uncached = [
        item
        for item in pending
        if not (cache_dir / _cache_name(item["corpus"], item["sample_id"])).exists()
    ]
    if uncached:
        tensors = [window["tensor"] for item in uncached for window in item["windows"]]
        inputs = torch.from_numpy(np.stack(tensors)).to(device=device, dtype=torch.float32)
        embeddings = []
        with torch.no_grad():
            for batch_start in range(0, len(inputs), 32):
                batch = inputs[batch_start : batch_start + 32]
                embeddings.append(trainer.transform(batch, three_dim=True, to_numpy=False))
        all_embeddings = torch.cat(embeddings, dim=0)
        offset = 0
        for item in uncached:
            count = len(item["windows"])
            pooled = all_embeddings[offset : offset + count].mean(dim=0)
            features = pooled.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
            if features.shape != (1, FEATURE_DIMENSION) or not np.isfinite(features).all():
                raise ValueError(f"unexpected Mantis feature output for {item['corpus']}/{item['sample_id']}")
            output = cache_dir / _cache_name(item["corpus"], item["sample_id"])
            np.savez_compressed(output, features=features)
            offset += count
    return [
        {key: value for key, value in item.items() if key != "windows"} | {
            "path": str((cache_dir / _cache_name(item["corpus"], item["sample_id"])).resolve()),
            "status": "VALID",
            "windows": len(item["windows"]),
        }
        for item in pending
    ]

def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    temporalizer = _load_temporalizer()
    validator = temporalizer.load_validator()
    channels = validator.registry_channels(args.registry)
    if len(channels) != 25:
        raise ValueError(f"expected 25 canonical eGeMAPS channels, got {len(channels)}")
    checkpoint_file = args.checkpoint / "model.safetensors"
    if _sha256(checkpoint_file) != CHECKPOINT_SHA256:
        raise ValueError("Mantis checkpoint fingerprint does not match the pinned revision")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {(row["corpus"], row["sample_id"]): row for row in _read_rows(args.manifest)}
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer
    network = MantisV2(device=args.device, return_transf_layer=2, output_token="combined").from_pretrained(str(args.checkpoint))
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    trainer = MantisTrainer(device=args.device, network=network)
    records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for corpus, root in (("CARE", args.care_root), ("WSM", args.wsm_root)):
        split_rows = [
            _read_rows(root / "splits" / "folds" / "fold_0_train.csv"),
            _read_rows(root / "splits" / "folds" / "fold_0_val.csv"),
            _read_rows(root / "splits" / "test.csv"),
        ]
        for rows in split_rows:
            for row in rows:
                source = manifest.get((corpus, row["sample_id"]))
                if source is None or source.get("egemaps_lld_status") != "VALID":
                    continue
                prepared = _prepare_sample(corpus, root, row, temporalizer, validator, channels)
                if prepared is None:
                    continue
                pending.append(prepared)
                if len(pending) == 32:
                    records.extend(_infer_batch(pending, args.output, args.device, trainer))
                    pending = []
    if pending:
        records.extend(_infer_batch(pending, args.output, args.device, trainer))
    metadata = {
        "feature_family": FEATURE_FAMILY,
        "feature_dimension": FEATURE_DIMENSION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_layer": 2,
        "output_token": "combined",
        "pooling": "mean_windows",
        "source_feature": "egemaps_lld",
        "channels": channels,
        "window_seconds": 1.0,
        "hop_seconds": 0.5,
        "min_consecutive_valid_frames": 95,
        "max_windows_per_recording": 2,
        "policy_c": "exclude terminal out-of-turn one-sample frames",
        "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("reports/feature_audit/sample_feature_availability.csv"))
    parser.add_argument("--care-root", type=Path, default=Path("/media/maxim/Databases/CARE"))
    parser.add_argument("--wsm-root", type=Path, default=Path("/media/maxim/Databases/WSM_NEW"))
    parser.add_argument("--registry", type=Path, default=Path("/media/maxim/Databases/CARE/splits/features.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    metadata = build_cache(args)
    print(json.dumps({"records": len(metadata["records"]), "output": str(args.output)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
