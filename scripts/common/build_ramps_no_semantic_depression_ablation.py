#!/usr/bin/env python3
"""Build the no-semantic-enabled-depression pseudo-path RAMPS cache."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import torch

FROZEN_SOURCE_SHA256 = "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945"
ROWS = 6325
D_TASK = 0
P_TASK = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return left == right
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.is_floating_point():
        return bool(torch.equal(torch.isnan(left), torch.isnan(right)) and torch.all((left == right) | (torch.isnan(left) & torch.isnan(right))))
    return bool(torch.equal(left, right))


def assert_invariants(source: dict[str, Any], derived: dict[str, Any]) -> None:
    changed = {"pseudo_accept_mask", "pseudo_targets", "pseudo_reliability", "pseudo_class", "semantic_ablation"}
    for key, value in source.items():
        if key not in changed:
            if key not in derived or not tensor_equal(value, derived[key]):
                raise ValueError(f"field changed unexpectedly: {key}")
    if source["version"] != "ramps-r2-semantic-v1" or derived["version"] != source["version"]:
        raise ValueError("cache version changed")
    ids = source["segment_ids"]
    if len(ids) != ROWS or len(set(ids)) != ROWS or ids != derived["segment_ids"]:
        raise ValueError("segment IDs changed or are not unique")
    observed = source["observed_mask"].bool()
    source_accept = source["pseudo_accept_mask"].bool()
    derived_accept = derived["pseudo_accept_mask"].bool()
    if bool(derived_accept[:, D_TASK].any()) or not torch.equal(derived_accept[:, P_TASK], source_accept[:, P_TASK]):
        raise ValueError("acceptance transformation failed")
    if not bool(torch.isnan(derived["pseudo_targets"][:, D_TASK]).all()) or not tensor_equal(derived["pseudo_targets"][:, P_TASK], source["pseudo_targets"][:, P_TASK]):
        raise ValueError("pseudo target transformation failed")
    if not bool(torch.equal(derived["pseudo_reliability"][:, D_TASK], torch.zeros(ROWS, dtype=source["pseudo_reliability"].dtype))):
        raise ValueError("depression reliability is not zero")
    if not torch.equal(derived["pseudo_reliability"][:, P_TASK], source["pseudo_reliability"][:, P_TASK]):
        raise ValueError("Parkinson reliability changed")
    if not bool(torch.equal(derived["pseudo_class"][:, D_TASK], torch.full((ROWS,), -1, dtype=source["pseudo_class"].dtype))):
        raise ValueError("depression pseudo classes are not neutral")
    if not torch.equal(derived["pseudo_class"][:, P_TASK], source["pseudo_class"][:, P_TASK]):
        raise ValueError("Parkinson pseudo classes changed")
    if bool((derived_accept & observed).any()):
        raise ValueError("derived acceptance overlaps observed truth")
    if not torch.equal(derived["pseudo_targets"][derived_accept[:, P_TASK], P_TASK], source["calibrated_audio_probs"][derived_accept[:, P_TASK], P_TASK]):
        raise ValueError("Parkinson targets do not equal calibrated probabilities")
    if tuple(int(derived_accept[:, t].sum()) for t in range(2)) != (0, 1801):
        raise ValueError("accepted count contract failed")
    if tuple(int((derived_accept[:, t] & (derived["pseudo_class"][:, t] == 1)).sum()) for t in range(2)) != (0, 212):
        raise ValueError("positive count contract failed")
    if tuple(int((derived_accept[:, t] & (derived["pseudo_class"][:, t] == 0)).sum()) for t in range(2)) != (0, 1589):
        raise ValueError("negative count contract failed")


def build(source_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    source_before = sha256(source_path)
    if source_before != FROZEN_SOURCE_SHA256:
        raise ValueError(f"source SHA mismatch: {source_before}")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise ValueError("source cache must be a mapping")
    derived = copy.deepcopy(source)
    accept = source["pseudo_accept_mask"].bool()
    removed = int(accept[:, D_TASK].sum())
    retained = int(accept[:, P_TASK].sum())
    derived["pseudo_accept_mask"][:, D_TASK] = False
    derived["pseudo_targets"][:, D_TASK] = float("nan")
    derived["pseudo_reliability"][:, D_TASK] = 0.0
    derived["pseudo_class"][:, D_TASK] = -1
    derived["semantic_ablation"] = {
        "type": "remove_semantic_enabled_depression_pseudo_path",
        "source_cache_sha256": source_before,
        "removed_depression_accepted_count": removed,
        "retained_parkinson_accepted_count": retained,
        "historical_basis": "pre_semantic_depression_not_deployable_parkinson_frozen",
    }
    assert_invariants(source, derived)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(derived, output_path)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
    if not isinstance(reloaded, dict):
        raise ValueError("reloaded cache is not a mapping")
    assert_invariants(source, reloaded)
    source_after = sha256(source_path)
    if source_after != FROZEN_SOURCE_SHA256:
        raise ValueError(f"source SHA changed: {source_after}")
    print(f"SOURCE_SHA256={source_before}")
    print(f"DERIVED_SHA256={sha256(output_path)}")
    print(f"REMOVED_DEPRESSION_ACCEPTED={removed}")
    print(f"RETAINED_PARKINSON_ACCEPTED={retained}")
    print("CACHE_INVARIANTS_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
