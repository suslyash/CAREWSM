#!/usr/bin/env python3
"""Build the frozen uniform-accepted-reliability RAMPS cache."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import torch


FROZEN_SOURCE_SHA256 = "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945"
EXPECTED_ROWS = 6325
EXPECTED_ACCEPTED = (376, 1801)
EXPECTED_POSITIVE = (376, 212)
EXPECTED_NEGATIVE = (0, 1589)


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


def assert_cache_invariants(source: dict[str, Any], derived: dict[str, Any]) -> tuple[int, int]:
    required = (
        "version", "task_names", "segment_ids", "observed_mask", "observed_targets",
        "pseudo_accept_mask", "pseudo_targets", "pseudo_class", "calibrated_audio_probs",
    )
    for field in required:
        if field not in source or field not in derived:
            raise ValueError(f"missing required cache field: {field}")
        if not tensor_equal(source[field], derived[field]):
            raise ValueError(f"derived cache changed non-reliability field: {field}")
    if source["version"] != "ramps-r2-semantic-v1":
        raise ValueError("unexpected canonical cache version")
    ids = source["segment_ids"]
    if len(ids) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS or ids != derived["segment_ids"]:
        raise ValueError("segment IDs are not canonical and unique")
    observed = source["observed_mask"].bool()
    accept = source["pseudo_accept_mask"].bool()
    targets = source["pseudo_targets"]
    calibrated = source["calibrated_audio_probs"]
    pseudo_class = source["pseudo_class"].long()
    if bool((accept & observed).any()):
        raise ValueError("accepted pseudo rows overlap observed truth")
    if not torch.equal(targets[accept], calibrated[accept]):
        raise ValueError("accepted pseudo targets differ from calibrated probabilities")
    accepted_counts = tuple(int(accept[:, task].sum()) for task in range(2))
    positive_counts = tuple(int((accept[:, task] & (pseudo_class[:, task] == 1)).sum()) for task in range(2))
    negative_counts = tuple(int((accept[:, task] & (pseudo_class[:, task] == 0)).sum()) for task in range(2))
    if accepted_counts != EXPECTED_ACCEPTED or positive_counts != EXPECTED_POSITIVE or negative_counts != EXPECTED_NEGATIVE:
        raise ValueError("accepted counts/classes do not match frozen contract")
    reliability = derived["pseudo_reliability"]
    if tuple(reliability.shape) != (EXPECTED_ROWS, 2):
        raise ValueError("pseudo_reliability has unexpected shape")
    if not bool(torch.all(reliability[accept] == 1.0)):
        raise ValueError("accepted reliability is not exactly 1.0")
    if not bool(torch.all(reliability[~accept] == 0.0)):
        raise ValueError("rejected/observed reliability is not exactly 0.0")
    return accepted_counts


def build(source_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing derived cache: {output_path}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha_before = sha256(source_path)
    if source_sha_before != FROZEN_SOURCE_SHA256:
        raise ValueError(f"source SHA mismatch before generation: {source_sha_before}")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise ValueError("source cache must contain a mapping")
    derived = copy.deepcopy(source)
    accept = source["pseudo_accept_mask"].bool()
    source_reliability = source["pseudo_reliability"].detach().cpu()
    summaries = []
    changed = []
    for task in range(2):
        values = source_reliability[:, task][accept[:, task]]
        summaries.append({"min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())})
        changed.append(int((values != 1.0).sum()))
    reliability = torch.zeros_like(source_reliability)
    reliability[accept] = 1.0
    derived["pseudo_reliability"] = reliability
    derived["reliability_ablation"] = {
        "type": "uniform_accepted_reliability",
        "source_cache_sha256": source_sha_before,
        "accepted_value": 1.0,
        "rejected_or_observed_value": 0.0,
        "source_accepted_reliability_summary_by_task": summaries,
    }
    assert_cache_invariants(source, derived)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(derived, output_path)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
    if not isinstance(reloaded, dict):
        raise ValueError("reloaded derived cache is not a mapping")
    assert_cache_invariants(source, reloaded)
    source_sha_after = sha256(source_path)
    if source_sha_after != FROZEN_SOURCE_SHA256:
        raise ValueError(f"source SHA changed after generation: {source_sha_after}")
    print(f"SOURCE_SHA256={source_sha_before}")
    print(f"DERIVED_SHA256={sha256(output_path)}")
    print(f"ACCEPTED_COUNTS_D_P={EXPECTED_ACCEPTED[0]}/{EXPECTED_ACCEPTED[1]}")
    print(f"POSITIVE_COUNTS_D_P={EXPECTED_POSITIVE[0]}/{EXPECTED_POSITIVE[1]}")
    print(f"NEGATIVE_COUNTS_D_P={EXPECTED_NEGATIVE[0]}/{EXPECTED_NEGATIVE[1]}")
    print(f"SOURCE_ACCEPTED_RELIABILITY_SUMMARY={summaries}")
    print(f"CHANGED_ACCEPTED_RELIABILITY_D_P={changed[0]}/{changed[1]}")
    print("CACHE_INVARIANTS_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
