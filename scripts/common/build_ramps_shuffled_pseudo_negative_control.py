#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import torch

PERMUTED_FIELDS = (
    "pseudo_accept_mask",
    "pseudo_targets",
    "pseudo_reliability",
    "pseudo_class",
    "calibrated_audio_probs",
)
EXPECTED_SOURCE_SHA = "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945"
EXPECTED_ROWS = 6325
EXPECTED_MISSING = (2665, 3660)
EXPECTED_ACCEPTED = (376, 1801)
EXPECTED_CLASS_COUNTS = ((376, 0), (212, 1589))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def tuple_counter(cache: dict[str, Any], task: int, rows: torch.Tensor) -> Counter[bytes]:
    result: Counter[bytes] = Counter()
    for row in rows.tolist():
        parts = [tensor_bytes(cache[field][row, task].reshape(-1)) for field in PERMUTED_FIELDS]
        result[b"".join(parts)] += 1
    return result


def permutation_sha(destination_order: torch.Tensor, source_order: torch.Tensor) -> str:
    pairs = torch.stack([destination_order, source_order], dim=1).to(torch.int64).contiguous()
    return hashlib.sha256(pairs.numpy().tobytes()).hexdigest()


def assert_cache_invariants(
    source: dict[str, Any],
    derived: dict[str, Any],
    source_sha: str,
    derived_sha: str | None = None,
    permutations: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    assert source_sha == EXPECTED_SOURCE_SHA, source_sha
    assert len(source["segment_ids"]) == EXPECTED_ROWS
    assert len(derived["segment_ids"]) == EXPECTED_ROWS
    assert source["segment_ids"] == derived["segment_ids"]
    for field in ("observed_mask", "observed_targets"):
        assert tensor_bytes(source[field]) == tensor_bytes(derived[field]), field

    observed = source["observed_mask"]
    accept = source["pseudo_accept_mask"]
    derived_accept = derived["pseudo_accept_mask"]
    missing_counts, accepted_counts, class_counts = [], [], []
    hamming, fixed_points, permutation_hashes = [], [], {}
    for task, task_name in enumerate(("depression", "parkinson")):
        missing = (~observed[:, task]).nonzero(as_tuple=False).flatten()
        missing_counts.append(int(missing.numel()))
        assert int(missing.numel()) == EXPECTED_MISSING[task]
        assert not bool((observed[:, task] & derived_accept[:, task]).any())
        accepted = derived_accept[missing, task]
        accepted_counts.append(int(accepted.sum().item()))
        assert int(accepted.sum().item()) == EXPECTED_ACCEPTED[task]
        pos = int((derived["pseudo_class"][missing, task][accepted] == 1).sum().item())
        neg = int((derived["pseudo_class"][missing, task][accepted] == 0).sum().item())
        class_counts.append((pos, neg))
        assert (pos, neg) == EXPECTED_CLASS_COUNTS[task]
        hamming.append(int((accept[missing, task] != derived_accept[missing, task]).sum().item()))
        if permutations is not None:
            destination_order, source_order = permutations[task_name]
            assert destination_order.shape == missing.shape
            assert source_order.shape == missing.shape
            assert torch.equal(torch.sort(destination_order).values, torch.sort(missing).values)
            assert torch.equal(torch.sort(source_order).values, torch.sort(missing).values)
            assert torch.equal(source_order, torch.roll(destination_order, shifts=1, dims=0))
            assert torch.all(destination_order != source_order)
            fixed_points.append(int((destination_order == source_order).sum().item()))
            assert fixed_points[-1] == 0
            permutation_hashes[task_name] = permutation_sha(destination_order, source_order)
        assert tuple_counter(source, task, missing) == tuple_counter(derived, task, missing)
        rejected = ~derived_accept[:, task]
        assert torch.isnan(derived["pseudo_targets"][rejected, task]).all()
        assert torch.equal(derived["pseudo_reliability"][rejected, task], torch.zeros_like(derived["pseudo_reliability"][rejected, task]))
        assert torch.equal(derived["pseudo_class"][rejected, task], torch.full_like(derived["pseudo_class"][rejected, task], -1))
        accepted_rows = (~derived["observed_mask"][:, task]) & derived_accept[:, task]
        assert torch.isfinite(derived["pseudo_targets"][accepted_rows, task]).all()
        assert torch.isfinite(derived["pseudo_reliability"][accepted_rows, task]).all()
        assert ((derived["pseudo_reliability"][accepted_rows, task] >= 0) & (derived["pseudo_reliability"][accepted_rows, task] <= 1)).all()
        assert torch.equal(derived["pseudo_targets"][accepted_rows, task], derived["calibrated_audio_probs"][accepted_rows, task])
    assert tuple(missing_counts) == EXPECTED_MISSING
    assert tuple(accepted_counts) == EXPECTED_ACCEPTED
    assert tuple(class_counts) == EXPECTED_CLASS_COUNTS
    assert int((~derived["observed_mask"] & derived["pseudo_accept_mask"]).sum().item()) == sum(EXPECTED_ACCEPTED)
    if derived_sha is not None:
        assert derived.get("negative_control", {}).get("source_cache_sha256") == source_sha
    return {
        "source_sha256": source_sha,
        "derived_sha256": derived_sha,
        "missing_counts": missing_counts,
        "accepted_counts": accepted_counts,
        "class_counts": class_counts,
        "acceptance_hamming_differences": hamming,
        "fixed_points": fixed_points,
        "permutation_sha256": permutation_hashes,
    }


def build(source_path: Path, output_path: Path, shuffle_seed: int) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    source_sha = sha256_file(source_path)
    assert source_sha == EXPECTED_SOURCE_SHA, source_sha
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    assert isinstance(source, dict)
    assert len(source["segment_ids"]) == EXPECTED_ROWS
    derived = copy.deepcopy(source)
    permutations: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for task, task_name in enumerate(("depression", "parkinson")):
        missing = (~source["observed_mask"][:, task]).nonzero(as_tuple=False).flatten()
        generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed + task))
        order = missing[torch.randperm(missing.numel(), generator=generator)]
        source_order = torch.roll(order, shifts=1, dims=0)
        assert torch.all(order != source_order)
        permutations[task_name] = (order, source_order)
        for field in PERMUTED_FIELDS:
            derived[field][order, task] = source[field][source_order, task].clone()
    audit = assert_cache_invariants(source, derived, source_sha, permutations=permutations)
    derived["negative_control"] = {
        "type": "within_task_missing_row_tuple_derangement",
        "source_cache_sha256": source_sha,
        "shuffle_seed": int(shuffle_seed),
        "depression_permutation_sha256": audit["permutation_sha256"]["depression"],
        "parkinson_permutation_sha256": audit["permutation_sha256"]["parkinson"],
        "permuted_fields": list(PERMUTED_FIELDS),
        "fixed_points": audit["fixed_points"],
        "acceptance_assignment_hamming_differences": audit["acceptance_hamming_differences"],
        "mapping": "destination_order_to_roll1_source_order",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(derived, output_path)
    assert sha256_file(source_path) == EXPECTED_SOURCE_SHA
    derived_sha = sha256_file(output_path)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
    post = assert_cache_invariants(source, reloaded, source_sha, derived_sha=derived_sha, permutations=permutations)
    post["derived_sha256"] = derived_sha
    post["source_sha256_after_generation"] = sha256_file(source_path)
    assert post["source_sha256_after_generation"] == EXPECTED_SOURCE_SHA
    return post


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shuffle-seed", required=True, type=int)
    args = parser.parse_args()
    result = build(args.source, args.output, args.shuffle_seed)
    print("RAMPS_SHUFFLED_NEGATIVE_CONTROL_PASS")
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
