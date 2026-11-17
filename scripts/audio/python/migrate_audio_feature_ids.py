"""Safely migrate extracted audio representation filenames to owner-approved IDs."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

MIGRATIONS = {
    "deep": "id1",
    "deep_phon": "id2",
    "deep_egemaps_phon_artic": "id4",
}


def valid_payload(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    expected = {
        "shared_features": (1, 64),
        "depression_features": (1, 64),
        "pd_features": (1, 64),
        "logits": (2,),
        "probs": (2,),
    }
    if set(payload) != set(expected):
        raise ValueError("unexpected payload keys")
    for key, shape in expected.items():
        value = np.asarray(payload[key])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid {key}")
    return payload


def same_payload(left: Path, right: Path) -> bool:
    return all(
        np.array_equal(value, valid_payload(right)[key])
        for key, value in valid_payload(left).items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--wsm-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    by_corpus: dict[str, Counter[str]] = {"CARE": Counter(), "WSM": Counter()}
    for corpus, root in (("CARE", args.care_root), ("WSM", args.wsm_root)):
        for old_name, new_id in MIGRATIONS.items():
            for source in root.glob(f"extracted_unimodal/**/audio_{old_name}_fold*.pickle"):
                target = source.with_name(source.name.replace(f"audio_{old_name}_", f"audio_{new_id}_"))
                try:
                    valid_payload(source)
                except (OSError, ValueError, pickle.UnpicklingError):
                    counts["invalid_source"] += 1
                    by_corpus[corpus]["invalid_source"] += 1
                    continue
                if target.exists():
                    if same_payload(source, target):
                        counts["matching_target"] += 1
                        by_corpus[corpus]["matching_target"] += 1
                    else:
                        counts["conflict"] += 1
                        by_corpus[corpus]["conflict"] += 1
                    continue
                source.rename(target)
                counts["renamed"] += 1
                by_corpus[corpus]["renamed"] += 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"counts": counts, "by_corpus": by_corpus}, default=dict, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
