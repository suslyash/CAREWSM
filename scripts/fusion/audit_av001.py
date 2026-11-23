"""Read-only reproducibility audit for the AV-001 legacy artifacts."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path


FOLDS = range(5)
FAMILIES = {
    "audio": ("audio_id1", "audio_id2", "audio_id4"),
    "video": ("video_4",),
    "text": ("text_id9", "text_id10"),
}


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def source_rows(root: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = []
    for name in ("train.csv", "dev.csv", "test.csv"):
        rows.extend(rows_from_csv(root / "splits" / name))
    result = {}
    for row in rows:
        key = (row["profile_id"], row["video_id"])
        result.setdefault(key, row)
    return result


def artifact_key(path: Path, corpus: str) -> tuple[str, str]:
    profile, video = path.parts[-3:-1]
    if corpus == "CARE":
        return profile, video
    return profile.removeprefix("PROFILE_"), video.removeprefix("VIDEO_")


def artifact_sets(root: Path, corpus: str) -> tuple[dict[str, dict[str, dict[int, set]]], int]:
    available = {
        family: {prefix: {fold: set() for fold in FOLDS} for prefix in prefixes}
        for family, prefixes in FAMILIES.items()
    }
    duplicate_count = 0
    for family, prefixes in FAMILIES.items():
        for prefix in prefixes:
            seen = Counter()
            for path in root.glob(
                f"extracted_unimodal/*/*/PROFILE_*/VIDEO_*/{prefix}_fold*.pickle"
            ):
                stem = path.stem
                marker = f"{prefix}_fold"
                if not stem.startswith(marker):
                    continue
                fold_text = stem.removeprefix(marker)
                if not fold_text.isdigit() or int(fold_text) not in FOLDS:
                    continue
                key = artifact_key(path, corpus)
                seen[(family, int(fold_text), key)] += 1
                available[family][prefix][int(fold_text)].add(key)
            duplicate_count += sum(count - 1 for count in seen.values() if count > 1)
    return available, duplicate_count


def fold_rows(root: Path, fold: int, split: str) -> set[tuple[str, str]]:
    path = root / "splits" / "folds" / f"fold_{fold}_{split}.csv"
    return {(row["profile_id"], row["video_id"]) for row in rows_from_csv(path)}


def fixed_test(root: Path) -> set[tuple[str, str]]:
    return {
        (row["profile_id"], row["video_id"])
        for row in rows_from_csv(root / "splits" / "test.csv")
    }


def audit_corpus(root: Path, corpus: str) -> None:
    sources = source_rows(root)
    artifacts, duplicate_count = artifact_sets(root, corpus)
    populations = {
        fold: {
            "train": fold_rows(root, fold, "train"),
            "val": fold_rows(root, fold, "val"),
            "test": fixed_test(root),
        }
        for fold in FOLDS
    }
    profile_intersections = []
    for fold, splits in populations.items():
        profile_sets = {
            split: {profile for profile, _ in rows}
            for split, rows in splits.items()
        }
        profile_intersections.extend(
            len(profile_sets[first] & profile_sets[second])
            for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
        )

    artifact_keys = set().union(
        *(
            prefix_sets[fold]
            for family_sets in artifacts.values()
            for prefix_sets in family_sets.values()
            for fold in FOLDS
        )
    )
    print(
        f"{corpus}: source_unique={len(sources)} "
        f"artifact_keys={len(artifact_keys & set(sources))} "
        f"artifact_keys_without_source={len(artifact_keys - set(sources))} "
        f"duplicate_artifact_keys={duplicate_count}"
    )
    print(f"{corpus}: profile_intersections={profile_intersections}")
    for fold in FOLDS:
        for split, population in populations[fold].items():
            availability = {}
            for modality, family_sets in artifacts.items():
                per_family = [
                    family_sets[prefix][fold]
                    for prefix in FAMILIES[modality]
                ]
                availability[modality] = set.intersection(*per_family) & population
            complete = set.intersection(*availability.values())
            exclusion = (
                len(population - availability["audio"]),
                len(population - availability["video"]),
                len(population - availability["text"]),
                len((availability["audio"] & availability["video"]) - availability["text"]),
                len((availability["audio"] & availability["text"]) - availability["video"]),
                len((availability["video"] & availability["text"]) - availability["audio"]),
                len(population - availability["audio"] - availability["video"] - availability["text"]),
            )
            print(
                f"{corpus} fold={fold} split={split} population={len(population)} "
                f"A={len(availability['audio'])} V={len(availability['video'])} "
                f"T={len(availability['text'])} AVT={len(complete)} "
                f"exclusion={exclusion}"
            )

    representative = next(
        root.glob("extracted_unimodal/*/*/PROFILE_*/VIDEO_*/audio_id1_fold0.pickle"),
        None,
    )
    if representative is not None:
        try:
            with representative.open("rb") as handle:
                payload = pickle.load(handle)
        except ModuleNotFoundError as error:
            print(f"{corpus}: representative_schema=UNKNOWN missing_dependency={error.name}")
        else:
            print(
                f"{corpus}: representative_keys={sorted(payload)} "
                f"feature_shape={getattr(payload['shared_features'], 'shape', None)} "
                f"feature_dtype={getattr(payload['shared_features'], 'dtype', None)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--wsm-root", type=Path, required=True)
    args = parser.parse_args()
    audit_corpus(args.care_root, "CARE")
    audit_corpus(args.wsm_root, "WSM")


if __name__ == "__main__":
    main()
