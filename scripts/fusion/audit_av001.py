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
IDENTITY_FIELDS = ("profile_id", "video_id", "label_id", "subgroup")


class AuditError(RuntimeError):
    """Raised when canonical identity or split invariants are violated."""


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def row_identity(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in IDENTITY_FIELDS)


def source_index_from_rows(
    rows: list[dict[str, str]], corpus: str
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, ...], str]]:
    by_sample = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            raise AuditError(f"{corpus}: BLOCKED source row without sample_id")
        previous = by_sample.get(sample_id)
        if previous is not None and row_identity(previous) != row_identity(row):
            raise AuditError(
                f"{corpus}: BLOCKED conflicting repeated sample_id metadata"
            )
        by_sample[sample_id] = row

    by_pair = {}
    for sample_id, row in by_sample.items():
        by_pair.setdefault((row["profile_id"], row["video_id"]), set()).add(sample_id)
    if any(len(sample_ids) > 1 for sample_ids in by_pair.values()):
        raise AuditError(f"{corpus}: BLOCKED ambiguous profile/video pairs")

    identity_index = {}
    for sample_id, row in by_sample.items():
        identity = row_identity(row)
        if identity in identity_index and identity_index[identity] != sample_id:
            raise AuditError(f"{corpus}: BLOCKED ambiguous artifact identity")
        identity_index[identity] = sample_id
    return by_sample, identity_index


def source_index(
    root: Path, corpus: str
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, ...], str]]:
    rows = []
    for name in ("train.csv", "dev.csv", "test.csv"):
        rows.extend(rows_from_csv(root / "splits" / name))
    return source_index_from_rows(rows, corpus)


def validate_population(
    rows: list[dict[str, str]],
    sources: dict[str, dict[str, str]],
    corpus: str,
    split_name: str,
) -> set[str]:
    sample_ids = set()
    for row in rows:
        sample_id = row.get("sample_id", "")
        if sample_id in sample_ids:
            raise AuditError(f"{corpus}: BLOCKED duplicate {split_name} sample_id")
        sample_ids.add(sample_id)
        source = sources.get(sample_id)
        if source is None:
            raise AuditError(
                f"{corpus}: BLOCKED {split_name} sample_id absent from source"
            )
        if row_identity(source) != row_identity(row):
            raise AuditError(f"{corpus}: BLOCKED {split_name} metadata mismatch")
    return sample_ids


def fold_rows(
    root: Path,
    fold: int,
    split: str,
    sources: dict[str, dict[str, str]],
    corpus: str,
) -> set[str]:
    path = root / "splits" / "folds" / f"fold_{fold}_{split}.csv"
    return validate_population(
        rows_from_csv(path), sources, corpus, f"fold {fold} {split}"
    )


def fixed_test(
    root: Path, sources: dict[str, dict[str, str]], corpus: str
) -> set[str]:
    return validate_population(
        rows_from_csv(root / "splits" / "test.csv"), sources, corpus, "fixed TEST"
    )


def artifact_identity(path: Path, corpus: str) -> tuple[str, ...]:
    label, subgroup, profile, video = path.parts[-5:-1]
    if corpus == "WSM":
        profile = profile.removeprefix("PROFILE_")
        video = video.removeprefix("VIDEO_")
    return profile, video, label, subgroup


def artifact_sets(
    root: Path,
    corpus: str,
    identity_index: dict[tuple[str, ...], str],
) -> tuple[dict[str, dict[str, dict[int, set[str]]]], int]:
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
                fold_text = path.stem.removeprefix(f"{prefix}_fold")
                if not fold_text.isdigit() or int(fold_text) not in FOLDS:
                    continue
                sample_id = identity_index.get(artifact_identity(path, corpus))
                if sample_id is None:
                    raise AuditError(
                        f"{corpus}: BLOCKED artifact identity without source"
                    )
                fold = int(fold_text)
                seen[(fold, sample_id)] += 1
                available[family][prefix][fold].add(sample_id)
            duplicate_count += sum(count - 1 for count in seen.values() if count > 1)
    if duplicate_count:
        raise AuditError(f"{corpus}: BLOCKED duplicate artifact-to-sample mappings")
    return available, duplicate_count


def check_profile_disjointness(
    populations: dict[int, dict[str, set[str]]],
    sources: dict[str, dict[str, str]],
    corpus: str,
) -> list[int]:
    intersections = []
    for splits in populations.values():
        profiles = {
            split: {sources[sample_id]["profile_id"] for sample_id in sample_ids}
            for split, sample_ids in splits.items()
        }
        intersections.extend(
            len(profiles[first] & profiles[second])
            for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
        )
    if any(intersections):
        raise AuditError(f"{corpus}: BLOCKED nonzero profile intersection")
    return intersections


def audit_corpus(root: Path, corpus: str) -> None:
    sources, identity_index = source_index(root, corpus)
    artifacts, duplicate_count = artifact_sets(root, corpus, identity_index)
    test_ids = fixed_test(root, sources, corpus)
    populations = {
        fold: {
            "train": fold_rows(root, fold, "train", sources, corpus),
            "val": fold_rows(root, fold, "val", sources, corpus),
            "test": test_ids,
        }
        for fold in FOLDS
    }
    profile_intersections = check_profile_disjointness(populations, sources, corpus)
    artifact_ids = set().union(
        *(
            prefix_sets[fold]
            for family_sets in artifacts.values()
            for prefix_sets in family_sets.values()
            for fold in FOLDS
        )
    )
    print(
        f"{corpus}: source_unique={len(sources)} "
        f"artifact_sample_ids={len(artifact_ids)} "
        f"artifact_ids_without_source={len(artifact_ids - set(sources))} "
        f"duplicate_artifact_keys={duplicate_count}"
    )
    print(f"{corpus}: profile_intersections={profile_intersections}")
    for fold in FOLDS:
        for split, population in populations[fold].items():
            availability = {
                modality: set.intersection(
                    *(family_sets[prefix][fold] for prefix in FAMILIES[modality])
                )
                & population
                for modality, family_sets in artifacts.items()
            }
            complete = set.intersection(*availability.values())
            exclusion = (
                len(population - availability["audio"]),
                len(population - availability["video"]),
                len(population - availability["text"]),
                len(
                    (availability["audio"] & availability["video"])
                    - availability["text"]
                ),
                len(
                    (availability["audio"] & availability["text"])
                    - availability["video"]
                ),
                len(
                    (availability["video"] & availability["text"])
                    - availability["audio"]
                ),
                len(
                    population
                    - availability["audio"]
                    - availability["video"]
                    - availability["text"]
                ),
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
            print(
                f"{corpus}: representative_schema=UNKNOWN "
                f"missing_dependency={error.name}"
            )
        else:
            print(
                f"{corpus}: representative_keys={sorted(payload)} "
                f"feature_shape={getattr(payload['shared_features'], 'shape', None)} "
                f"feature_dtype={getattr(payload['shared_features'], 'dtype', None)}"
            )


def expect_blocked(callback) -> None:
    try:
        callback()
    except AuditError:
        return
    raise AssertionError("synthetic invalid identity was accepted")


def synthetic_regression() -> None:
    base = {
        "sample_id": "s1",
        "profile_id": "p1",
        "video_id": "v1",
        "label_id": "CONTROL",
        "subgroup": "g1",
    }
    expect_blocked(
        lambda: source_index_from_rows(
            [base, {**base, "sample_id": "s2"}], "SYNTH"
        )
    )
    sources, _ = source_index_from_rows([base], "SYNTH")
    expect_blocked(
        lambda: validate_population(
            [{**base, "sample_id": "missing"}], sources, "SYNTH", "fold"
        )
    )
    expect_blocked(
        lambda: validate_population(
            [{**base, "profile_id": "wrong"}], sources, "SYNTH", "fold"
        )
    )
    expect_blocked(
        lambda: check_profile_disjointness(
            {0: {"train": {"s1"}, "val": {"s1"}, "test": set()}},
            sources,
            "SYNTH",
        )
    )
    print("synthetic_regression=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-root", type=Path)
    parser.add_argument("--wsm-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        synthetic_regression()
        return 0
    if args.care_root is None or args.wsm_root is None:
        parser.error(
            "--care-root and --wsm-root are required unless --self-test is used"
        )
    try:
        audit_corpus(args.care_root, "CARE")
        audit_corpus(args.wsm_root, "WSM")
    except AuditError as error:
        print(str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
