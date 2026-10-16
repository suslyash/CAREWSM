#!/usr/bin/env python3
"""Audit WSM manifest sources without changing the dataset."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

CORPORA = ("depression", "parkinson")
SPLITS = ("train", "dev", "test")
PRIORITY = {"train": 0, "dev": 1, "test": 2}
REQUIRED = ("video_id", "diagnosis", "segment_file")
BAD_SEGMENTS = {
    ("depression", "dev", "3s14gCKn-yA", "3s14gCKn-yA_005.mp4"),
    ("depression", "dev", "49LbNLMWZwM", "49LbNLMWZwM_001.mp4"),
    ("depression", "dev", "4QIa1kSG45A", "4QIa1kSG45A_001.mp4"),
}
SPEAKER_KEYS = {"speaker_id", "speaker", "subject_id", "subject", "participant_id", "participant", "person_id", "person"}
FIELDS = (
    "segment_id", "video_id", "speaker_id", "corpus", "split",
    "y_depression", "y_parkinson", "observed_depression", "observed_parkinson",
    "audio_available", "video_available", "text_available", "description_available",
)


class AuditError(RuntimeError):
    pass


def index_path(root: Path, corpus: str, split: str) -> Path:
    suffix = "_mishas" if split == "test" else ""
    return root / corpus / f"{split}_labels_segments_min_filtered{suffix}.csv"


def read_index(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    if not path.is_file():
        raise AuditError(f"missing required source: {path}")
    delimiter = ";" if path.stem.endswith("_mishas") else ","
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise AuditError(f"source has no header: {path}")
        columns = [str(x).strip() for x in reader.fieldnames]
        if len(columns) != len(set(columns)):
            raise AuditError(f"duplicate columns in {path}")
        missing = [x for x in REQUIRED if x not in columns]
        if missing:
            raise AuditError(f"{path} missing columns: {missing}")
        rows = []
        for line, raw in enumerate(reader, 2):
            row = {str(k).strip(): (str(v).strip() if v is not None else "") for k, v in raw.items()}
            if any(not row.get(k) for k in REQUIRED):
                raise AuditError(f"{path}:{line} has an empty required field")
            try:
                diagnosis = int(row["diagnosis"])
            except ValueError as exc:
                raise AuditError(f"{path}:{line} diagnosis is not an integer") from exc
            if diagnosis not in (0, 1):
                raise AuditError(f"{path}:{line} diagnosis is not 0/1")
            rows.append(row)
    return columns, rows, delimiter


def collect_keys(value: Any, keys: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).strip().casefold())
            collect_keys(child, keys)
    elif isinstance(value, list):
        for child in value:
            collect_keys(child, keys)


def speaker_source(root: Path) -> dict[str, Any]:
    inspected = 0
    matches = []
    keys_seen: set[str] = set()
    for corpus in CORPORA:
        for split in SPLITS:
            folder = root / corpus / f"{split}_labels"
            for path in sorted(folder.glob("*/*.json")) if folder.is_dir() else []:
                inspected += 1
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise AuditError(f"cannot inspect {path}: {exc}") from exc
                keys: set[str] = set()
                collect_keys(payload, keys)
                keys_seen.update(keys)
                found = sorted(keys & SPEAKER_KEYS)
                if found:
                    matches.append({"file": str(path), "keys": found})
    resolved = bool(matches)
    return {
        "status": "resolved" if resolved else "unresolved",
        "authoritative_files": matches,
        "candidate_keys_seen": sorted(keys_seen & SPEAKER_KEYS),
        "files_inspected": inspected,
        "fallback_to_video_id": False,
        "reason": "explicit speaker source found; review required" if resolved else
        "no explicit speaker/subject/participant identifier in raw indexes or adjacent JSON metadata",
    }


def overlaps(by_split: dict[str, set[str]]) -> dict[str, Any]:
    pairs = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        values = sorted(by_split[a] & by_split[b])
        pairs[f"{a}_vs_{b}"] = {"count": len(values), "values": values}
    return {
        "by_split": {k: sorted(v) for k, v in by_split.items()},
        "pairwise": pairs,
        "any_overlap": any(x["count"] for x in pairs.values()),
        "unique_values": len(set().union(*by_split.values())),
    }


def modality_audit(records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    rules = {
        "audio_available": "root/corpus/{split}_labels/{video_id}/segments/{segment_stem}.wav",
        "video_available": "root/corpus/{split}_labels/{video_id}/segments/{segment_file}",
        "text_available": "root/corpus/{split}_labels/{video_id}/{video_id}.txt; video-level transcript",
        "description_available": None,
    }
    counts = {k: {"true": 0, "false": 0} for k in rules}
    examples = defaultdict(list)
    for row in records:
        folder = root / row["corpus"] / f"{row['source_split']}_labels" / row["video_id"]
        segment = Path(row["segment_file"])
        paths = {
            "audio_available": folder / "segments" / f"{segment.stem}.wav",
            "video_available": folder / "segments" / row["segment_file"],
            "text_available": folder / f"{row['video_id']}.txt",
            "description_available": None,
        }
        for field, path in paths.items():
            present = bool(path and path.is_file() and path.stat().st_size > 0)
            counts[field]["true" if present else "false"] += 1
            if not present and len(examples[field]) < 5:
                examples[field].append(str(path) if path else None)
    out = {}
    for field, rule in rules.items():
        out[field] = {
            "status": "unresolved" if field == "description_available" else "resolved",
            "path_rule": rule,
            "availability_counts": counts[field],
            "missing_examples": examples[field],
        }
        if field == "description_available":
            out[field]["notes"] = "no authoritative description or semantic-feature source exists"
    return out


def build_audit(root: Path) -> dict[str, Any]:
    records = []
    source_files = OrderedDict()
    raw_columns = OrderedDict()
    source_audit = OrderedDict()
    for corpus in CORPORA:
        for split in SPLITS:
            path = index_path(root, corpus, split)
            columns, rows, delimiter = read_index(path)
            key = f"{corpus}/{split}"
            source_files[key] = str(path)
            raw_columns[key] = columns
            usable = []
            for row in rows:
                item = (corpus, split, row["video_id"], row["segment_file"])
                if item in BAD_SEGMENTS:
                    continue
                usable.append(row)
                records.append({
                    "corpus": corpus, "source_split": split,
                    "video_id": row["video_id"], "segment_file": row["segment_file"],
                    "diagnosis": int(row["diagnosis"]),
                })
            source_audit[key] = {
                "source_file": str(path), "delimiter": delimiter, "raw_columns": columns,
                "raw_row_count": len(rows),
                "excluded_by_existing_bad_segment_rules": len(rows) - len(usable),
                "post_existing_filter_row_count": len(usable),
                "post_existing_filter_unique_video_id_count": len({x["video_id"] for x in usable}),
            }

    candidate_counts = defaultdict(int)
    for row in records:
        candidate_counts[(row["corpus"], row["video_id"], row["segment_file"])] += 1
    collisions = [
        {"candidate": list(key), "count": count}
        for key, count in sorted(candidate_counts.items()) if count > 1
    ]

    raw_by_split = {s: {x["video_id"] for x in records if x["source_split"] == s} for s in SPLITS}
    assigned = {}
    for row in records:
        old = assigned.get(row["video_id"])
        if old is None or PRIORITY[row["source_split"]] > PRIORITY[old]:
            assigned[row["video_id"]] = row["source_split"]
    post_records = [x for x in records if assigned[x["video_id"]] == x["source_split"]]
    post_by_split = {s: {x["video_id"] for x in post_records if x["source_split"] == s} for s in SPLITS}
    speaker = speaker_source(root)
    modalities = modality_audit(post_records, root)

    mapping = OrderedDict([
        ("segment_id", {"source": "deterministic composite", "rule": "corpus + video_id + segment_file"}),
        ("video_id", {"source": "raw segment index", "columns": ["video_id"]}),
        ("speaker_id", {"source": None, "status": speaker["status"], "rule": "unresolved; never substitute video_id"}),
        ("corpus", {"source": "source file corpus directory", "rule": "depression or parkinson"}),
        ("split", {"source": "raw split plus existing priority", "rule": "test > dev > train by video_id"}),
        ("y_depression", {"source": "corpus and raw diagnosis", "depression": "raw diagnosis", "parkinson": None}),
        ("y_parkinson", {"source": "corpus and raw diagnosis", "depression": None, "parkinson": "raw diagnosis"}),
        ("observed_depression", {"source": "corpus identity", "depression": True, "parkinson": False}),
        ("observed_parkinson", {"source": "corpus identity", "depression": False, "parkinson": True}),
        ("audio_available", {"source": modalities["audio_available"]["path_rule"]}),
        ("video_available", {"source": modalities["video_available"]["path_rule"]}),
        ("text_available", {"source": modalities["text_available"]["path_rule"]}),
        ("description_available", {"source": None, "status": "unresolved", "rule": "null/unavailable"}),
    ])
    return {
        "audit_version": "stage1_manifest_source_audit_v1",
        "data_root": str(root),
        "source_files": source_files,
        "raw_columns": raw_columns,
        "corpus_split_audit": source_audit,
        "candidate_segment_identity": {
            "rule": "corpus + video_id + segment_file",
            "candidate_row_count": len(records),
            "unique_candidate_count": len(candidate_counts),
            "unique": not collisions, "collision_count": len(collisions), "collisions": collisions[:100],
        },
        "split_identity": {
            "existing_rule": "test > dev > train by video_id",
            "raw_source_video_overlap": overlaps(raw_by_split),
            "post_existing_rule_video_overlap": overlaps(post_by_split),
            "post_existing_rule_row_counts": {s: sum(x["source_split"] == s for x in post_records) for s in SPLITS},
        },
        "speaker_source": speaker,
        "modality_sources": modalities,
        "canonical_field_mapping": mapping,
        "label_contract": {
            "depression": {"observed_depression": True, "observed_parkinson": False, "y_depression": "raw diagnosis", "y_parkinson": None},
            "parkinson": {"observed_depression": False, "observed_parkinson": True, "y_depression": None, "y_parkinson": "raw diagnosis"},
            "unknown_is_null": True, "unobserved_targets_are_never_negative": True, "pseudo_labels_created": False,
        },
        "test_usage": {
            "structural_metadata_inspected": True, "model_predictions_inspected": False,
            "performance_metrics_inspected": False, "selection_or_tuning_performed": False,
        },
        "writes": {"dataset_root_modified": False, "production_manifest_created": False, "datamodule_created": False},
        "notes": [
            "source evidence only; no production manifest emitted",
            "test metadata inspected only for schema, IDs, split identity, filenames, and file existence",
            "existing exclusions and split priority were measured, not changed",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        raise AuditError(f"data root is not a directory: {root}")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise AuditError(f"refusing to write inside dataset root: {output}")
    audit = build_audit(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"wrote read-only manifest source audit: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
