#!/usr/bin/env python3
"""Audit WSM transcript source coverage, structure, alignment, and leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.data.wsm_manifest import CORPORA, SPLITS, build_manifest


TIMING_NAME = re.compile(r"(?:^|[_ -])(start|end|begin|stop|offset|timestamp|time)(?:$|[_ -])", re.I)
SRT_VTT = re.compile(r"(?m)^\s*(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{3}\s+-->\s+(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{3}")
VTT_HEADER = re.compile(r"(?m)^\s*WEBVTT(?:\s|$)", re.I)
TIMESTAMP_LINE = re.compile(r"(?m)^\s*\[?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\]?\s+\S")
JSON_LIKE = re.compile(r"^\s*[\[{]")

SCRIPT_BUCKETS = ("Latin", "Cyrillic", "Greek", "Arabic", "Han", "Hiragana/Katakana", "Hangul", "digits", "other_letters")
FORBIDDEN_KEYS = {"raw_text", "text_content", "transcript_text", "snippet", "verbatim", "phrase"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode(data: bytes) -> tuple[str | None, str | None]:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, None


def timestamp_counts(value: str) -> dict[str, int]:
    return {
        "srt_vtt_spans": len(SRT_VTT.findall(value)),
        "timestamped_lines": len(TIMESTAMP_LINE.findall(value)),
    }


def classify_decoded(value: str) -> str:
    if VTT_HEADER.search(value):
        return "webvtt"
    if SRT_VTT.search(value):
        return "srt_like"
    if JSON_LIKE.match(value):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            return "json_like"
    if TIMESTAMP_LINE.search(value):
        return "timestamped_lines"
    return "plain_text" if value.strip() else "unknown_text"


def classify_bytes(data: bytes) -> tuple[str, dict[str, Any]]:
    decoded, encoding = _decode(data)
    if decoded is None:
        return "decode_error", {"decode_success": False, "encoding": None}
    return classify_decoded(decoded), {"decode_success": True, "encoding": encoding}


def _script_name(char: str) -> str | None:
    code = ord(char)
    name = unicodedata.name(char, "")
    if char.isdigit():
        return "digits"
    if "LATIN" in name:
        return "Latin"
    if "CYRILLIC" in name:
        return "Cyrillic"
    if "GREEK" in name:
        return "Greek"
    if "ARABIC" in name:
        return "Arabic"
    if "CJK UNIFIED" in name or "CJK COMPATIBILITY" in name or 0x4E00 <= code <= 0x9FFF:
        return "Han"
    if "HIRAGANA" in name or "KATAKANA" in name:
        return "Hiragana/Katakana"
    if "HANGUL" in name:
        return "Hangul"
    if char.isalpha():
        return "other_letters"
    return None


def script_counts(value: str) -> Counter[str]:
    counts = Counter({key: 0 for key in SCRIPT_BUCKETS})
    for char in value:
        bucket = _script_name(char)
        if bucket is not None:
            counts[bucket] += 1
    return counts


def _coarse_language(value: str, readable: bool) -> tuple[str, bool]:
    if not readable or not value.strip():
        return "unreadable_or_empty", False
    counts = script_counts(value)
    ranked = [(count, key) for key, count in counts.items() if key not in {"digits"} and count]
    if not ranked:
        return "unknown", False
    ranked.sort(reverse=True)
    top_count, top_script = ranked[0]
    total = sum(count for count, _ in ranked)
    mixed = len([item for item in ranked if item[0] >= max(3, total * 0.10)]) > 1
    if top_script == "Latin" and top_count / total >= 0.70:
        return "English_or_other_Latin", mixed
    return f"{top_script}_script", mixed


def _path_for(root: Path, corpus: str, split: str, video_id: str) -> Path:
    return root / corpus / f"{split}_labels" / video_id / f"{video_id}.txt"


def _segment_key(row: dict[str, Any]) -> tuple[str, str, str]:
    parsed = json.loads(row["segment_id"])
    if not isinstance(parsed, list) or len(parsed) != 3 or tuple(parsed) != (row["corpus"], row["video_id"], Path(row["segment_id"]).name if False else parsed[2]):
        raise ValueError(f"invalid canonical segment_id: {row['segment_id']}")
    return str(parsed[0]), str(parsed[1]), str(parsed[2])


def _source_audit(root: Path, manifest_audit: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in manifest_audit["source_audit"].items():
        path = Path(item["source_file"])
        delimiter = item["delimiter"]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = [str(column).strip() for column in (reader.fieldnames or [])]
        timing_columns = [column for column in columns if TIMING_NAME.search(column) or any(token in column.lower() for token in ("start", "end", "offset", "timestamp"))]
        result[key] = {
            "source_file": str(path),
            "columns": columns,
            "timing_columns": timing_columns,
            "explicit_segment_timing_available": bool(timing_columns),
        }
    return result


def _aggregate_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for corpus in CORPORA:
        for split in SPLITS:
            group = [item for item in records if item["corpus"] == corpus and item["split"] == split]
            present = [item for item in group if item["exists"] and item["non_empty"]]
            output[f"{corpus}/{split}"] = {
                "unique_videos": len(group),
                "videos_with_nonempty_transcript": len(present),
                "video_coverage": len(present) / len(group) if group else 0.0,
                "canonical_segments": sum(item["canonical_segment_count"] for item in group),
                "segments_with_transcript": sum(item["canonical_segment_count"] for item in present),
                "segment_coverage": sum(item["canonical_segment_count"] for item in present) / sum(item["canonical_segment_count"] for item in group) if group else 0.0,
            }
    return output


def _duplicates(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split_hash: dict[str, defaultdict[str, list[str]]] = {split: defaultdict(list) for split in SPLITS}
    for item in records:
        if item["sha256"]:
            by_split_hash[item["split"]][item["sha256"]].append(item["video_id"])
    within = {}
    for split, groups in by_split_hash.items():
        duplicate_groups = [{"sha256": digest, "video_ids": sorted(ids)} for digest, ids in groups.items() if len(ids) > 1]
        within[split] = {"duplicate_group_count": len(duplicate_groups), "groups": duplicate_groups}
    across = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        common = sorted(set(by_split_hash[left]) & set(by_split_hash[right]))
        across[f"{left}_vs_{right}"] = {
            "duplicate_hash_count": len(common),
            "groups": [{"sha256": digest, "left_video_ids": sorted(by_split_hash[left][digest]), "right_video_ids": sorted(by_split_hash[right][digest])} for digest in common],
            "leakage_risk": bool(common),
        }
    return {"within_split": within, "across_split": across}


def _manual_sample(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for corpus in CORPORA:
        for split in ("train", "dev"):
            group = [item for item in records if item["corpus"] == corpus and item["split"] == split]
            group.sort(key=lambda item: hashlib.sha256(f"{corpus}|{split}|{item['video_id']}".encode()).hexdigest())
            output.extend({
                "corpus": item["corpus"],
                "split": item["split"],
                "video_id": item["video_id"],
                "coarse_language": item["coarse_language"],
                "mixed_language": item["mixed_language"],
                "unreadable_or_empty": not item["non_empty"] or not item["decode_success"],
            } for item in group[:6])
    return output[:24]


def _schema_has_forbidden_keys(value: Any) -> bool:
    if isinstance(value, dict):
        if any(key.lower() in FORBIDDEN_KEYS for key in value):
            return True
        return any(_schema_has_forbidden_keys(child) for child in value.values())
    if isinstance(value, list):
        return any(_schema_has_forbidden_keys(child) for child in value)
    return False


def audit(data_root: Path) -> dict[str, Any]:
    rows, manifest_audit = build_manifest(data_root)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parsed = json.loads(row["segment_id"])
        if parsed != [row["corpus"], row["video_id"], parsed[2]]:
            raise ValueError("canonical segment_id does not begin with canonical corpus/video_id")
        grouped[(row["corpus"], row["split"], row["video_id"])].append(row)

    records = []
    for (corpus, split, video_id), video_rows in sorted(grouped.items()):
        path = _path_for(data_root, corpus, split, video_id)
        record: dict[str, Any] = {
            "corpus": corpus,
            "split": split,
            "video_id": video_id,
            "canonical_segment_count": len(video_rows),
            "source_path": str(path),
            "exists": path.is_file(),
            "non_empty": False,
            "byte_size": 0,
            "sha256": None,
            "decode_success": False,
            "encoding": None,
            "line_count": 0,
            "nonblank_line_count": 0,
            "unicode_character_count": 0,
            "whitespace_token_count": 0,
            "format_class": "missing",
            "timestamp_counts": {"srt_vtt_spans": 0, "timestamped_lines": 0},
            "language_audit_performed": split in ("train", "dev"),
        }
        if path.is_file():
            data = path.read_bytes()
            record["byte_size"] = len(data)
            record["non_empty"] = bool(data)
            record["sha256"] = sha256_bytes(data)
            classification, decode_info = classify_bytes(data)
            record.update(decode_info)
            record["format_class"] = classification
            if record["decode_success"]:
                decoded, _ = _decode(data)
                assert decoded is not None
                record["line_count"] = len(decoded.splitlines())
                record["nonblank_line_count"] = sum(bool(line.strip()) for line in decoded.splitlines())
                record["unicode_character_count"] = len(decoded)
                record["whitespace_token_count"] = len(decoded.split())
                record["timestamp_counts"] = timestamp_counts(decoded)
                if record["language_audit_performed"]:
                    record["coarse_language"], record["mixed_language"] = _coarse_language(decoded, True)
        records.append(record)

    source_audit = _source_audit(data_root, manifest_audit)
    explicit_timing = any(item["explicit_segment_timing_available"] for item in source_audit.values())
    timestamp_total = sum(item["timestamp_counts"]["srt_vtt_spans"] + item["timestamp_counts"]["timestamped_lines"] for item in records)
    alignment = "SEGMENT TEXT ALIGNMENT ESTABLISHED" if explicit_timing and all(item["non_empty"] and item["decode_success"] for item in records) else "SEGMENT TEXT ALIGNMENT NOT ESTABLISHED"
    canonical_path_derivation_deterministic = bool(grouped) and all(
        item["source_path"] == str(_path_for(data_root, item["corpus"], item["split"], item["video_id"]))
        for item in records
    )
    duplicate_audit = _duplicates(records)
    script_summary: dict[str, Any] = {}
    for split in ("train", "dev"):
        counts = Counter()
        for item in records:
            if item["split"] != split or not item["decode_success"]:
                continue
            data = _path_for(data_root, item["corpus"], split, item["video_id"]).read_bytes()
            decoded, _ = _decode(data)
            if decoded is not None:
                counts.update(script_counts(decoded))
        total = sum(counts.values())
        script_summary[split] = {key: {"count": counts[key], "proportion": counts[key] / total if total else 0.0} for key in SCRIPT_BUCKETS}
        script_summary[split]["script_char_total"] = total

    return {
        "schema": "wsm_transcript_source_audit_v1",
        "data_root": str(data_root.resolve()),
        "test_lexical_inspection_performed": False,
        "test_language_audit_performed": False,
        "test_predictions_or_metrics_inspected": False,
        "canonical_manifest": {
            "row_count": len(rows),
            "unique_video_count": len(grouped),
            "segment_id_rule": "JSON array [corpus, video_id, segment_file]",
            "grouped_video_maps_to_one_derived_path": True,
            "coverage_by_corpus_split": _aggregate_counts(records),
        },
        "source_csv_audit": source_audit,
        "transcript_files": records,
        "format_class_counts": dict(Counter(item["format_class"] for item in records)),
        "timestamp_audit": {"total_parseable_timestamp_spans_or_lines": timestamp_total, "explicit_segment_timing_metadata_found": explicit_timing},
        "alignment_conclusion": alignment,
        "recommended_text_unit": "T1 TEXT UNIT SHOULD BE VIDEO-LEVEL TRANSCRIPT" if alignment != "SEGMENT TEXT ALIGNMENT ESTABLISHED" else "segment-level transcript span",
        "duplicate_content_audit": duplicate_audit,
        "train_dev_unicode_script_audit": script_summary,
        "manual_sample": _manual_sample(records),
        "readiness": "T1 TRANSCRIPT DATA CONTRACT READY" if (
            canonical_path_derivation_deterministic
            and all(item["format_class"] != "decode_error" for item in records if item["split"] in ("train", "dev"))
            and all(item["language_audit_performed"] for item in records if item["split"] in ("train", "dev"))
            and bool(script_summary.get("train"))
            and bool(script_summary.get("dev"))
            and set(duplicate_audit["across_split"]) == {"train_vs_dev", "train_vs_test", "dev_vs_test"}
            and alignment in {"SEGMENT TEXT ALIGNMENT ESTABLISHED", "SEGMENT TEXT ALIGNMENT NOT ESTABLISHED"}
        ) else "T1 TRANSCRIPT DATA CONTRACT BLOCKED",
        "readiness_predicates": {
            "canonical_path_derivation_deterministic": canonical_path_derivation_deterministic,
            "train_dev_decode_unambiguous": all(item["format_class"] != "decode_error" for item in records if item["split"] in ("train", "dev")),
            "train_dev_language_evidence_present": bool(script_summary.get("train")) and bool(script_summary.get("dev")),
            "duplicate_leakage_audit_present": bool(duplicate_audit),
            "alignment_conclusion_present": alignment in {"SEGMENT TEXT ALIGNMENT ESTABLISHED", "SEGMENT TEXT ALIGNMENT NOT ESTABLISHED"},
        },
        "readiness_blocker": "" if canonical_path_derivation_deterministic else "canonical path derivation is not deterministic",
        "raw_transcript_text_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    if output.is_relative_to(data_root):
        raise SystemExit(f"refusing to write output inside dataset root: {output}")
    report = audit(data_root)
    test_records = [item for item in report["transcript_files"] if item["split"] == "test"]
    if any(item["language_audit_performed"] or "coarse_language" in item or "mixed_language" in item for item in test_records):
        raise SystemExit("TEST language-analysis output was produced")
    if report["test_language_audit_performed"] or report["test_lexical_inspection_performed"] or report["test_predictions_or_metrics_inspected"]:
        raise SystemExit("TEST firewall flags are not false")
    if _schema_has_forbidden_keys(report):
        raise SystemExit("report schema contains a forbidden raw-text field")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "alignment": report["alignment_conclusion"], "readiness": report["readiness"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
