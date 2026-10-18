#!/usr/bin/env python3
"""Read-only CARE parity validator for the eleven audio families.

TASK-003A validates the seven descriptor CSVs. The four TASK-003B families are
recognized from the CARE registry but are not produced by this task.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

SEVEN = (
    "egemaps_func.csv", "egemaps_lld.csv", "disvoice_prosody_static.csv",
    "disvoice_prosody_dynamic.csv", "disvoice_phonation_static.csv",
    "disvoice_phonation_dynamic.csv", "disvoice_articulation_static.csv",
)
DEEP = ("wav2vec_xlsr300m.npz", "trillsson4_1.npz", "emotionmsp_trajectories.csv", "emotionmsp_embeds.npz")
META = {
    "egemaps_func.csv": ("turn_id", "turn_start_sec", "turn_end_sec"),
    "egemaps_lld.csv": ("turn_id", "turn_start_sec", "turn_end_sec", "frame_start_sec", "frame_end_sec"),
    "disvoice_prosody_static.csv": ("turn_id", "turn_start_sec", "turn_end_sec"),
    "disvoice_prosody_dynamic.csv": ("turn_id", "turn_start_sec", "turn_end_sec", "frame_start_sec", "frame_end_sec"),
    "disvoice_phonation_static.csv": ("turn_id", "turn_start_sec", "turn_end_sec"),
    "disvoice_phonation_dynamic.csv": ("turn_id", "turn_start_sec", "turn_end_sec", "frame_start_sec", "frame_end_sec"),
    "disvoice_articulation_static.csv": ("turn_id", "turn_start_sec", "turn_end_sec"),
}


def reference_headers(care_root: Path, filename: str) -> list[str]:
    matches = sorted(care_root.glob(f"extracted/**/{filename}"))
    if not matches:
        raise FileNotFoundError(f"no CARE reference file for {filename}")
    with matches[0].open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def validate_csv(path: Path, headers: list[str], max_rows: int = 100) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        actual = next(csv.reader(handle))
    if actual != headers:
        return {"status": "fail", "reason": "header_mismatch", "actual_columns": len(actual), "expected_columns": len(headers)}
    numeric = [column for column in headers if column not in META.get(path.name, ())]
    bad = []
    rows_checked = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_checked += 1
            for column in numeric:
                try:
                    float(row[column])
                except (TypeError, ValueError):
                    bad.append(column)
            if rows_checked >= max_rows:
                break
    if bad:
        return {"status": "fail", "reason": "non_numeric_feature_columns", "columns": sorted(set(bad))}
    return {"status": "ok", "rows_checked": rows_checked, "columns": len(headers), "numeric_columns": len(numeric)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-root", required=True, type=Path)
    parser.add_argument("--wsm-root", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "dev", "test"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--all-audio", action="store_true", help="also report the four TASK-003B families without requiring them")
    args = parser.parse_args()
    with (args.wsm_root / "splits" / f"{args.split}.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("split") == args.split and row.get("storage_type") == "files"]
    rows.sort(key=lambda row: (row["sample_id"], row["video_path"]))
    selected = []
    used = set()
    for stratum in ("DEPRESSION", "CONTROL", "PARKINSON"):
        row = next((item for item in rows if item.get("stratum") == stratum), None)
        if row and row["sample_id"] not in used:
            selected.append(row)
            used.add(row["sample_id"])
    rows = selected[:args.limit]
    result = {"task": "TASK-003A", "mode": "read_only", "split": args.split, "families": list(SEVEN), "samples": []}
    if args.all_audio:
        result["families"] += list(DEEP)
    for row in rows:
        destination = (args.wsm_root / row["feature_dir"]).resolve()
        sample = {"sample_id": row["sample_id"], "feature_dir": row["feature_dir"], "files": {}}
        for filename in SEVEN:
            path = destination / filename
            headers = reference_headers(args.care_root, filename)
            sample["files"][filename] = {"path": str(path), "exists": path.is_file()}
            if path.is_file():
                sample["files"][filename]["validation"] = validate_csv(path, headers)
        result["samples"].append(sample)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
