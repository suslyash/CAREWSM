#!/usr/bin/env python3
"""Read-only CARE parity validator for the eleven audio families.

TASK-003A validates seven descriptor CSVs; --all-audio validates all eleven
owned audio families, including the four TASK-003B NPZ/CSV contracts.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

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


def validate_csv(path: Path, headers: list[str]) -> dict:
    errors = []
    blank_values = 0
    nonfinite_values = 0
    row_count = 0
    metadata = set(META.get(path.name, ()))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != headers:
            return {"status": "fail", "reason": "header_mismatch", "actual_columns": len(reader.fieldnames or []), "expected_columns": len(headers)}
        for row_number, row in enumerate(reader, 2):
            row_count += 1
            if None in row:
                errors.append(f"row {row_number}: extra columns")
            for column in headers:
                value = row.get(column)
                if value is None:
                    errors.append(f"row {row_number}: missing {column}")
                    continue
                if value.strip() == "":
                    if column in metadata:
                        errors.append(f"row {row_number}: blank metadata {column}")
                    else:
                        blank_values += 1
                    continue
                try:
                    number = float(value)
                except ValueError:
                    errors.append(f"row {row_number}: nonnumeric {column}")
                    continue
                if not math.isfinite(number):
                    nonfinite_values += 1
                    errors.append(f"row {row_number}: nonfinite {column}")
            if "turn_id" in metadata:
                value = row.get("turn_id", "")
                try:
                    if value.strip() and float(value) != int(float(value)):
                        errors.append(f"row {row_number}: noninteger turn_id")
                except ValueError:
                    pass
    if row_count == 0:
        errors.append("empty data rows")
    return {"status": "ok" if not errors else "fail", "rows": row_count, "columns": len(headers), "numeric_columns": len(headers) - len(metadata), "blank_feature_values": blank_values, "nonfinite_values": nonfinite_values, **({"errors": errors[:20]} if errors else {})}


def validate_npz(path: Path, filename: str) -> dict:
    expected = {
        "wav2vec_xlsr300m.npz": ("mean", "std", "turn_ids"),
        "trillsson4_1.npz": ("mean", "std", "turn_ids"),
        "emotionmsp_embeds.npz": ("embed", "turn_ids"),
    }[filename]
    with np.load(path, allow_pickle=False) as data:
        if tuple(data.files) != expected:
            return {"status": "fail", "reason": "keys", "actual_keys": data.files, "expected_keys": expected}
        turn_ids = data["turn_ids"]
        if turn_ids.dtype != np.dtype("int64") or turn_ids.ndim != 1 or len(turn_ids) == 0 or not np.array_equal(turn_ids, np.arange(1, len(turn_ids) + 1, dtype=np.int64)):
            return {"status": "fail", "reason": "turn_ids_dtype_shape_or_order"}
        errors = []
        shapes = {}; dtypes = {}
        for key in expected:
            value = data[key]; shapes[key] = list(value.shape); dtypes[key] = str(value.dtype)
            if key != "turn_ids" and (value.dtype != np.dtype("float32") or value.shape != (len(turn_ids), 1024)):
                errors.append(f"{key}: expected float32 ({len(turn_ids)},1024), got {value.dtype} {value.shape}")
            if not np.isfinite(value).all():
                errors.append(f"{key}: NaN/Inf")
        return {"status": "ok" if not errors else "fail", "rows": len(turn_ids), "keys": list(data.files), "shapes": shapes, "dtypes": dtypes, **({"errors": errors} if errors else {})}


def validate_deep_csv(path: Path, headers: list[str]) -> dict:
    return validate_csv(path, headers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-root", required=True, type=Path)
    parser.add_argument("--wsm-root", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "dev", "test"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--all-audio", action="store_true", help="also require and validate the four TASK-003B families")
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
    families = list(SEVEN) + (list(DEEP) if args.all_audio else [])
    result = {"task": "TASK-003B" if args.all_audio else "TASK-003A", "mode": "read_only", "split": args.split, "families": families, "samples": [], "status": "ok", "errors": []}
    for row in rows:
        for field in ("label_id", "subgroup", "profile_id", "video_id"):
            if not row.get(field):
                raise RuntimeError(f"manifest row missing output field: {field}")
        destination = (args.wsm_root / "extracted" / row["label_id"] / row["subgroup"] / f"PROFILE_{row['profile_id']}" / f"VIDEO_{row['video_id']}").resolve()
        sample = {"sample_id": row["sample_id"], "source_feature_dir_ignored": row["feature_dir"], "destination": str(destination), "files": {}}
        root = args.wsm_root.resolve()
        if root not in destination.parents:
            result["errors"].append(f"destination escapes WSM root: {destination}")
        for filename in families:
            path = destination / filename
            file_result = {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
            if path.is_file():
                if filename.endswith(".npz"):
                    file_result["validation"] = validate_npz(path, filename)
                else:
                    headers = reference_headers(args.care_root, filename)
                    file_result["validation"] = validate_deep_csv(path, headers) if filename == "emotionmsp_trajectories.csv" else validate_csv(path, headers)
                if file_result["validation"]["status"] != "ok":
                    result["errors"].append(f"{row['sample_id']}:{filename}")
            else:
                result["errors"].append(f"missing {row['sample_id']}:{filename}")
            sample["files"][filename] = file_result
        result["samples"].append(sample)
    result["status"] = "ok" if not result["errors"] else "fail"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
