#!/usr/bin/env python3
"""Audit missing audio extraction outputs against the audited input manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


FAMILIES = (
    "egemaps_func",
    "egemaps_lld",
    "disvoice_prosody_static",
    "disvoice_prosody_dynamic",
    "disvoice_phonation_static",
    "disvoice_phonation_dynamic",
    "disvoice_articulation_static",
    "wav2vec_xlsr300m",
    "trillsson4_1",
    "emotionmsp_trajectories",
    "emotionmsp_embeds",
)
MODELS = ("id1", "id2", "id4")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def output_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    if path.stat().st_size == 0:
        return "INVALID_ZERO_BYTE"
    return "PRESENT"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--wsm-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = {"CARE": args.care_root, "WSM": args.wsm_root}
    manifest_rows = read_rows(args.manifest)
    split_rows: dict[tuple[str, str], dict[str, str]] = {}
    for corpus, root in roots.items():
        for split_name in ("train", "dev", "test"):
            split_path = root / "splits" / f"{split_name}.csv"
            for row in read_rows(split_path):
                split_rows[(corpus, row["sample_id"])] = row

    detailed_rows: list[dict[str, str]] = []
    seen_video_keys: set[tuple[str, str]] = set()
    summary = Counter()
    for manifest in manifest_rows:
        corpus = manifest["corpus"]
        root = roots[corpus]
        sample_id = manifest["sample_id"]
        split = split_rows.get((corpus, sample_id))
        label = split["label_id"] if split else manifest.get("label_id", "")
        subgroup = split["subgroup"] if split else manifest.get("subgroup", "")
        profile = split["profile_id"] if split else manifest["profile_id"]
        video = split["video_id"] if split else manifest["video_id"]
        relative_path = "/".join((label, subgroup, f"PROFILE_{profile.removeprefix('PROFILE_')}",
                                   f"VIDEO_{video.removeprefix('VIDEO_')}"))
        source_dir = root / "extracted" / relative_path
        output_dir = root / "extracted_unimodal" / relative_path
        source_exists = source_dir.is_dir()
        output_states = {
            f"audio_{model}_fold{fold}": output_status(
                output_dir / f"audio_{model}_fold{fold}.pickle"
            )
            for model in MODELS
            for fold in range(5)
        }
        incomplete_outputs = [name for name, state in output_states.items() if state != "PRESENT"]
        if not incomplete_outputs:
            continue

        key = (corpus, sample_id)
        if source_exists:
            category = "VIDEO_DIR_WITH_INCOMPLETE_OUTPUT"
            seen_video_keys.add(key)
            summary[f"affected_videos/{corpus}"] += 1
        else:
            category = "MANIFEST_WITHOUT_SOURCE_VIDEO_DIR"
            summary[f"missing_source/{corpus}"] += 1
        problem_families = [
            (family, manifest[f"{family}_status"])
            for family in FAMILIES
            if manifest[f"{family}_status"] != "VALID"
        ]
        if not problem_families:
            problem_families = [("NONE", "VALID")]
        has_complete_case_cause = any(status != "VALID" for _, status in problem_families)
        reason = (
            "complete-case exclusion supported by non-VALID input status"
            if has_complete_case_cause
            else "не установлена"
        )
        missing_outputs = [name for name, state in output_states.items() if state == "MISSING"]
        invalid_outputs = [f"{name}={state}" for name, state in output_states.items()
                           if state not in ("PRESENT", "MISSING")]
        for family, status in problem_families:
            detailed_rows.append({
                "category": category,
                "corpus": corpus,
                "relative_video_path": relative_path,
                "sample_id": sample_id,
                "in_split_metadata": "YES" if split else "NO",
                "split": split["split"] if split else manifest.get("split", ""),
                "split_cv_fold": split["cv_fold"] if split else manifest.get("cv_fold", ""),
                "missing_model_fold_files": ";".join(missing_outputs),
                "invalid_model_fold_files": ";".join(invalid_outputs),
                "input_family": family,
                "input_status": status,
                "complete_case_reason": reason,
            })
        summary[f"output_files_missing/{corpus}"] += len(missing_outputs)
        summary[f"output_files_invalid/{corpus}"] += len(invalid_outputs)
        for family, status in problem_families:
            if family != "NONE":
                summary[f"input_problem/{corpus}/{family}/{status}"] += 1
        if not has_complete_case_cause:
            summary[f"unexplained/{corpus}"] += 1

    for corpus in roots:
        summary.setdefault(f"affected_videos/{corpus}", 0)
        summary.setdefault(f"missing_source/{corpus}", 0)
        summary.setdefault(f"unexplained/{corpus}", 0)

    args.output.mkdir(parents=True, exist_ok=True)
    detailed_path = args.output / "missing_videos.tsv"
    fieldnames = list(detailed_rows[0]) if detailed_rows else []
    with detailed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(detailed_rows, key=lambda row: (
            row["corpus"], row["category"], row["sample_id"], row["input_family"]
        )))

    summary_path = args.output / "summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "count"))
        writer.writerows(sorted(summary.items()))

    print(f"detail_rows={len(detailed_rows)}")
    print(f"affected_video_keys={len(seen_video_keys)}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
