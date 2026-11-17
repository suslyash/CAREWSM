#!/usr/bin/env python3
"""Audit missing audio extraction outputs against the audited input manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml


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
MODEL_CONFIG_PREFIX = {
    "id1": "deep",
    "id2": "deep_phon",
    "id4": "deep_egemaps_phon_artic",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def output_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    if path.stat().st_size == 0:
        return "INVALID_ZERO_BYTE"
    return "PRESENT"


def load_cohorts(config_root: Path) -> dict[str, set[str]]:
    cohorts = {}
    for model, prefix in MODEL_CONFIG_PREFIX.items():
        for fold in range(5):
            path = config_root / f"{prefix}_fold{fold}.yaml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            cohorts[f"audio_{model}_fold{fold}"] = set(
                config["data"]["params"]["cohort_feature_family"]
            )
    return cohorts


def complete_case_reason(
    source_exists: bool,
    split_exists: bool,
    manifest_has_statuses: bool,
    output_states: dict[str, str],
    statuses: dict[str, str],
    cohorts: dict[str, set[str]],
) -> tuple[str, set[str]]:
    if not source_exists:
        return "не установлена: исходная VIDEO-папка отсутствует", set()
    if not split_exists:
        return "не установлена: запись отсутствует в split metadata", set()
    if not manifest_has_statuses:
        return "не установлена: статусы входных семейств отсутствуют в manifest", set()

    missing_outputs = [name for name, state in output_states.items() if state == "MISSING"]
    cause_families = {
        family
        for output in missing_outputs
        for family in cohorts[output]
        if statuses[family] != "VALID"
    }
    if not cause_families:
        return "не установлена", set()
    details = ", ".join(
        f"{family}={statuses[family]}" for family in sorted(cause_families)
    )
    return f"complete-case exclusion: {details}", cause_families


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--wsm-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/audio_feature_extraction"),
    )
    args = parser.parse_args()

    roots = {"CARE": args.care_root, "WSM": args.wsm_root}
    cohorts = load_cohorts(args.config_root)
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
        manifest_has_statuses = all(f"{family}_status" in manifest for family in FAMILIES)
        statuses = {
            family: manifest.get(f"{family}_status", "UNKNOWN") for family in FAMILIES
        }
        problem_families = [
            (family, statuses[family])
            for family in FAMILIES
            if statuses[family] != "VALID"
        ]
        if not problem_families:
            problem_families = [("NONE", "VALID")]
        reason, cause_families = complete_case_reason(
            source_exists,
            split is not None,
            manifest_has_statuses,
            output_states,
            statuses,
            cohorts,
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
                "input_scope": (
                    "cohort_feature_family"
                    if family in cause_families
                    else "additional_information"
                ),
                "complete_case_reason": reason,
            })
        summary[f"output_files_missing/{corpus}"] += len(missing_outputs)
        summary[f"output_files_invalid/{corpus}"] += len(invalid_outputs)
        for family, status in problem_families:
            if family != "NONE":
                summary[f"input_problem/{corpus}/{family}/{status}"] += 1
        if reason == "не установлена":
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
