#!/usr/bin/env python3
"""Audit expected audio extraction paths against canonical DataModule views."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from audio.data.carewsm_audio_datamodule import CAREWSMAudioDataModule

CONFIGS = {"id1": "deep", "id2": "deep_phon", "id4": "deep_egemaps_phon_artic"}
ROOTS = {"CARE": Path("/media/maxim/Databases/CARE"), "WSM": Path("/media/maxim/Databases/WSM_NEW")}


def output_path(record: dict[str, Any], model: str, fold: int) -> Path:
    profile = str(record["profile_id"]).removeprefix("PROFILE_")
    video = str(record["video_id"]).removeprefix("VIDEO_")
    return (ROOTS[record["corpus"]] / "extracted_unimodal" / record["label_id"] /
            record["subgroup"] / f"PROFILE_{profile}" / f"VIDEO_{video}" /
            f"audio_{model}_fold{fold}.pickle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected: dict[Path, tuple[str, str, int, str]] = {}
    summary: Counter[str] = Counter()
    for model, prefix in CONFIGS.items():
        for fold in range(5):
            config = Path("configs/audio_feature_extraction") / f"{prefix}_fold{fold}.yaml"
            module = CAREWSMAudioDataModule(**yaml.safe_load(config.read_text())["data"]["params"])
            for loader, records in module.views.items():
                for record in records:
                    path = output_path(record, model, fold)
                    if path in expected:
                        raise RuntimeError(f"duplicate expected output: {path}")
                    expected[path] = (record["corpus"], model, fold, loader)
                    summary[f"expected/{record['corpus']}/{model}/fold{fold}"] += 1
    found = set()
    legacy = set()
    for corpus, root in ROOTS.items():
        for base, target in (("extracted_unimodal", found), ("extracted", legacy)):
            for path in (root / base).rglob("audio_id*_fold*.pickle"):
                if path.name.startswith(("audio_id1_", "audio_id2_", "audio_id4_")):
                    target.add(path)
    missing = sorted(set(expected) - found)
    extra = sorted(found - set(expected))
    rows = [("missing_expected", str(path), expected[path]) for path in missing]
    rows += [("unexpected_output", str(path), "") for path in extra]
    rows += [("legacy_output", str(path), "") for path in sorted(legacy)]
    with (args.output / "path_exceptions.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("kind", "path", "context"))
        writer.writerows(rows)
    summary.update({"found": len(found), "missing": len(missing), "extra": len(extra), "legacy": len(legacy)})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
