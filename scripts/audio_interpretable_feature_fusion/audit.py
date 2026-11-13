#!/usr/bin/env python3
"""Audit the complete-case cohort and write privacy-safe provenance artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

import chimera_plugin  # noqa: F401
from audio.data.carewsm_audio_dataset import feature_column_names, load_feature_array
from audio.data.carewsm_audio_datamodule import CAREWSMAudioDataModule

MANIFEST = Path("reports/feature_audit/sample_feature_availability.csv")
OUT = Path("reports/audio_interpretable_feature_fusion")
REGISTRY = Path("/media/maxim/Databases/CARE/splits/features.json")
CONFIG = Path(
    "configs/audio_interpretable_feature_fusion/interp_BEPA_mean_gated64_equal_fold0.yaml"
)
FAMILIES = (
    "emotionmsp_embeds",
    "emotionmsp_trajectories",
    "trillsson4_1",
    "wav2vec_xlsr300m",
    "egemaps_func",
    "disvoice_phonation_static",
    "disvoice_articulation_static",
)


def digest(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def observed_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n": len(records),
        "depression_observed": sum(
            bool(record["depression_observed"]) for record in records
        ),
        "parkinson_observed": sum(
            bool(record["parkinson_observed"]) for record in records
        ),
        "depression_positive": sum(
            bool(record["depression_observed"] and record["depression"] == 1.0)
            for record in records
        ),
        "parkinson_positive": sum(
            bool(record["parkinson_observed"] and record["parkinson"] == 1.0)
            for record in records
        ),
        "sample_fingerprint": digest([record["sample_id"] for record in records]),
        "profile_fingerprint": digest([record["profile_id"] for record in records]),
    }


def build_feature_dictionary(
    data_module: CAREWSMAudioDataModule,
    registry: dict[str, dict[str, Any]],
    model_params: dict[str, Any],
) -> dict[str, Any]:
    record = data_module.views["train"][0]
    families = []
    offset = 0
    for family, expected_dimension in zip(
        FAMILIES, model_params["family_dims"], strict=True
    ):
        specification = registry[family]
        path = Path(record["feature_paths"][family])
        values = load_feature_array(path, specification, family)
        columns = feature_column_names(path, specification, family, values.shape[1])
        if values.shape[1] != expected_dimension or len(columns) != expected_dimension:
            raise ValueError(
                f"actual dimension mismatch for {family}: "
                f"{values.shape[1]} vs {expected_dimension}"
            )
        groups = {
            column: group
            for group, names in specification.get("families", {}).items()
            for column in names
        }
        families.append(
            {
                "name": family,
                "input_index_start": offset,
                "input_index_end": offset + expected_dimension,
                "dimension": expected_dimension,
                "level": specification.get("level", "unknown"),
                "format": specification.get("format", "unknown"),
                "source_file": specification.get("file", "unknown"),
                "columns": [
                    {
                        "name": column,
                        "meaning": groups.get(column, "unknown"),
                        "units": "unknown",
                    }
                    for column in columns
                ],
            }
        )
        offset += expected_dimension
    if offset != model_params["input_dim"]:
        raise ValueError("feature dictionary does not cover model input")
    indices = [
        index
        for family in families
        for index in range(family["input_index_start"], family["input_index_end"])
    ]
    if len(indices) != len(set(indices)) or set(indices) != set(range(offset)):
        raise ValueError(
            "feature dictionary input indices are not unique and contiguous"
        )
    return {
        "families": families,
        "cohort_policy": "complete-case across all seven families",
        "aggregation": "independent per-family arithmetic mean over native turns",
        "family_order": list(FAMILIES),
        "input_dim": offset,
    }


def build_fold_audit() -> dict[str, Any]:
    fold_audit = {}
    for fold in range(5):
        data_module = CAREWSMAudioDataModule(
            care_root="/media/maxim/Databases/CARE",
            wsm_root="/media/maxim/Databases/WSM_NEW",
            manifest=str(MANIFEST),
            registry=str(REGISTRY),
            feature_family=list(FAMILIES),
            cohort_feature_family=list(FAMILIES),
            independent_family_mean=True,
            pooling="mean",
            fold=fold,
            batch_size=64,
            num_workers=0,
        )
        populations = {
            "train": data_module.views["train"],
            "val_care": data_module.views["val_care"],
            "val_wsm": data_module.views["val_wsm"],
            "test_care": data_module.views["test_care"],
            "test_wsm": data_module.views["test_wsm"],
        }
        fold_audit[str(fold)] = {
            name: observed_counts(records) for name, records in populations.items()
        }
    return fold_audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    complete = [
        row
        for row in rows
        if all(row.get(f"{family}_status") == "VALID" for family in FAMILIES)
    ]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["sources"]
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data_module = CAREWSMAudioDataModule(**config["data"]["params"])
    dictionary = build_feature_dictionary(
        data_module, registry, config["model"]["params"]
    )
    (OUT / "feature_dictionary.json").write_text(
        json.dumps(dictionary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "complete_case_sample_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "corpus",
            "split",
            "fold",
            "sample_hash",
            "profile_hash",
            "feature_families",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in complete:
            writer.writerow(
                {
                    "corpus": row.get("corpus", ""),
                    "split": row.get("split", ""),
                    "fold": row.get("cv_fold", ""),
                    "sample_hash": hashlib.sha256(
                        row.get("sample_id", "").encode()
                    ).hexdigest()[:20],
                    "profile_hash": hashlib.sha256(
                        row.get("profile_id", "").encode()
                    ).hexdigest()[:20],
                    "feature_families": "+".join(FAMILIES),
                }
            )
    result = {
        "manifest": str(MANIFEST),
        "families": list(FAMILIES),
        "manifest_rows": len(rows),
        "complete_case_rows": len(complete),
        "folds": build_fold_audit(),
        "status": "PASS",
    }
    (OUT / "cohort_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
