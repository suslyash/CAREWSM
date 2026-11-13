#!/usr/bin/env python3
"""Validate static mean and population-std pooling on live feature artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audio.data.carewsm_audio_datamodule import CAREWSMAudioDataModule

FAMILIES = [
    "emotionmsp_embeds",
    "emotionmsp_trajectories",
    "trillsson4_1",
    "wav2vec_xlsr300m",
]


def main() -> None:
    module = CAREWSMAudioDataModule(
        care_root="/media/maxim/Databases/CARE",
        wsm_root="/media/maxim/Databases/WSM_NEW",
        manifest="reports/feature_audit/sample_feature_availability.csv",
        registry="/media/maxim/Databases/CARE/splits/features.json",
        feature_family=FAMILIES,
        fold=0,
        pooling="mean_std",
    )
    sample = module.views["train"][0]
    dataset = module.train_dataset
    item = dataset[0]
    values = []
    for family in FAMILIES:
        path = Path(sample["feature_paths"][family])
        specification = module.feature_specs[family]
        from audio.data.carewsm_audio_dataset import load_feature_component

        array, turn_ids = load_feature_component(path, specification, family)
        values.append((family, array, turn_ids))
    if any(not np.isfinite(array).all() for _, array, _ in values):
        raise ValueError("non-finite live feature values")
    if any(ids != values[0][2] for _, _, ids in values[1:]):
        raise ValueError("turn IDs are not aligned")
    expected = np.concatenate(
        [np.concatenate((array.mean(0), array.std(0))) for _, array, _ in values]
    )
    actual = item["inputs"]["native"][0].numpy()
    if actual.shape != (6150,) or not np.array_equal(
        actual, expected.astype(np.float32)
    ):
        raise ValueError("mean+std pooling arithmetic or dimension mismatch")
    singleton = next((array for _, array, _ in values if array.shape[0] == 1), None)
    if singleton is None or not np.array_equal(
        singleton.std(0), np.zeros(singleton.shape[1])
    ):
        raise ValueError("singleton population standard deviation is not zero")
    result = {
        "status": "PASS",
        "feature_families": FAMILIES,
        "mean_dimensions": [1024, 3, 1024, 1024],
        "mean_std_dimension": 6150,
        "turn_count": len(values[0][2]),
        "train_profile_count": len(
            {row["profile_id"] for row in module.views["train"]}
        ),
        "val_profile_count": len(
            {
                row["profile_id"]
                for row in module.views["val_care"] + module.views["val_wsm"]
            }
        ),
    }
    output = Path("logs/audio_turns_combined/launcher/pooling_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
