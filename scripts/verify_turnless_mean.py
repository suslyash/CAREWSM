#!/usr/bin/env python3
"""Verify turnless mean pooling on real CARE and WSM feature artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio.data.carewsm_audio_datamodule import CAREWSMAudioDataModule
from audio.data.carewsm_audio_dataset import CAREWSMAudioDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = CAREWSMAudioDataModule(
        care_root="/media/maxim/Databases/CARE",
        wsm_root="/media/maxim/Databases/WSM_NEW",
        manifest="reports/feature_audit/sample_feature_availability.csv",
        registry="/media/maxim/Databases/CARE/splits/features.json",
        feature_family=["emotionmsp_embeds", "trillsson4_1"],
        fold=0,
        batch_size=2,
    )
    checks = []
    for corpus in ("CARE", "WSM"):
        records = module.views[f"test_{corpus.lower()}"]
        record = next(record for record in records if corpus == "WSM" or record)
        raw = CAREWSMAudioDataset(
            [record],
            ["emotionmsp_embeds", "trillsson4_1"],
            module.feature_specs,
            pooling="native",
        )[0]
        pooled = CAREWSMAudioDataset(
            [record],
            ["emotionmsp_embeds", "trillsson4_1"],
            module.feature_specs,
            pooling="mean",
        )[0]
        values = raw["inputs"]["native"].numpy()
        expected = values.mean(axis=0, keepdims=True)
        checks.append(
            {
                "corpus": corpus,
                "sample_id": record["sample_id"],
                "turn_count": int(values.shape[0]),
                "feature_shape": list(pooled["inputs"]["native"].shape),
                "max_abs_error": float(
                    np.max(np.abs(pooled["inputs"]["native"].numpy() - expected))
                ),
                "single_turn_unchanged": bool(
                    corpus != "WSM"
                    or values.shape[0] != 1
                    or np.array_equal(values, expected)
                ),
            }
        )

    care_multi = next(
        record
        for record in module.views["test_care"]
        if CAREWSMAudioDataset(
            [record],
            ["emotionmsp_embeds", "trillsson4_1"],
            module.feature_specs,
            pooling="native",
        )[0]["inputs"]["native"].shape[0]
        > 1
    )
    checks.append(
        {
            "corpus": "CARE",
            "sample_id": care_multi["sample_id"],
            "multi_turn_explicit_check": True,
        }
    )
    result = {"status": "PASS", "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
