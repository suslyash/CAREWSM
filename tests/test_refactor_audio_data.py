"""Focused synthetic coverage for the simplified CAREWSM audio data layer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import chimera_plugin
from audio.data.carewsm_audio_datamodule import CAREWSMAudioDataModule
from chimera_ml.core.config import load_yaml
from chimera_ml.core.registry import DATAMODULES
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel
from chimera_ml.training.builders import BuildContext, build_datamodule


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    specifications = {
        "static_a": {
            "modality": "audio",
            "file": "static_a.npz",
            "format": "npz",
            "embedding_keys": ["embed"],
        },
        "static_b": {
            "modality": "audio",
            "file": "static_b.npz",
            "format": "npz",
            "embedding_keys": ["embed"],
        },
        "static_c": {
            "modality": "audio",
            "file": "static_c.npz",
            "format": "npz",
            "embedding_keys": ["embed"],
        },
        "temporal": {
            "modality": "audio",
            "file": "temporal.npz",
            "format": "npz",
            "embedding_keys": ["embed"],
        },
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"sources": specifications}), encoding="utf-8")

    manifest_fields = [
        "corpus",
        "sample_id",
        "depression",
        "parkinson",
        "depression_observed",
        "parkinson_observed",
        "static_a_status",
        "static_b_status",
        "static_c_status",
        "temporal_status",
    ]
    manifest_rows: list[dict[str, str]] = []
    for corpus in ("CARE", "WSM"):
        root = tmp_path / corpus
        source_fields = [
            "sample_id",
            "profile_id",
            "video_id",
            "label_id",
            "subgroup",
            "video_dir",
        ]
        split_rows: dict[str, list[dict[str, str]]] = {}
        for split, prefix in (("train", "tr"), ("val", "va"), ("test", "te")):
            rows = []
            for number in (1, 2):
                sample_id = f"{prefix}_{corpus}_{number}"
                profile_id = f"profile_{prefix}_{corpus}_{number}"
                source = {
                    "sample_id": sample_id,
                    "profile_id": profile_id,
                    "video_id": sample_id,
                    "label_id": "DEPRESSION",
                    "subgroup": "group",
                    "video_dir": f"{corpus}/legacy/{profile_id}/VIDEO_{sample_id}",
                }
                rows.append(source)
                manifest_rows.append(
                    {
                        "corpus": corpus,
                        "sample_id": sample_id,
                        "depression": "1",
                        "parkinson": "",
                        "depression_observed": "True",
                        "parkinson_observed": "False",
                        "static_a_status": "VALID",
                        "static_b_status": "VALID",
                        "static_c_status": "VALID",
                        "temporal_status": "VALID",
                    }
                )
                if number == 1:
                    directory = (
                        root
                        / "extracted"
                        / "DEPRESSION"
                        / "group"
                        / f"PROFILE_{profile_id}"
                        / f"VIDEO_{sample_id}"
                    )
                else:
                    directory = root / "legacy" / profile_id / f"VIDEO_{sample_id}"
                directory.mkdir(parents=True, exist_ok=True)
                np.savez(
                    directory / "static_a.npz", embed=np.ones((1, 2), dtype=np.float32)
                )
                np.savez(
                    directory / "static_b.npz", embed=np.ones((1, 3), dtype=np.float32)
                )
                np.savez(
                    directory / "static_c.npz", embed=np.ones((1, 4), dtype=np.float32)
                )
                np.savez(
                    directory / "temporal.npz", embed=np.ones((5, 2), dtype=np.float32)
                )
            split_rows[split] = rows
            write_csv(root / "splits" / f"{split}.csv", source_fields, rows)
        write_csv(root / "splits" / "dev.csv", source_fields, split_rows["val"])
        for fold in range(5):
            write_csv(
                root / "splits" / "folds" / f"fold_{fold}_train.csv",
                source_fields,
                split_rows["train"],
            )
            write_csv(
                root / "splits" / "folds" / f"fold_{fold}_val.csv",
                source_fields,
                split_rows["val"],
            )
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, manifest_fields, manifest_rows)
    return tmp_path / "CARE", tmp_path / "WSM", manifest, registry


def build_dm(tmp_path: Path, **kwargs) -> CAREWSMAudioDataModule:
    care, wsm, manifest, registry = make_fixture(tmp_path)
    return CAREWSMAudioDataModule(
        str(care), str(wsm), str(manifest), str(registry), **kwargs
    )


def test_joint_five_population_data_contract(tmp_path: Path) -> None:
    dm = build_dm(tmp_path, feature_family="static_a")
    assert set(dm.val_dataset) == {"val_care", "val_wsm", "test_care", "test_wsm"}
    assert dm.test_dataset is None
    assert len(dm.train_dataset) == 4
    assert len(dm.val_dataset["val_care"]) == 2
    assert len(dm.val_dataset["val_wsm"]) == 2
    assert len(dm.val_dataset["test_care"]) == 2
    assert len(dm.val_dataset["test_wsm"]) == 2
    sample = dm.train_dataset[0]
    assert tuple(sample["inputs"]["native"].shape) == (1, 2)
    assert sample["targets"].tolist() == [1.0, 0.0]
    assert sample["meta"]["observed_targets"].tolist() == [True, False]


def test_all_folds_keep_corpus_and_profile_isolation_in_fixture(tmp_path: Path) -> None:
    care, wsm, manifest, registry = make_fixture(tmp_path)
    for fold in range(5):
        dm = CAREWSMAudioDataModule(
            str(care), str(wsm), str(manifest), str(registry),
            fold=fold, feature_family="static_a",
        )
        train_profiles = {(r["corpus"], r["profile_id"]) for r in dm.views["train"]}
        val_profiles = {
            (r["corpus"], r["profile_id"])
            for name in ("val_care", "val_wsm")
            for r in dm.views[name]
        }
        test_profiles = {
            (r["corpus"], r["profile_id"])
            for name in ("test_care", "test_wsm")
            for r in dm.views[name]
        }
        assert not train_profiles & val_profiles
        assert not train_profiles & test_profiles
        assert not val_profiles & test_profiles


def test_missing_feature_is_filtered_and_temporal_values_are_unchanged(tmp_path: Path) -> None:
    care, wsm, manifest, registry = make_fixture(tmp_path)
    rows = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["static_a_status"] = "MISSING"
    write_csv(manifest, list(rows[0]), rows)
    dm = CAREWSMAudioDataModule(
        str(care), str(wsm), str(manifest), str(registry),
        feature_family="static_a",
    )
    assert len(dm.train_dataset) == 3
    temporal = CAREWSMAudioDataModule(
        str(care), str(wsm), str(manifest), str(registry),
        feature_family="temporal",
    )
    assert tuple(temporal.train_dataset[0]["inputs"]["native"].shape) == (5, 2)


def test_registration_collate_and_model_forward_backward(tmp_path: Path) -> None:
    care, wsm, manifest, registry = make_fixture(tmp_path)
    chimera_plugin.register()
    dm = CAREWSMAudioDataModule(
        str(care), str(wsm), str(manifest), str(registry),
        feature_family="static_a", batch_size=2,
    )
    batch = next(iter(dm.train_dataloader()))
    assert tuple(batch.inputs["native"].shape) == (2, 1, 2)
    assert tuple(batch.get_masks("sequence_mask").shape) == (2, 1)
    assert [meta["corpus"] for meta in batch.meta["sample_meta"]] == ["CARE", "CARE"]

    from audio.models.carewsm_audio_models import CAREWSMMDFormerModel, CAREWSMPooledMLPModel
    for model in (
        CAREWSMPooledMLPModel(input_dim=2, hidden_dim=8),
        CAREWSMMDFormerModel(input_dim=2, hidden_dim=8, num_heads=2, num_layers=1),
    ):
        output = model(batch)
        output.preds.sum().backward()
        assert output.preds.shape == (2, 2)
