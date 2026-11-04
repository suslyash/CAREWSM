from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from audio.data.carewsm_audio_datamodule import CAREWSMAudioDataModule
from audio.data.carewsm_audio_dataset import CAREWSMAudioDataset, DataContractError


def _fixture(tmp_path):
    specs = {
        "static_a": {"modality": "audio", "file": "a.npz", "format": "npz", "embedding_keys": ["x"]},
        "static_b": {"modality": "audio", "file": "b.npz", "format": "npz", "embedding_keys": ["x"]},
        "temporal": {"modality": "audio", "file": "t.npz", "format": "npz", "embedding_keys": ["x"]},
    }
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"sources": specs}))
    manifest = tmp_path / "manifest.csv"
    fields = ["corpus", "sample_id", "profile_id", "video_id", "label_id", "subgroup", "depression", "parkinson", "depression_observed", "parkinson_observed"] + [f"{x}_status" for x in specs]
    records = []
    for corpus in ("CARE", "WSM"):
        root = tmp_path / corpus
        for split, profile, ids in (("train", "p1", ["tr"]), ("val", "p2", ["va"]), ("test", "p3", ["te"])):
            (root / "splits" / "folds").mkdir(parents=True, exist_ok=True)
            (root / "splits" / f"{split}.csv").write_text("sample_id,profile_id\n" + f"{ids[0]},{profile}\n")
            for sample in ids:
                directory = root / "extracted" / "DEPRESSION" / "" / f"PROFILE_{profile}" / f"VIDEO_{sample}"
                directory.mkdir(parents=True, exist_ok=True)
                np.savez(directory / "a.npz", x=np.ones((1, 2), np.float32))
                np.savez(directory / "b.npz", x=np.ones((1, 3), np.float32))
                np.savez(directory / "t.npz", x=np.ones((3, 2), np.float32))
                records.append([corpus, sample, profile, sample, "DEPRESSION", "", "1", "", "True", "False", "VALID", "VALID", "VALID"])
        for fold in range(5):
            (root / "splits" / "folds" / f"fold_{fold}_train.csv").write_text("sample_id,profile_id\ntr," + "p1\n")
            (root / "splits" / "folds" / f"fold_{fold}_val.csv").write_text("sample_id,profile_id\nva," + "p2\n")
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(records)
    return tmp_path / "CARE", tmp_path / "WSM", manifest, registry


def test_fold_masks_and_fixed_tests(tmp_path):
    care, wsm, manifest, registry = _fixture(tmp_path)
    dm = CAREWSMAudioDataModule(str(care), str(wsm), str(manifest), str(registry), feature_family="static_a")
    assert len(dm.views["train"]) == 2 and len(dm.views["val"]) == 2
    assert len(dm.test_dataloader()["CARE"].dataset) == 1 and len(dm.test_dataloader()["WSM"].dataset) == 1
    sample = dm.train_dataset[0]
    assert sample["targets"].tolist() == [1.0, 0.0]
    assert sample["meta"]["observed_targets"].tolist() == [True, False]


def test_static_order_and_temporal_pooling(tmp_path):
    care, wsm, manifest, registry = _fixture(tmp_path)
    dm = CAREWSMAudioDataModule(str(care), str(wsm), str(manifest), str(registry), feature_families=["static_b", "static_a"])
    sample = dm.train_dataset[0]
    assert tuple(sample["inputs"]["native"].shape) == (1, 5)
    assert sample["meta"]["feature_families"] == ["static_b", "static_a"]
    temporal = CAREWSMAudioDataModule(str(care), str(wsm), str(manifest), str(registry), feature_family="temporal", max_sequence_length=2)
    assert tuple(temporal.train_dataset[0]["inputs"]["native"].shape) == (2, 2)


def test_invalid_fusion_and_duplicates(tmp_path):
    care, wsm, manifest, registry = _fixture(tmp_path)
    with pytest.raises(ValueError):
        CAREWSMAudioDataModule(str(care), str(wsm), str(manifest), str(registry), feature_families=["static_a", "static_a"])
    with pytest.raises(DataContractError):
        ds = CAREWSMAudioDataset([{"corpus": "CARE", "sample_id": "tr", "profile_id": "p1", "video_id": "tr", "label_id": "DEPRESSION", "subgroup": "", "static_a_status": "VALID", "temporal_status": "VALID"}], ["static_a", "temporal"], {"CARE": care}, json.loads(registry.read_text())["sources"])
        ds[0]
