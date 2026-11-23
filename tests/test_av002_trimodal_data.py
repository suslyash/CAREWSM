"""Focused AV-002 data-layer checks without training or GPU work."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

import chimera_plugin
from chimera_ml.core.registry import DATAMODULES

from fusion.data.carewsm_trimodal_dataset import CAREWSMTrimodalDataset, trimodal_collate


def write_artifact(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"shared_features": np.asarray(values, dtype=np.float32), "logits": np.zeros(2)}, handle)


def make_record(tmp_path, sample_id="sample"):
    paths = {}
    for modality, values in {"audio": [[1, 2]], "video": [[1, 2], [3, 4]], "text": [[5, 6]]}.items():
        path = tmp_path / f"{modality}.pickle"
        write_artifact(path, values)
        paths[modality] = str(path)
    return {
        "corpus": "WSM", "sample_id": sample_id, "profile_id": "p", "video_id": "v",
        "label_id": "CONTROL", "subgroup": "g", "artifact_paths": paths,
        "targets": {"depression": 0.0, "parkinson": None},
        "target_provenance": "WSM videos.csv y_* columns",
    }


def test_joined_modalities_masks_and_padding(tmp_path):
    record = make_record(tmp_path)
    sample = CAREWSMTrimodalDataset([record], "audio_id1", "video_4", "text_id9")[0]
    assert tuple(sample["inputs"]["audio"].shape) == (1, 2)
    assert tuple(sample["inputs"]["video"].shape) == (2, 2)
    assert torch.equal(sample["meta"]["masks"]["target_mask"], torch.tensor([True, False]))
    batch = trimodal_collate([sample])
    assert torch.equal(batch.masks["video_lengths"], torch.tensor([2]))
    assert torch.equal(batch.masks["video_valid_mask"], torch.tensor([[True, True]]))


def test_missing_or_invalid_modality_is_not_imputed(tmp_path):
    record = make_record(tmp_path)
    record["artifact_paths"]["text"] = str(tmp_path / "missing.pickle")
    with pytest.raises(FileNotFoundError):
        CAREWSMTrimodalDataset([record], "audio_id1", "video_4", "text_id9")[0]
    write_artifact(tmp_path / "invalid.pickle", [[np.nan, 1]])
    record["artifact_paths"]["text"] = str(tmp_path / "invalid.pickle")
    with pytest.raises(ValueError, match="NaN/Inf"):
        CAREWSMTrimodalDataset([record], "audio_id1", "video_4", "text_id9")[0]


def test_trimodal_registration_and_care_unknown_masks(tmp_path):
    chimera_plugin.register()
    assert DATAMODULES.get("carewsm_trimodal") is not None
    record = make_record(tmp_path)
    record["corpus"] = "CARE"
    record["targets"] = {"depression": None, "parkinson": None}
    sample = CAREWSMTrimodalDataset([record], "audio_id1", "video_4", "text_id9")[0]
    assert not sample["meta"]["masks"]["target_mask"].any()
    assert torch.isnan(sample["targets"]).all()
