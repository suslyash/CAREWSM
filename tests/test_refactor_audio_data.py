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
        "static_d": {
            "modality": "audio",
            "file": "static_d.npz",
            "format": "npz",
            "embedding_keys": ["embed"],
            "metadata": ["turn_ids"],
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
        "static_d_status",
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
                        "static_d_status": "VALID",
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
                    directory / "static_a.npz",
                    embed=np.array([[1.0, 2.0]], dtype=np.float32),
                    turn_ids=np.array([1], dtype=np.int64),
                )
                np.savez(
                    directory / "static_b.npz",
                    embed=np.array([[10.0, 11.0, 12.0]], dtype=np.float32),
                    turn_ids=np.array([1], dtype=np.int64),
                )
                np.savez(
                    directory / "static_c.npz",
                    embed=np.array([[20.0, 21.0, 22.0, 23.0]], dtype=np.float32),
                    turn_ids=np.array([1], dtype=np.int64),
                )
                np.savez(
                    directory / "static_d.npz",
                    embed=np.array([[30.0, 31.0, 32.0, 33.0, 34.0]], dtype=np.float32),
                    turn_ids=np.array([1], dtype=np.int64),
                )
                np.savez(
                    directory / "temporal.npz",
                    embed=np.ones((5, 2), dtype=np.float32),
                    turn_ids=np.arange(1, 6, dtype=np.int64),
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
        feature_family=["static_a", "static_b"], batch_size=2,
    )
    batch = next(iter(dm.train_dataloader()))
    assert tuple(batch.inputs["native"].shape) == (2, 1, 5)
    assert tuple(batch.get_masks("sequence_mask").shape) == (2, 1)
    assert [meta["corpus"] for meta in batch.meta["sample_meta"]] == ["CARE", "CARE"]

    from audio.models.carewsm_audio_models import CAREWSMMDFormerModel, CAREWSMPooledMLPModel
    for model in (
        CAREWSMPooledMLPModel(input_dim=5, hidden_dim=8),
        CAREWSMMDFormerModel(input_dim=5, hidden_dim=8, num_heads=2, num_layers=1),
    ):
        output = model(batch)
        output.preds.sum().backward()
        assert output.preds.shape == (2, 2)


def test_two_three_and_four_family_fusion_preserves_order_and_metadata(tmp_path: Path) -> None:
    pair = build_dm(tmp_path / "pair", feature_family=["static_a", "static_b"])
    triple = build_dm(
        tmp_path / "triple",
        feature_family=["static_a", "static_b", "static_c"],
    )
    four = build_dm(
        tmp_path / "four",
        feature_family=["static_a", "static_b", "static_c", "static_d"],
    )

    pair_sample = pair.train_dataset[0]
    triple_sample = triple.train_dataset[0]
    four_sample = four.train_dataset[0]
    assert pair_sample["inputs"]["native"].tolist() == [[1.0, 2.0, 10.0, 11.0, 12.0]]
    assert tuple(triple_sample["inputs"]["native"].shape) == (1, 9)
    assert tuple(four_sample["inputs"]["native"].shape) == (1, 14)
    assert four_sample["meta"]["feature_families"] == [
        "static_a", "static_b", "static_c", "static_d"
    ]
    assert four_sample["meta"]["feature_dimensions"] == {
        "static_a": 2, "static_b": 3, "static_c": 4, "static_d": 5
    }
    assert four_sample["meta"]["turn_alignment"]["turn_ids"] == ["1"]


def test_fused_variable_t_preserves_sequence_mask(tmp_path: Path) -> None:
    dm = build_dm(tmp_path, feature_family=["static_a", "static_b"])
    record = dm.train_dataset.records[0]
    for family, dimension in (("static_a", 2), ("static_b", 3)):
        np.savez(
            record["feature_paths"][family],
            embed=np.arange(5 * dimension, dtype=np.float32).reshape(5, dimension),
            turn_ids=np.arange(1, 6, dtype=np.int64),
        )
    sample = dm.train_dataset[0]
    assert tuple(sample["inputs"]["native"].shape) == (5, 5)
    assert sample["meta"]["masks"]["sequence_mask"].tolist() == [True] * 5


def test_same_length_mismatched_turn_ids_fail(tmp_path: Path) -> None:
    dm = build_dm(tmp_path, feature_family=["static_a", "static_b"])
    record = dm.train_dataset.records[0]
    np.savez(
        record["feature_paths"]["static_b"],
        embed=np.array([[10.0, 11.0, 12.0]], dtype=np.float32),
        turn_ids=np.array([2], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="not aligned"):
        dm.train_dataset[0]


@pytest.mark.parametrize("ids", ([1, 1], [""], None))
def test_missing_duplicate_or_malformed_turn_ids_fail(tmp_path: Path, ids) -> None:
    dm = build_dm(tmp_path, feature_family=["static_a", "static_b"])
    record = dm.train_dataset.records[0]
    path = Path(record["feature_paths"]["static_b"])
    values = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    if ids is None:
        np.savez(path, embed=values)
    else:
        np.savez(path, embed=values, turn_ids=np.asarray(ids))
    with pytest.raises(ValueError, match="turn"):
        dm.train_dataset[0]


def test_fusion_filters_rows_missing_any_selected_family(tmp_path: Path) -> None:
    care, wsm, manifest, registry = make_fixture(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["static_b_status"] = "MISSING"
    write_csv(manifest, list(rows[0]), rows)
    dm = CAREWSMAudioDataModule(
        str(care), str(wsm), str(manifest), str(registry),
        feature_family=["static_a", "static_b"],
    )
    assert len(dm.train_dataset) == 3


@pytest.mark.parametrize("selection", ([], ["static_a"], ["static_a", "static_a"], ["unknown", "static_b"]))
def test_invalid_fusion_selections_fail(tmp_path: Path, selection: list[str]) -> None:
    with pytest.raises(ValueError):
        build_dm(tmp_path, feature_family=selection)


def test_static_pooling_per_family_mean_and_population_sd(tmp_path: Path) -> None:
    """Two aligned turns must collapse to one recording vector without mixing families."""
    for pooling, expected in (
        ("native", [[1., 2., 10., 11., 12.], [3., 4., 14., 15., 16.]]),
        ("mean", [[2., 3., 12., 13., 14.]]),
        ("mean_std", [[2., 3., 1., 1., 12., 13., 14., 2., 2., 2.]]),
    ):
        dm = build_dm(
            tmp_path / pooling,
            feature_family=["static_a", "static_b"],
            pooling=pooling,
        )
        record = dm.train_dataset.records[0]
        np.savez(
            record["feature_paths"]["static_a"],
            embed=np.array([[1., 2.], [3., 4.]], dtype=np.float32),
            turn_ids=np.array([1, 2]),
        )
        np.savez(
            record["feature_paths"]["static_b"],
            embed=np.array([[10., 11., 12.], [14., 15., 16.]], dtype=np.float32),
            turn_ids=np.array([1, 2]),
        )
        sample = dm.train_dataset[0]
        np.testing.assert_allclose(
            sample["inputs"]["native"].numpy(),
            np.array(expected, dtype=np.float32),
            rtol=0, atol=0,
        )
        assert sample["meta"]["masks"]["sequence_mask"].shape[0] == len(expected)
        assert sample["meta"]["observed_targets"].tolist() == [True, False]


def test_static_mean_std_singleton_is_zero(tmp_path: Path) -> None:
    dm = build_dm(
        tmp_path,
        feature_family=["static_a", "static_b"],
        pooling="mean_std",
    )
    sample = dm.train_dataset[0]
    np.testing.assert_array_equal(
        sample["inputs"]["native"].numpy(),
        np.array([[1., 2., 0., 0., 10., 11., 12., 0., 0., 0.]], dtype=np.float32),
    )
    assert sample["inputs"]["native"].shape == (1, 10)



def test_independent_family_mean_preserves_family_boundaries_and_metadata(tmp_path: Path) -> None:
    dm = build_dm(
        tmp_path,
        feature_family=["static_a", "static_b", "static_c", "static_d"],
        pooling="mean",
        independent_family_mean=True,
        cohort_feature_family=["static_a", "static_b", "static_c", "static_d"],
    )
    sample = dm.train_dataset[0]
    assert tuple(sample["inputs"]["native"].shape) == (1, 14)
    assert sample["meta"]["aggregation"] == "independent_mean"
    assert list(sample["meta"]["feature_dimensions"].values()) == [2, 3, 4, 5]
    assert [len(columns) for columns in sample["meta"]["feature_columns"].values()] == [2, 3, 4, 5]


def test_independent_family_mean_does_not_require_aligned_turn_ids(tmp_path: Path) -> None:
    care, _, manifest, registry = make_fixture(tmp_path)
    record = {
        "corpus": "CARE",
        "sample_id": "tr_CARE_1",
        "profile_id": "profile_tr_CARE_1",
        "video_id": "tr_CARE_1",
        "label_id": "DEPRESSION",
        "subgroup": "group",
        "video_dir": "CARE/extracted/DEPRESSION/group/PROFILE_profile_tr_CARE_1/VIDEO_tr_CARE_1",
        "depression": 1.0,
        "parkinson": float("nan"),
        "depression_observed": True,
        "parkinson_observed": False,
    }
    from audio.data.carewsm_audio_dataset import CAREWSMAudioDataset

    path = care / "extracted/DEPRESSION/group/PROFILE_profile_tr_CARE_1/VIDEO_tr_CARE_1"
    record["feature_paths"] = {
        "static_a": str(path / "static_a.npz"),
        "static_b": str(path / "static_b.npz"),
    }
    np.savez(
        path / "static_b.npz",
        embed=np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32),
        turn_ids=np.array([9, 10]),
    )
    specs = json.loads(registry.read_text())["sources"]
    dataset = CAREWSMAudioDataset(
        [record],
        ["static_a", "static_b"],
        specs,
        pooling="mean",
        independent_family_mean=True,
    )
    sample = dataset[0]
    assert sample["inputs"]["native"].tolist() == [[1.0, 2.0, 15.0, 16.0, 17.0]]
    assert sample["meta"]["aggregation"] == "independent_mean"


def test_static_gated_fusion_supports_seven_family_dimensions() -> None:
    from audio.models.carewsm_static_models import CAREWSMStaticGatedFusionModel

    dimensions = [2, 3, 4, 5, 6, 7, 8]
    model = CAREWSMStaticGatedFusionModel(
        input_dim=sum(dimensions), family_dims=dimensions, hidden_dim=8, dropout=0.0
    )
    from chimera_ml.core.batch import Batch

    values = torch.randn(3, 1, sum(dimensions))
    batch = Batch(
        inputs={"native": values},
        targets=None,
        masks={"sequence_mask": torch.ones(3, 1, dtype=torch.bool)},
    )
    output = model(batch)
    assert output.preds.shape == (3, 2)
    output.preds.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
