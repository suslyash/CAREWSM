"""Chimera-ML DataModule adapter for the joint CARE+WSM fold protocol."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule
from torch.utils.data import Dataset

from audio.data.joint_datamodule import JointDataModule, Sample
from audio.features.representations import (
    native_representation,
    pooled_native_representation,
)


class NativeDataset(Dataset):
    def __init__(
        self,
        dm: JointDataModule,
        rows: tuple[Sample, ...],
        family: str,
        max_sequence_length: int | None = None,
    ):
        self.dm = dm
        self.rows = rows
        self.family = family
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.rows[index]
        if self.max_sequence_length is not None:
            representation = pooled_native_representation(
                self.dm,
                sample,
                self.family,
                self.max_sequence_length,
            )
        else:
            representation = native_representation(
                self.dm,
                sample,
                self.family,
            )

        values = torch.from_numpy(representation.values[0])
        observed = torch.tensor(
            [sample.depression_observed, sample.parkinson_observed],
            dtype=torch.bool,
        )
        targets = torch.tensor(
            [
                sample.depression if sample.depression_observed else 0.0,
                sample.parkinson if sample.parkinson_observed else 0.0,
            ],
            dtype=torch.float32,
        )
        return {
            "inputs": {"native": values},
            "targets": targets,
            "meta": {
                "corpus": sample.corpus,
                "sample_id": sample.sample_id,
                "profile_id": sample.profile_id,
                "family": self.family,
                "observed_targets": observed,
                "masks": {
                    "sequence_mask": torch.ones(
                        values.shape[0],
                        dtype=torch.bool,
                    )
                },
            },
        }


class CarewsmJointDataModule(DataModule):
    """Registered DataModule with fold VAL and fixed CARE/WSM TEST loaders."""

    def __init__(
        self,
        care_root: str,
        wsm_root: str,
        manifest: str,
        registry: str,
        fold: int = 0,
        feature_family: str = "egemaps_lld",
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = False,
        max_sequence_length: int | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
    ):
        self.fold_index = int(fold)
        self.feature_family = feature_family
        self.joint = JointDataModule(care_root, wsm_root, manifest, registry)
        views = self.joint.fold(self.fold_index, [feature_family])
        limits = {
            "train": max_train_samples,
            "val": max_val_samples,
            "care_val": max_val_samples,
            "wsm_val": max_val_samples,
            "care_test": max_test_samples,
            "wsm_test": max_test_samples,
        }

        def limited_rows(key: str, view: Any) -> tuple[Sample, ...]:
            limit = limits.get(key)
            if limit is None:
                return view.rows
            if key in {"train", "val"}:
                return tuple(
                    row
                    for corpus in ("CARE", "WSM")
                    for row in tuple(
                        row for row in view.rows if row.corpus == corpus
                    )[:limit]
                )
            return view.rows[:limit]

        self.views = {
            key: replace(view, rows=limited_rows(key, view))
            for key, view in views.items()
        }

        def dataset(key: str) -> NativeDataset:
            return NativeDataset(
                self.joint,
                self.views[key].rows,
                feature_family,
                max_sequence_length,
            )

        super().__init__(
            train_dataset=dataset("train"),
            val_dataset={
                "joint": dataset("val"),
                "CARE": dataset("care_val"),
                "WSM": dataset("wsm_val"),
            },
            test_dataset={
                "CARE": dataset("care_test"),
                "WSM": dataset("wsm_test"),
            },
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle_train=shuffle_train,
        )

    def describe_context(self, context: Any) -> None:
        context.set("data.module", self)
        context.set("data.fold", self.fold_index)
        context.set("data.feature_family", self.feature_family)
        for key in ("train", "val", "care_test", "wsm_test"):
            context.set("data." + key + "_count", len(self.views[key]))


@DATAMODULES.register("carewsm_joint")
def carewsm_joint(
    context: Any | None = None,
    **params: Any,
) -> CarewsmJointDataModule:
    """Build the CAREWSM DataModule from YAML parameters."""
    return CarewsmJointDataModule(**params)
