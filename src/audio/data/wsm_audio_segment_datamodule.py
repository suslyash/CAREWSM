from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audio.data.wsm_audio_segment_dataset import WSMAudioSegmentDataset, collate_wsm_audio_features
from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule
from common.utils.segment_index import TASKS, build_wsm_multitask_segment_index, build_wsm_segment_index


@dataclass
class WSMAudioSegmentDataModule(DataModule):
    data_root: str = "/media/maxim/Databases/WSM_NEW"
    task: str = "multitask"
    feature_cache_root: str | None = None
    test_filters: tuple[str, ...] = ("none", "soft", "hard")
    feature_extractor_type: str = "transformers_ssl"
    feature_model_name: str = "microsoft/wavlm-base-plus"
    feature_sample_rate: int = 16000
    feature_layer: int = -1
    temporal_pool: int = 8
    class_names: tuple[str, ...] = ("healthy", "pathological")

    def __post_init__(self) -> None:
        self.collate_fn = collate_wsm_audio_features
        self.is_multitask = self.task in {"all", "multitask"}
        self.records = (
            build_wsm_multitask_segment_index(self.data_root)
            if self.is_multitask
            else build_wsm_segment_index(self.data_root, task=self.task)
        )
        feature_cache_root = self.feature_cache_root or str(Path(self.data_root) / "features")
        common = {
            "feature_cache_root": feature_cache_root,
            "feature_extractor_type": self.feature_extractor_type,
            "feature_model_name": self.feature_model_name,
            "feature_sample_rate": self.feature_sample_rate,
            "feature_layer": self.feature_layer,
            "temporal_pool": self.temporal_pool,
        }

        train_records = self.records[self.records["split"] == "train"].reset_index(drop=True)
        dev_records = self.records[self.records["split"] == "dev"].reset_index(drop=True)
        self.train_dataset = WSMAudioSegmentDataset(records=train_records, split_name="train", **common)
        if self.is_multitask:
            self.val_dataset = {
                f"dev_{task}": WSMAudioSegmentDataset(
                    records=dev_records[dev_records["task"] == task].reset_index(drop=True),
                    split_name=f"dev_{task}",
                    **common,
                )
                for task in TASKS
            }
        else:
            self.val_dataset = {"dev": WSMAudioSegmentDataset(records=dev_records, split_name="dev", **common)}

        self.test_dataset = {}
        for test_filter in self.test_filters:
            test_records = self.records[self.records["split"] == "test"].copy()
            if str(test_filter) != "none":
                test_records = test_records[test_records[f"{test_filter}_filter"].fillna(0).astype(int) == 1]

            if self.is_multitask:
                for task in TASKS:
                    name = f"test_{test_filter}_{task}"
                    self.test_dataset[name] = WSMAudioSegmentDataset(
                        records=test_records[test_records["task"] == task].reset_index(drop=True),
                        split_name=name,
                        **common,
                    )
            else:
                name = f"test_{test_filter}"
                self.test_dataset[name] = WSMAudioSegmentDataset(
                    records=test_records.reset_index(drop=True),
                    split_name=name,
                    **common,
                )

        self.val_dataset.update(self.test_dataset)
        self.feature_dim = int(self.train_dataset.feature_dim)
        self.class_weights = self._compute_class_weights(train_records)

    def describe_context(self, context: Any) -> None:
        context.set("data.feature_dim", self.feature_dim)
        context.set("data.audio_feature_dim", self.feature_dim)
        context.set("data.num_classes", len(self.class_names))
        context.set("data.class_names", list(self.class_names))
        context.set("data.task", self.task)
        context.set("data.task_names", list(TASKS))
        context.set("data.num_tasks", len(TASKS))
        context.set("data.class_weights", self.class_weights)
        context.set("data.test_filters", list(self.test_filters))

    def _compute_class_weights(self, records: pd.DataFrame) -> list[list[float]]:
        weights = np.ones((len(TASKS), len(self.class_names)), dtype=np.float64)
        for task_id, task in enumerate(TASKS):
            labels = records.loc[records["task"] == task, "label"].to_numpy(dtype=np.int64)
            if labels.size == 0:
                continue
            counts = np.bincount(labels, minlength=len(self.class_names)).astype(np.float64)
            task_weights = counts.sum() / np.maximum(counts, 1.0)
            weights[task_id] = task_weights / np.mean(task_weights)

        return weights.astype(float).tolist()


@DATAMODULES.register("wsm_audio_segment_datamodule")
def wsm_audio_segment_datamodule(context: Any | None = None, **params: Any) -> WSMAudioSegmentDataModule:
    return WSMAudioSegmentDataModule(**params)
