from __future__ import annotations

from pathlib import Path
from typing import Any

import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from audio.features.wsm_audio_feature_extractor import WSMAudioFeatureExtractor


def collate_wsm_audio_features(batch: list[dict[str, Any]]) -> Any:
    from chimera_ml.core.batch import Batch

    audio, audio_mask = _pad_sequences([item["inputs"]["audio"] for item in batch])
    return Batch(
        inputs={
            "audio": audio,
            "audio_cls": torch.stack([item["inputs"]["audio_cls"] for item in batch]),
            "task_ids": torch.tensor([item["inputs"]["task_id"] for item in batch], dtype=torch.long),
        },
        targets=torch.tensor([item["target"] for item in batch], dtype=torch.long),
        masks={"audio_mask": audio_mask},
        meta={"sample_meta": [item["meta"] for item in batch]},
    )


class WSMAudioSegmentDataset(Dataset):
    def __init__(
        self,
        *,
        records: pd.DataFrame,
        feature_cache_root: str | Path,
        split_name: str,
        feature_extractor_type: str = "transformers_ssl",
        feature_model_name: str = "microsoft/wavlm-base-plus",
        feature_sample_rate: int = 16000,
        feature_layer: int = -1,
        temporal_pool: int = 8,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.feature_cache_root = Path(feature_cache_root)
        self.split_name = str(split_name)
        self.cache_split_name = str(self.records.iloc[0]["split"]) if len(self.records) else self.split_name
        extractor_tag = str(feature_extractor_type).replace("/", "__").replace(":", "_")
        model_tag = str(feature_model_name).replace("/", "__").replace(":", "_")
        self.cache_dir = (
            self.feature_cache_root
            / "audio"
            / extractor_tag
            / model_tag
            / f"layer{feature_layer}_pool{temporal_pool}"
            / self.cache_split_name
        )
        self.feature_paths = [
            self.cache_dir / str(row.video_id) / f"{Path(str(row.segment_file)).stem}.pkl"
            for row in self.records.itertuples(index=False)
        ]

        missing = {path for path in self.feature_paths if not path.exists()}
        if missing:
            extractor = WSMAudioFeatureExtractor(
                extractor_type=feature_extractor_type,
                model_name=feature_model_name,
                sample_rate=feature_sample_rate,
                layer=feature_layer,
                temporal_pool=temporal_pool,
            )
            self._build_feature_cache(extractor, missing)

        self.feature_dim = self._read_feature_dim()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        with self.feature_paths[index].open("rb") as f:
            features = pickle.load(f)

        return {
            "inputs": {
                "audio": torch.nan_to_num(features["audio_temporal"].float(), nan=0.0, posinf=0.0, neginf=0.0),
                "audio_cls": torch.nan_to_num(features["audio_cls"].float(), nan=0.0, posinf=0.0, neginf=0.0),
                "task_id": int(row.task_id),
            },
            "target": int(row.label),
            "meta": {
                "task": str(row.task),
                "task_id": int(row.task_id),
                "split": str(row.split),
                "source_split": str(row.source_split),
                "video_id": str(row.video_id),
                "segment_file": str(row.segment_file),
                "label": int(row.label),
                "soft_filter": row.soft_filter,
                "hard_filter": row.hard_filter,
            },
        }

    def _build_feature_cache(self, extractor: WSMAudioFeatureExtractor, missing: set[Path]) -> None:
        rows = list(zip(self.records.itertuples(index=False), self.feature_paths, strict=True))
        for row, cache_path in tqdm(rows, desc=f"cache WSM audio features [{self.split_name}]"):
            if cache_path not in missing:
                continue

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = extractor.extract(Path(row.segment_path).with_suffix(".wav"))
            payload["audio_temporal"] = torch.nan_to_num(payload["audio_temporal"].float(), nan=0.0, posinf=0.0, neginf=0.0)
            payload["audio_cls"] = torch.nan_to_num(payload["audio_cls"].float(), nan=0.0, posinf=0.0, neginf=0.0)
            payload["meta"] = {
                "task": str(row.task),
                "split": str(row.split),
                "source_split": str(row.source_split),
                "video_id": str(row.video_id),
                "segment_file": str(row.segment_file),
                "label": int(row.label),
                "soft_filter": row.soft_filter,
                "hard_filter": row.hard_filter,
            }
            with cache_path.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _read_feature_dim(self) -> int:
        with self.feature_paths[0].open("rb") as f:
            item = pickle.load(f)
        return int(item["audio_temporal"].shape[-1])


def _pad_sequences(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_t = max(int(sequence.shape[0]) for sequence in sequences)
    dim = int(sequences[0].shape[-1])
    padded = torch.zeros(len(sequences), max_t, dim, dtype=torch.float32)
    mask = torch.zeros(len(sequences), max_t, dtype=torch.bool)

    for index, sequence in enumerate(sequences):
        length = int(sequence.shape[0])
        padded[index, :length] = sequence.float()
        mask[index, :length] = True

    return padded, mask
