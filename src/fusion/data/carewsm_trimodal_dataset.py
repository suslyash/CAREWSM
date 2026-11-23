"""Direct Dataset for audited CAREWSM audio, video, and text artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("audio", "video", "text")
TARGETS = ("depression", "parkinson")


def load_pickle_array(path: Path, key: str) -> np.ndarray:
    """Load one representation array and reject invalid or supervised outputs."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if key not in payload:
        raise ValueError(f"missing representation key {key!r}: {path}")
    values = np.asarray(payload[key])
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"representation must be non-empty [T,D]: {path}")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"representation must have a floating dtype: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"representation contains NaN/Inf: {path}")
    return values


class CAREWSMTrimodalDataset(Dataset[dict[str, Any]]):
    """Load exactly one explicitly selected source for each modality."""

    def __init__(
        self,
        records,
        audio_source,
        video_source,
        text_source,
        representation_key="shared_features",
    ):
        self.records = tuple(records)
        self.sources = {
            "audio": audio_source,
            "video": video_source,
            "text": text_source,
        }
        self.representation_key = representation_key

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        inputs, lengths, paths = {}, {}, {}
        for modality in MODALITIES:
            path = Path(record["artifact_paths"][modality])
            values = load_pickle_array(path, self.representation_key)
            inputs[modality] = torch.from_numpy(values)
            lengths[modality] = int(values.shape[0])
            paths[modality] = str(path)
        observed = torch.tensor(
            [record["targets"][target] is not None for target in TARGETS],
            dtype=torch.bool,
        )
        targets = torch.tensor(
            [
                record["targets"][target] if observed[idx] else float("nan")
                for idx, target in enumerate(TARGETS)
            ],
            dtype=torch.float32,
        )
        metadata = {
            "corpus": record["corpus"],
            "sample_id": record["sample_id"],
            "profile_id": record["profile_id"],
            "video_id": record["video_id"],
            "label_id": record["label_id"],
            "subgroup": record["subgroup"],
            "sources": dict(self.sources),
            "representation_key": self.representation_key,
            "artifact_paths": paths,
            "lengths": lengths,
            "target_provenance": record["target_provenance"],
            "provenance_status": "UNKNOWN / BLOCKED: upstream FIT/SELECTION",
            "observed_targets": observed,
            "masks": {
                "sequence_mask": torch.ones(lengths["video"], dtype=torch.bool),
                "target_mask": observed,
                "observed_targets": observed,
            },
        }
        return {"inputs": inputs, "targets": targets, "meta": metadata}


def trimodal_collate(batch: list[dict[str, Any]]) -> Any:
    """Use Chimera padding and add modality-specific lengths and valid masks."""
    from chimera_ml.core.batch import Batch
    from chimera_ml.data.masking_collate import MaskingCollate

    collated = MaskingCollate()(batch)
    lengths = {
        modality: torch.tensor(
            [int(sample["inputs"][modality].shape[0]) for sample in batch],
            dtype=torch.long,
        )
        for modality in MODALITIES
    }
    video_max = int(lengths["video"].max().item())
    video_valid = torch.arange(video_max).unsqueeze(0) < lengths["video"].unsqueeze(1)
    masks = dict(collated.masks or {})
    masks.update(
        {
            "video_lengths": lengths["video"],
            "video_valid_mask": video_valid,
            "video_padding_mask": ~video_valid,
            "audio_lengths": lengths["audio"],
            "text_lengths": lengths["text"],
            "target_mask": torch.stack(
                [sample["meta"]["masks"]["target_mask"] for sample in batch]
            ),
            "observed_targets": torch.stack(
                [sample["meta"]["masks"]["observed_targets"] for sample in batch]
            ),
        }
    )
    return Batch(
        inputs=collated.inputs,
        targets=collated.targets,
        masks=masks,
        meta=collated.meta,
    )
