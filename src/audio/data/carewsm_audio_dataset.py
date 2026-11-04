"""Direct native feature Dataset for the CAREWSM audio pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV while preserving the source column names."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_feature_array(
    path: Path, specification: Mapping[str, Any], family: str
) -> np.ndarray:
    """Load and validate one native CSV or NPZ feature artifact."""
    if not path.exists():
        raise ValueError(f"feature artifact does not exist: {path}")

    if specification.get("format") == "csv":
        rows = read_csv_rows(path)
        if not rows:
            raise ValueError(f"feature artifact is empty: {path}")
        metadata_columns = set(specification.get("metadata", []))
        feature_columns = [
            column for column in rows[0] if column not in metadata_columns
        ]
        if not feature_columns:
            raise ValueError(f"feature artifact has no numeric columns: {path}")
        try:
            values = np.asarray(
                [[float(row[column]) for column in feature_columns] for row in rows],
                dtype=np.float32,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid CSV feature schema: {path}") from exc
    else:
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = specification.get("embedding_keys", [])
                key = (
                    "embed"
                    if family == "emotionmsp_embeds"
                    else (keys[0] if keys else None)
                )
                if key is None or key not in archive.files:
                    raise ValueError(f"required NPZ key {key!r} is missing: {path}")
                values = np.asarray(archive[key], dtype=np.float32)
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("required NPZ"):
                raise
            raise ValueError(f"unreadable NPZ feature artifact: {path}") from exc

    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"feature must be a non-empty 2-D array: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"feature contains non-finite values: {path}")
    return values


class CAREWSMAudioDataset(Dataset[dict[str, Any]]):
    """Load one configured native feature family without changing its shape."""

    def __init__(
        self,
        records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        feature_family: str,
        feature_specs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if not feature_family:
            raise ValueError("CAREWSMAudioDataset requires feature_family")
        if feature_family not in feature_specs:
            raise ValueError(f"unknown feature family: {feature_family}")
        self.records = tuple(records)
        self.feature_family = feature_family
        self.feature_spec = feature_specs[feature_family]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        family = self.feature_family
        if record.get(f"{family}_status") != "VALID":
            raise ValueError(
                f"feature is not VALID for {record['corpus']}/{record['sample_id']}: {family}"
            )
        path = Path(record["feature_paths"][family])
        values = load_feature_array(path, self.feature_spec, family)
        tensor = torch.from_numpy(values.astype(np.float32, copy=False))

        observed = torch.tensor(
            [record["depression_observed"], record["parkinson_observed"]],
            dtype=torch.bool,
        )
        targets = torch.tensor(
            [
                record["depression"] if observed[0] else 0.0,
                record["parkinson"] if observed[1] else 0.0,
            ],
            dtype=torch.float32,
        )
        return {
            "inputs": {"native": tensor},
            "targets": targets,
            "meta": {
                "corpus": record["corpus"],
                "sample_id": record["sample_id"],
                "profile_id": record["profile_id"],
                "feature_family": family,
                "feature_dimension": int(tensor.shape[1]),
                "feature_path": str(path),
                "observed_targets": observed,
                "masks": {
                    "sequence_mask": torch.ones(tensor.shape[0], dtype=torch.bool)
                },
            },
        }
