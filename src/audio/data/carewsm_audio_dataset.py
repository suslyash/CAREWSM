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
    """Load native feature arrays and return Chimera-compatible samples."""

    def __init__(
        self,
        records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        feature_families: list[str] | tuple[str, ...],
        feature_specs: Mapping[str, Mapping[str, Any]],
        max_sequence_length: int | None = None,
    ) -> None:
        self.records = tuple(records)
        self.feature_families = tuple(feature_families)
        self.feature_specs = feature_specs
        self.max_sequence_length = max_sequence_length

        if not self.feature_families:
            raise ValueError("at least one feature family is required")
        if len(set(self.feature_families)) != len(self.feature_families):
            raise ValueError("duplicate feature families are not allowed")
        unknown = set(self.feature_families) - set(self.feature_specs)
        if unknown:
            raise ValueError(f"unknown feature family: {sorted(unknown)}")
        if max_sequence_length is not None and max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")

    def __len__(self) -> int:
        return len(self.records)

    def _load_record_features(
        self, record: Mapping[str, Any]
    ) -> list[tuple[np.ndarray, Path]]:
        loaded: list[tuple[np.ndarray, Path]] = []
        for family in self.feature_families:
            if record.get(f"{family}_status") != "VALID":
                raise ValueError(
                    f"feature is not VALID for {record['corpus']}/{record['sample_id']}: {family}"
                )
            path = Path(record["feature_paths"][family])
            values = load_feature_array(path, self.feature_specs[family], family)
            loaded.append((values, path))
        return loaded

    def _combine_features(
        self, loaded: list[tuple[np.ndarray, Path]]
    ) -> tuple[np.ndarray, list[int]]:
        arrays = [values for values, _ in loaded]
        dimensions = [int(values.shape[1]) for values in arrays]
        if len(arrays) > 1:
            if any(values.shape[0] != 1 for values in arrays):
                raise ValueError(
                    "multiple feature families require genuine static T=1 arrays; "
                    "temporal alignment is not defined"
                )
            return np.concatenate(arrays, axis=1), dimensions

        values = arrays[0]
        if self.max_sequence_length is None or len(values) <= self.max_sequence_length:
            return values, dimensions

        edges = np.linspace(
            0, len(values), self.max_sequence_length + 1, dtype=np.int64
        )
        pooled = np.stack(
            [
                values[edges[i] : edges[i + 1]].mean(axis=0)
                for i in range(self.max_sequence_length)
            ]
        )
        return pooled, dimensions

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        loaded = self._load_record_features(record)
        values, dimensions = self._combine_features(loaded)
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
        paths = [str(path) for _, path in loaded]

        return {
            "inputs": {"native": tensor},
            "targets": targets,
            "meta": {
                "corpus": record["corpus"],
                "sample_id": record["sample_id"],
                "profile_id": record["profile_id"],
                "feature_families": list(self.feature_families),
                "feature_dimensions": dimensions,
                "feature_paths": paths,
                "observed_targets": observed,
                "masks": {
                    "sequence_mask": torch.ones(tensor.shape[0], dtype=torch.bool)
                },
            },
        }
