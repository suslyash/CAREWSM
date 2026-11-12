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


def _normalise_turn_ids(raw_ids: Any, count: int, path: Path) -> tuple[str, ...]:
    """Validate and normalize one artifact's ordered turn identifiers."""
    identifiers = np.asarray(raw_ids)
    if identifiers.ndim != 1 or identifiers.shape[0] != count:
        raise ValueError(
            f"turn IDs must be one-dimensional with one ID per row: {path}"
        )

    normalized: list[str] = []
    for raw_id in identifiers.tolist():
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode("utf-8")
        if isinstance(raw_id, (float, np.floating)):
            if not np.isfinite(raw_id) or not float(raw_id).is_integer():
                raise ValueError(f"turn IDs must be finite scalar values: {path}")
            raw_id = int(raw_id)
        value = str(raw_id).strip()
        if not value:
            raise ValueError(f"turn IDs must not be empty: {path}")
        normalized.append(value)

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"turn IDs must be unique: {path}")
    return tuple(normalized)


def _load_values(
    path: Path, specification: Mapping[str, Any], family: str
) -> np.ndarray:
    """Load and validate one native CSV or NPZ feature value array."""
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


def load_feature_array(
    path: Path, specification: Mapping[str, Any], family: str
) -> np.ndarray:
    """Load one native value array without changing the single-family API."""
    return _load_values(path, specification, family)


def load_feature_component(
    path: Path, specification: Mapping[str, Any], family: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load one native value array and its required ordered turn identifiers."""
    values = _load_values(path, specification, family)
    if specification.get("format") == "csv":
        rows = read_csv_rows(path)
        if not rows or "turn_id" not in rows[0]:
            raise ValueError(
                f"CSV feature is missing required turn_id metadata: {path}"
            )
        raw_ids = [row["turn_id"] for row in rows]
    else:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "turn_ids" not in archive.files:
                    raise ValueError(
                        f"NPZ feature is missing required turn_ids: {path}"
                    )
                raw_ids = archive["turn_ids"]
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and "missing required turn_ids" in str(exc):
                raise
            raise ValueError(f"unreadable NPZ turn IDs: {path}") from exc
    return values, _normalise_turn_ids(raw_ids, values.shape[0], path)


class CAREWSMAudioDataset(Dataset[dict[str, Any]]):
    """Load one native family or concatenate aligned native turn features."""

    def __init__(
        self,
        records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        feature_family: str | list[str],
        feature_specs: Mapping[str, Mapping[str, Any]],
        pooling: str = "native",
    ) -> None:
        if isinstance(feature_family, str):
            families = [feature_family]
        elif isinstance(feature_family, list):
            families = feature_family
        else:
            raise ValueError("feature_family must be a string or ordered list")
        if not families:
            raise ValueError("CAREWSMAudioDataset requires feature_family")
        if len(families) > 1 and len(families) not in (2, 3, 4):
            raise ValueError(
                "fused feature_family lists must contain two to four families"
            )
        if len(set(families)) != len(families):
            raise ValueError("feature_family selections must not contain duplicates")
        unknown = set(families) - set(feature_specs)
        if unknown:
            raise ValueError(f"unknown feature family: {sorted(unknown)}")
        self.records = tuple(records)
        self.feature_family = feature_family
        self.feature_families = tuple(families)
        self.feature_specs = feature_specs
        if pooling not in ("native", "mean", "mean_std"):
            raise ValueError("pooling must be native, mean, or mean_std")
        self.pooling = pooling

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        for family in self.feature_families:
            if record.get(f"{family}_status") != "VALID":
                raise ValueError(
                    f"feature is not VALID for {record['corpus']}/{record['sample_id']}: {family}"
                )

        paths = {
            family: Path(record["feature_paths"][family])
            for family in self.feature_families
        }
        if len(self.feature_families) == 1:
            family = self.feature_families[0]
            values = load_feature_array(
                paths[family], self.feature_specs[family], family
            )
            feature_dimensions = {family: int(values.shape[1])}
            turn_ids = None
        else:
            components = [
                load_feature_component(
                    paths[family], self.feature_specs[family], family
                )
                for family in self.feature_families
            ]
            turn_ids = components[0][1]
            for family, (_, component_ids) in zip(
                self.feature_families[1:], components[1:]
            ):
                if component_ids != turn_ids:
                    raise ValueError(
                        f"turn IDs are not aligned for {record['corpus']}/{record['sample_id']}: "
                        f"{self.feature_families[0]} and {family}"
                    )
            if self.pooling == "mean_std":
                values = np.concatenate(
                    [
                        np.concatenate(
                            (component[0].mean(axis=0), component[0].std(axis=0))
                        )[None, :]
                        for component in components
                    ],
                    axis=1,
                )
            else:
                values = np.concatenate(
                    [component[0] for component in components], axis=1
                )
            feature_dimensions = {
                family: int(component[0].shape[1])
                for family, component in zip(self.feature_families, components)
            }

        if self.pooling == "mean":
            values = values.mean(axis=0, keepdims=True)
        elif self.pooling == "mean_std" and len(self.feature_families) == 1:
            values = np.concatenate(
                (values.mean(axis=0, keepdims=True), values.std(axis=0, keepdims=True)),
                axis=1,
            )
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
        metadata: dict[str, Any] = {
            "corpus": record["corpus"],
            "sample_id": record["sample_id"],
            "profile_id": record["profile_id"],
            "feature_family": self.feature_family,
            "feature_families": list(self.feature_families),
            "feature_dimensions": feature_dimensions,
            "feature_paths": {family: str(path) for family, path in paths.items()},
            "feature_dimension": int(tensor.shape[1]),
            "observed_targets": observed,
            "masks": {"sequence_mask": torch.ones(tensor.shape[0], dtype=torch.bool)},
        }
        if turn_ids is not None:
            metadata["turn_alignment"] = {
                "turn_ids": list(turn_ids),
                "families": list(self.feature_families),
            }
        else:
            metadata["feature_path"] = str(paths[self.feature_families[0]])
        return {"inputs": {"native": tensor}, "targets": targets, "meta": metadata}
