"""Fold-aware CARE+WSM DataModule for native audio features."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule

from audio.data.carewsm_audio_dataset import CAREWSMAudioDataset


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a source CSV with its original fields."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_label(value: str | None) -> float | None:
    """Parse an independent binary target while preserving missing labels."""
    if value is None or not str(value).strip():
        return None
    parsed = float(value)
    if parsed not in (0.0, 1.0):
        raise ValueError(f"label must be binary or missing: {value!r}")
    return parsed


class CAREWSMAudioDataModule(DataModule):
    """Construct canonical fold populations and native feature datasets."""

    def __init__(
        self,
        care_root: str,
        wsm_root: str,
        manifest: str,
        registry: str,
        feature_family: str,
        fold: int = 0,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = False,
    ) -> None:
        if not feature_family:
            raise ValueError("CAREWSMAudioDataModule requires feature_family")

        self.fold_index = fold
        self.feature_family = feature_family
        self.roots = {"CARE": Path(care_root), "WSM": Path(wsm_root)}
        self.feature_specs = self._read_registry(Path(registry))
        unknown = {feature_family} - set(self.feature_specs)
        if unknown:
            raise ValueError(f"unknown feature family: {sorted(unknown)}")
        self.records = self._read_and_join_manifest(Path(manifest))
        self.views = self._build_views(feature_family)

        datasets = {
            name: CAREWSMAudioDataset(
                records,
                self.feature_family,
                self.feature_specs,
            )
            for name, records in self.views.items()
        }
        validation_datasets = {
            "val_care": datasets["val_care"],
            "val_wsm": datasets["val_wsm"],
        }
        validation_datasets.update(
            {
                "test_care": datasets["test_care"],
                "test_wsm": datasets["test_wsm"],
            }
        )
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset=validation_datasets,
            test_dataset=None,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle_train=shuffle_train,
        )

    @staticmethod
    def _read_registry(path: Path) -> dict[str, dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            family: specification
            for family, specification in data["sources"].items()
            if specification.get("modality") == "audio"
        }

    def _source_index(self, corpus: str) -> dict[str, dict[str, str]]:
        index: dict[str, dict[str, str]] = {}
        for split_name in ("dev", "train", "test"):
            split_path = self.roots[corpus] / "splits" / f"{split_name}.csv"
            for row in read_rows(split_path):
                index.setdefault(row["sample_id"], row)
                if split_name == "dev":
                    index[row["sample_id"]] = row
        return index

    def _read_and_join_manifest(
        self, path: Path
    ) -> dict[tuple[str, str], dict[str, Any]]:
        manifest: dict[tuple[str, str], dict[str, Any]] = {}
        source_indexes = {corpus: self._source_index(corpus) for corpus in self.roots}
        for row in read_rows(path):
            key = (row["corpus"], row["sample_id"])
            if key in manifest:
                raise ValueError(f"duplicate manifest key: {key}")
            source = source_indexes[key[0]].get(key[1])
            if source is None:
                raise ValueError(f"manifest sample absent from source splits: {key}")

            record = dict(row)
            record.update(self._source_metadata(key[0], source))
            for target in ("depression", "parkinson"):
                value = parse_label(row.get(target))
                observed = value is not None
                declared = (
                    str(row.get(f"{target}_observed", observed)).lower() == "true"
                )
                if declared != observed:
                    raise ValueError(f"observed-label mismatch for {key}: {target}")
                record[target] = value
                record[f"{target}_observed"] = observed
            record["feature_paths"] = self._feature_paths(key[0], source)
            manifest[key] = record
        return manifest

    def _source_metadata(self, corpus: str, source: dict[str, str]) -> dict[str, str]:
        profile_id = source.get("profile_id", "")
        video_id = source.get("video_id", "")
        if not video_id:
            video_id = Path(source.get("video_dir", "")).name
            video_id = video_id.removeprefix("VIDEO_")
        return {
            "profile_id": profile_id,
            "video_id": video_id,
            "label_id": source.get("label_id", source.get("stratum", "")),
            "subgroup": source.get("subgroup", ""),
            "video_dir": source.get("video_dir", ""),
            "source_corpus": corpus,
        }

    def _feature_paths(self, corpus: str, source: dict[str, str]) -> dict[str, str]:
        root = self.roots[corpus]
        profile = source.get("profile_id", "")
        video = source.get("video_id", "") or Path(source.get("video_dir", "")).name
        profile_dir = (
            profile if profile.startswith("PROFILE_") else f"PROFILE_{profile}"
        )
        video_dir = video if video.startswith("VIDEO_") else f"VIDEO_{video}"
        extracted = (
            root / "extracted" / source.get("label_id", source.get("stratum", ""))
        )
        extracted /= Path(source.get("subgroup", "")) / profile_dir / video_dir

        fallback = Path(source.get("video_dir", ""))
        parts = fallback.parts
        if parts and parts[0] in (root.name, "CARE", "WSM_NEW"):
            fallback = Path(*parts[1:])
        fallback_directory = root / fallback

        paths: dict[str, str] = {}
        for family, specification in self.feature_specs.items():
            directory = extracted if extracted.exists() else fallback_directory
            paths[family] = str(directory / specification["file"])
        return paths

    def _fold_records(self, corpus: str, part: str) -> list[dict[str, Any]]:
        path = (
            self.roots[corpus]
            / "splits"
            / "folds"
            / f"fold_{self.fold_index}_{part}.csv"
        )
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in read_rows(path):
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"duplicate fold sample: {corpus}/{sample_id}")
            seen.add(sample_id)
            key = (corpus, sample_id)
            if key not in self.records:
                raise ValueError(f"fold sample absent from manifest: {key}")
            record = dict(self.records[key])
            if row.get("profile_id") and row["profile_id"] != record["profile_id"]:
                raise ValueError(f"profile mismatch for {key}")
            records.append(record)
        return records

    def _build_views(
        self, selected: str
    ) -> dict[str, list[dict[str, Any]]]:
        unknown = {selected} - set(self.feature_specs)
        if unknown:
            raise ValueError(f"unknown feature family: {sorted(unknown)}")

        train = self._fold_records("CARE", "train") + self._fold_records("WSM", "train")
        validation = self._fold_records("CARE", "val") + self._fold_records(
            "WSM", "val"
        )
        tests = {
            corpus: [
                self.records[(corpus, row["sample_id"])]
                for row in sorted(
                    read_rows(self.roots[corpus] / "splits" / "test.csv"),
                    key=lambda item: item["sample_id"],
                )
            ]
            for corpus in self.roots
        }

        views = {
            "train": train,
            "val_care": [record for record in validation if record["corpus"] == "CARE"],
            "val_wsm": [record for record in validation if record["corpus"] == "WSM"],
            "test_care": tests["CARE"],
            "test_wsm": tests["WSM"],
        }
        return {
            name: [
                record
                for record in sorted(
                    records, key=lambda item: (item["corpus"], item["sample_id"])
                )
                if record.get(f"{selected}_status") == "VALID"
            ]
            for name, records in views.items()
        }

    def describe_context(self, context: Any) -> None:
        """Expose fold and population counts to Chimera context logging."""
        context.set("data.module", self)
        context.set("data.fold", self.fold_index)
        context.set("data.feature_family", self.feature_family)
        for name, records in self.views.items():
            context.set(f"data.{name}_count", len(records))


@DATAMODULES.register("carewsm_joint")
def carewsm_joint(
    context: Any | None = None,
    **params: Any,
) -> CAREWSMAudioDataModule:
    """Build the compatibility-registered CAREWSM DataModule."""
    del context
    return CAREWSMAudioDataModule(**params)
