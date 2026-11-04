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
        fold: int = 0,
        feature_family: str | None = None,
        feature_families: list[str] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = False,
        max_sequence_length: int | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
    ) -> None:
        if fold not in range(5):
            raise ValueError("fold must be an integer from 0 through 4")
        if feature_families is not None and feature_family is not None:
            raise ValueError(
                "feature_family and feature_families cannot both be configured"
            )

        if feature_families is not None:
            selected = tuple(feature_families)
        else:
            selected = (feature_family or "egemaps_lld",)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("feature families must be non-empty and non-duplicate")

        self.fold_index = fold
        self.feature_families = selected
        self.feature_family = selected[0] if len(selected) == 1 else None
        self.roots = {"CARE": Path(care_root), "WSM": Path(wsm_root)}
        self.feature_specs = self._read_registry(Path(registry))
        unknown = set(selected) - set(self.feature_specs)
        if unknown:
            raise ValueError(f"unknown feature family: {sorted(unknown)}")
        self.records = self._read_and_join_manifest(Path(manifest))
        self.views = self._build_views(selected)

        datasets = {
            name: CAREWSMAudioDataset(
                self._limited_records(
                    name,
                    records,
                    max_train_samples,
                    max_val_samples,
                    max_test_samples,
                ),
                selected,
                self.feature_specs,
                max_sequence_length,
            )
            for name, records in self.views.items()
        }
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset={"joint": datasets["val"]},
            test_dataset={"CARE": datasets["care_test"], "WSM": datasets["wsm_test"]},
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
        self, selected: tuple[str, ...]
    ) -> dict[str, list[dict[str, Any]]]:
        unknown = set(selected) - set(self.feature_specs)
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

        self._assert_isolated(train, validation, "TRAIN", "VAL")
        self._assert_isolated(
            train + validation, tests["CARE"] + tests["WSM"], "TRAIN/VAL", "TEST"
        )
        views = {
            "train": train,
            "val": validation,
            "care_val": [record for record in validation if record["corpus"] == "CARE"],
            "wsm_val": [record for record in validation if record["corpus"] == "WSM"],
            "care_test": tests["CARE"],
            "wsm_test": tests["WSM"],
        }
        return {
            name: [
                record
                for record in sorted(
                    records, key=lambda item: (item["corpus"], item["sample_id"])
                )
                if all(record.get(f"{family}_status") == "VALID" for family in selected)
            ]
            for name, records in views.items()
        }

    @staticmethod
    def _assert_isolated(
        left: list[dict[str, Any]],
        right: list[dict[str, Any]],
        left_name: str,
        right_name: str,
    ) -> None:
        left_keys = {(row["corpus"], row["sample_id"]) for row in left}
        right_keys = {(row["corpus"], row["sample_id"]) for row in right}
        left_profiles = {(row["corpus"], row["profile_id"]) for row in left}
        right_profiles = {(row["corpus"], row["profile_id"]) for row in right}
        if left_keys & right_keys:
            raise ValueError(f"{left_name}/{right_name} sample overlap")
        if left_profiles & right_profiles:
            raise ValueError(f"{left_name}/{right_name} profile overlap")

    @staticmethod
    def _limited_records(
        name: str,
        records: list[dict[str, Any]],
        max_train_samples: int | None,
        max_val_samples: int | None,
        max_test_samples: int | None,
    ) -> list[dict[str, Any]]:
        limit = max_test_samples if name in {"care_test", "wsm_test"} else None
        if name == "train":
            limit = max_train_samples
        elif name in {"val", "care_val", "wsm_val"}:
            limit = max_val_samples
        if limit is None:
            return records
        if name in {"train", "val"}:
            return [
                record
                for corpus in ("CARE", "WSM")
                for record in [row for row in records if row["corpus"] == corpus][
                    :limit
                ]
            ]
        return records[:limit]

    def describe_context(self, context: Any) -> None:
        """Expose fold and population counts to Chimera context logging."""
        context.set("data.module", self)
        context.set("data.fold", self.fold_index)
        context.set(
            "data.feature_family", self.feature_family or list(self.feature_families)
        )
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
