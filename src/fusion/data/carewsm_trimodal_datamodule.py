"""Canonical fold DataModule for audited CAREWSM fusion artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule

from fusion.data.carewsm_trimodal_dataset import (
    CAREWSMTrimodalDataset,
    trimodal_collate,
)

FOLDS = range(5)
FAMILIES = {
    "audio": ("audio_id1", "audio_id2", "audio_id4"),
    "video": ("video_4",),
    "text": ("text_id9", "text_id10"),
}
ALL_FAMILIES = tuple(source for values in FAMILIES.values() for source in values)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_binary(value: str | None, field: str, key: str) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if parsed not in (0.0, 1.0):
        raise ValueError(f"{field} must be 0, 1, or empty for {key}: {value!r}")
    return parsed


class CAREWSMTrimodalDataModule(DataModule):
    """Build common-cohort joint TRAIN and corpus-specific VAL/TEST views."""

    def __init__(
        self,
        care_root: str,
        wsm_root: str,
        manifest: str,
        fold: int = 0,
        modalities: list[str] | tuple[str, ...] = ("audio", "video", "text"),
        audio_source: str = "audio_id1",
        video_source: str = "video_4",
        text_source: str = "text_id9",
        representation_key: str = "shared_features",
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = False,
        common_cohort: bool = False,
    ) -> None:
        if fold not in FOLDS:
            raise ValueError("fold must be one of 0, 1, 2, 3, 4")
        self.fold = fold
        self.common_cohort = bool(common_cohort)
        self.modalities = tuple(modalities)
        if len(self.modalities) not in (2, 3) or len(set(self.modalities)) != len(
            self.modalities
        ):
            raise ValueError(
                "modalities must contain exactly two or three distinct names"
            )
        if any(modality not in FAMILIES for modality in self.modalities):
            raise ValueError(f"unsupported modality in {self.modalities}")
        selected = {"audio": audio_source, "video": video_source, "text": text_source}
        self.sources = {modality: selected[modality] for modality in self.modalities}
        for modality, source in self.sources.items():
            if source not in FAMILIES[modality]:
                raise ValueError(f"unsupported {modality} source: {source}")
        self.representation_key = representation_key
        self.roots = {"CARE": Path(care_root), "WSM": Path(wsm_root)}
        self.manifest = Path(manifest)
        self.target_manifest = self._read_target_manifest()
        self.records = self._join_sources()
        self.views = self._build_views()
        datasets = {
            name: CAREWSMTrimodalDataset(view, self.sources, representation_key)
            for name, view in self.views.items()
        }
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset={
                "val_care": datasets["val_care"],
                "val_wsm": datasets["val_wsm"],
            },
            test_dataset={
                "test_care": datasets["test_care"],
                "test_wsm": datasets["test_wsm"],
            },
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle_train=shuffle_train,
            collate_fn=trimodal_collate,
        )

    def _source_index(self, corpus: str) -> dict[str, dict[str, str]]:
        if corpus == "WSM":
            rows = read_rows(self.roots[corpus] / "splits" / "videos.csv")
        else:
            rows = [
                row
                for split in ("train", "dev", "test")
                for row in read_rows(self.roots[corpus] / "splits" / f"{split}.csv")
            ]
        index: dict[str, dict[str, str]] = {}
        identity_fields = ("profile_id", "video_id", "label_id", "subgroup")
        for row in rows:
            sample_id = row.get("sample_id", "")
            if not sample_id:
                raise ValueError(f"{corpus} source row has no sample_id")
            previous = index.get(sample_id)
            if previous is not None and tuple(
                previous.get(field, "") for field in identity_fields
            ) != tuple(row.get(field, "") for field in identity_fields):
                raise ValueError(f"conflicting metadata for {corpus}/{sample_id}")
            index[sample_id] = row
        pairs: dict[tuple[str, str], str] = {}
        for sample_id, row in index.items():
            pair = (row.get("profile_id", ""), row.get("video_id", ""))
            if pair in pairs and pairs[pair] != sample_id:
                raise ValueError(f"profile/video collision: {corpus}/{pair}")
            pairs[pair] = sample_id
        return index

    def _read_target_manifest(self) -> dict[tuple[str, str], dict[str, str]]:
        manifest = {}
        for row in read_rows(self.manifest):
            key = (row["corpus"], row["sample_id"])
            if key in manifest:
                raise ValueError(f"duplicate target manifest key: {key}")
            manifest[key] = row
        return manifest

    def _join_sources(self) -> dict[tuple[str, str], dict[str, Any]]:
        joined = {}
        for corpus, root in self.roots.items():
            for sample_id, source in self._source_index(corpus).items():
                manifest = self.target_manifest.get((corpus, sample_id))
                if manifest is None:
                    raise ValueError(
                        f"target manifest sample absent: {(corpus, sample_id)}"
                    )
                for field in ("profile_id", "video_id"):
                    if manifest.get(field, "") != source.get(field, ""):
                        raise ValueError(
                            f"target manifest metadata mismatch for {(corpus, sample_id)}: {field}"
                        )
                if corpus == "CARE":
                    targets = {
                        target: parse_binary(manifest.get(target), target, sample_id)
                        for target in ("depression", "parkinson")
                    }
                    for target, value in targets.items():
                        observed = (
                            str(manifest.get(f"{target}_observed", "")).lower()
                            == "true"
                        )
                        if observed != (value is not None):
                            raise ValueError(
                                f"target manifest observed mismatch for {(corpus, sample_id)}: {target}"
                            )
                    provenance = "CARE feature-audit target manifest"
                else:
                    targets = {
                        target: parse_binary(
                            source.get(f"y_{target}"), f"y_{target}", sample_id
                        )
                        for target in ("depression", "parkinson")
                    }
                    provenance = "WSM videos.csv y_* columns"
                joined[(corpus, sample_id)] = {
                    "corpus": corpus,
                    "sample_id": sample_id,
                    "profile_id": source.get("profile_id", ""),
                    "video_id": source.get("video_id", ""),
                    "label_id": source.get("label_id", ""),
                    "subgroup": source.get("subgroup", ""),
                    "targets": targets,
                    "target_provenance": provenance,
                    "root": root,
                }
        return joined

    def _fold_rows(self, corpus: str, split: str) -> list[dict[str, str]]:
        return read_rows(
            self.roots[corpus] / "splits" / "folds" / f"fold_{self.fold}_{split}.csv"
        )

    def _artifact_path(self, record: dict[str, Any], source: str) -> Path:
        return (
            record["root"]
            / "extracted_unimodal"
            / record["label_id"]
            / record["subgroup"]
            / f"PROFILE_{record['profile_id'].removeprefix('PROFILE_')}"
            / f"VIDEO_{record['video_id'].removeprefix('VIDEO_')}"
            / f"{source}_fold{self.fold}.pickle"
        )

    def _records_for_rows(
        self, corpus: str, rows: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        output, seen = [], set()
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"duplicate fold sample: {corpus}/{sample_id}")
            seen.add(sample_id)
            key = (corpus, sample_id)
            if key not in self.records:
                raise ValueError(f"fold sample absent from source index: {key}")
            record = dict(self.records[key])
            for field in ("profile_id", "video_id", "label_id", "subgroup"):
                if row.get(field) and row[field] != record[field]:
                    raise ValueError(f"fold metadata mismatch for {key}: {field}")
            required = ALL_FAMILIES if self.common_cohort else tuple(self.sources.values())
            all_paths = {
                source: self._artifact_path(record, source) for source in required
            }
            if not all(path.is_file() for path in all_paths.values()):
                continue
            record["artifact_paths"] = {
                modality: str(all_paths[self.sources[modality]])
                for modality in self.modalities
            }
            record["common_cohort"] = self.common_cohort
            output.append(record)
        return output

    def _build_views(self) -> dict[str, list[dict[str, Any]]]:
        train = self._records_for_rows(
            "CARE", self._fold_rows("CARE", "train")
        ) + self._records_for_rows("WSM", self._fold_rows("WSM", "train"))
        val_care = self._records_for_rows("CARE", self._fold_rows("CARE", "val"))
        val_wsm = self._records_for_rows("WSM", self._fold_rows("WSM", "val"))
        test_care = self._records_for_rows(
            "CARE", read_rows(self.roots["CARE"] / "splits" / "test.csv")
        )
        test_wsm = self._records_for_rows(
            "WSM", read_rows(self.roots["WSM"] / "splits" / "test.csv")
        )
        self._check_profile_isolation(train, val_care + val_wsm, test_care + test_wsm)
        return {
            "train": train,
            "val_care": val_care,
            "val_wsm": val_wsm,
            "test_care": test_care,
            "test_wsm": test_wsm,
        }

    @staticmethod
    def _check_profile_isolation(*populations: list[dict[str, Any]]) -> None:
        profiles = [
            {(row["corpus"], row["profile_id"]) for row in population}
            for population in populations
        ]
        if (
            profiles[0] & profiles[1]
            or profiles[0] & profiles[2]
            or profiles[1] & profiles[2]
        ):
            raise ValueError("profile overlap across canonical split views")

    def describe_context(self, context: Any) -> None:
        context.set("data.module", self)
        context.set("data.fold", self.fold)
        context.set("data.modalities", self.modalities)
        context.set(
            "data.provenance_status", "UNKNOWN / BLOCKED: upstream FIT/SELECTION"
        )
        for name, records in self.views.items():
            context.set(f"data.{name}_count", len(records))


@DATAMODULES.register("carewsm_trimodal")
def carewsm_trimodal(
    context: Any | None = None, **params: Any
) -> CAREWSMTrimodalDataModule:
    del context
    return CAREWSMTrimodalDataModule(**params)


@DATAMODULES.register("carewsm_fusion")
def carewsm_fusion(
    context: Any | None = None, **params: Any
) -> CAREWSMTrimodalDataModule:
    del context
    params.setdefault("common_cohort", True)
    return CAREWSMTrimodalDataModule(**params)
