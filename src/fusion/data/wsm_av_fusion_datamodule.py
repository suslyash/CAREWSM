"""Canonical sparse audio+video fusion DataModule."""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule
from common.data.wsm_manifest import build_manifest
from common.utils.segment_index import build_wsm_multitask_segment_index
from video.features.clip_video_features import PREPROCESSING_VERSION, ROI_POLICY_VERSION
from video.data.wsm_video_cache_datamodule import FEATURE_DIM, MODEL_NAME, MODEL_REVISION, TARGET_FRAMES, YOLO_SHA256

TASK_NAMES = ("depression", "parkinson")
PROTOCOLS = ("test_none", "test_soft", "test_hard")
AUDIO_EXTRACTOR = "transformers_ssl"
AUDIO_MODEL = "microsoft/wavlm-base-plus"
AUDIO_LAYER = 9
AUDIO_POOL = 4
EXPECTED_COUNTS = {"train": 6325, "dev": 933, "test_none": 1364, "test_soft": 1208, "test_hard": 1014}


class WSMAVFusionDataError(ValueError):
    """Raised when the canonical A+V join violates its frozen contracts."""


def _filter_flag(value: Any, field: str) -> bool:
    if value is None or (isinstance(value, str) and (not value.strip() or value.strip().casefold() == "nan")):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WSMAVFusionDataError(f"{field} must be missing, 0, or 1: {value!r}") from exc
    if math.isnan(number):
        return False
    if number not in (0.0, 1.0):
        raise WSMAVFusionDataError(f"{field} must be missing, 0, or 1: {value!r}")
    return number == 1.0


def _load_video_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise WSMAVFusionDataError(f"video cache index is missing: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WSMAVFusionDataError(f"invalid video cache index JSON at line {line_number}") from exc
        segment_id = record.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            raise WSMAVFusionDataError(f"video cache index line {line_number} has no segment_id")
        if segment_id in records:
            raise WSMAVFusionDataError(f"duplicate video cache index segment_id: {segment_id}")
        records[segment_id] = record
    return records


def _audio_path(root: Path, row: dict[str, Any]) -> Path:
    return (root / "audio" / AUDIO_EXTRACTOR / AUDIO_MODEL.replace("/", "__")
            / f"layer{AUDIO_LAYER}_pool{AUDIO_POOL}" / row["split"] / row["video_id"]
            / f"{Path(row['segment_file']).stem}.pkl")


def _validate_audio(path: Path, row: dict[str, Any], feature_dim: int | None) -> int:
    if not path.is_file():
        raise WSMAVFusionDataError(f"missing audio feature cache for {row['segment_id']}: {path}")
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception as exc:
        raise WSMAVFusionDataError(f"cannot read audio feature cache: {path}") from exc
    if not isinstance(payload, dict) or not {"audio_temporal", "audio_cls", "meta"}.issubset(payload):
        raise WSMAVFusionDataError(f"audio payload keys are invalid: {path}")
    temporal, pooled, meta = payload["audio_temporal"], payload["audio_cls"], payload["meta"]
    if not isinstance(temporal, torch.Tensor) or temporal.ndim != 2 or temporal.shape[0] < 1:
        raise WSMAVFusionDataError(f"audio_temporal must be [T,D] with T>=1: {path}")
    if not isinstance(pooled, torch.Tensor) or pooled.ndim != 1 or pooled.shape[0] != temporal.shape[1]:
        raise WSMAVFusionDataError(f"audio_cls dimension does not match audio_temporal: {path}")
    if not torch.is_floating_point(temporal) or not torch.is_floating_point(pooled) or not torch.isfinite(temporal).all() or not torch.isfinite(pooled).all():
        raise WSMAVFusionDataError(f"audio payload is not finite floating point: {path}")
    if not isinstance(meta, dict) or str(meta.get("video_id")) != row["video_id"] or str(meta.get("segment_file")) != row["segment_file"]:
        raise WSMAVFusionDataError(f"audio payload identity mismatch: {path}")
    if feature_dim is not None and int(temporal.shape[1]) != feature_dim:
        raise WSMAVFusionDataError(f"audio feature dimension changed at {path}")
    return int(temporal.shape[1])


def _validate_video(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if record.get("split") != row["split"] or record.get("status") not in {"extracted", "reused"}:
        raise WSMAVFusionDataError(f"video cache record is invalid for {row['segment_id']}")
    path = Path(str(record.get("cache_path", "")))
    if not path.is_file():
        raise WSMAVFusionDataError(f"missing video cache artifact for {row['segment_id']}: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("segment_id") != row["segment_id"] or artifact.get("cache_fingerprint") != record.get("cache_fingerprint"):
        raise WSMAVFusionDataError(f"video artifact identity mismatch: {path}")
    if artifact.get("model_name") != MODEL_NAME or artifact.get("model_revision") != MODEL_REVISION or artifact.get("detector_weights_sha256") != YOLO_SHA256:
        raise WSMAVFusionDataError(f"video preprocessing identity mismatch: {path}")
    preprocessing = artifact.get("preprocessing")
    if not isinstance(preprocessing, dict) or preprocessing.get("target_frames") != TARGET_FRAMES or preprocessing.get("version") != PREPROCESSING_VERSION or preprocessing.get("roi_policy_version") != ROI_POLICY_VERSION:
        raise WSMAVFusionDataError(f"video preprocessing metadata mismatch: {path}")
    features, valid_mask = artifact.get("features"), artifact.get("valid_mask")
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != FEATURE_DIM or not 1 <= features.shape[0] <= TARGET_FRAMES:
        raise WSMAVFusionDataError(f"video features must be [T,{FEATURE_DIM}], 1<=T<={TARGET_FRAMES}: {path}")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != (features.shape[0],) or not bool(valid_mask.all()) or not torch.isfinite(features).all():
        raise WSMAVFusionDataError(f"video valid_mask/features are invalid: {path}")
    return {"path": path, "fingerprint": str(record["cache_fingerprint"]), "length": int(features.shape[0])}


def _targets(row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    values = [row["y_depression"], row["y_parkinson"]]
    observed = torch.tensor([bool(row["observed_depression"]), bool(row["observed_parkinson"])], dtype=torch.bool)
    targets = torch.tensor([float(value) if value is not None else float("nan") for value in values], dtype=torch.float32)
    if any(bool(observed[i]) != (values[i] is not None) for i in range(2)):
        raise WSMAVFusionDataError(f"target/observed mismatch for {row['segment_id']}")
    return targets, observed


class WSMAVFusionDataset(Dataset):
    def __init__(self, samples: Iterable[dict[str, Any]]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        with sample["audio_path"].open("rb") as handle:
            payload = pickle.load(handle)
        artifact = torch.load(sample["video_path"], map_location="cpu", weights_only=False)
        audio = payload["audio_temporal"].float()
        audio_cls = payload["audio_cls"].float()
        video = artifact["features"].float()
        length_a, length_v = audio.shape[0], video.shape[0]
        return {
            "inputs": {"audio": audio, "audio_cls": audio_cls, "video": video},
            "targets": sample["targets"].clone(), "observed_mask": sample["observed_mask"].clone(),
            "audio_mask": torch.ones(length_a, dtype=torch.bool), "video_mask": torch.ones(length_v, dtype=torch.bool),
            "meta": {**sample["meta"], "audio_temporal_length": length_a, "video_temporal_length": length_v},
        }


def collate_wsm_av_fusion(samples: list[dict[str, Any]]) -> Batch:
    if not samples:
        raise WSMAVFusionDataError("cannot collate an empty fusion batch")
    batch_size = len(samples)
    audio_dim = int(samples[0]["inputs"]["audio"].shape[1])
    max_audio = max(int(s["inputs"]["audio"].shape[0]) for s in samples)
    max_video = max(int(s["inputs"]["video"].shape[0]) for s in samples)
    audio = torch.zeros(batch_size, max_audio, audio_dim, dtype=torch.float32)
    video = torch.zeros(batch_size, max_video, FEATURE_DIM, dtype=torch.float32)
    audio_mask = torch.zeros(batch_size, max_audio, dtype=torch.bool)
    video_mask = torch.zeros(batch_size, max_video, dtype=torch.bool)
    for i, sample in enumerate(samples):
        a, v = sample["inputs"]["audio"].float(), sample["inputs"]["video"].float()
        audio[i, :a.shape[0]] = a; audio_mask[i, :a.shape[0]] = sample["audio_mask"]
        video[i, :v.shape[0]] = v; video_mask[i, :v.shape[0]] = sample["video_mask"]
    return Batch(
        inputs={"audio": audio, "audio_cls": torch.stack([s["inputs"]["audio_cls"].float() for s in samples]), "video": video},
        targets=torch.stack([s["targets"].float() for s in samples]),
        masks={"audio_mask": audio_mask, "video_mask": video_mask, "observed_mask": torch.stack([s["observed_mask"] for s in samples]), "modality_available": torch.ones(batch_size, 2, dtype=torch.bool)},
        meta={"sample_meta": [s["meta"] for s in samples]},
    )


class WSMAVFusionDataModule(DataModule):
    def __init__(self, data_root: str, audio_feature_cache_root: str, video_cache_root: str, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = True, persistent_workers: bool = False, shuffle_train: bool = True, drop_last_train: bool = False) -> None:
        self.data_root = str(Path(data_root).expanduser().resolve())
        self.audio_feature_cache_root = Path(audio_feature_cache_root).expanduser().resolve()
        self.video_cache_root = Path(video_cache_root).expanduser().resolve()
        rows, manifest_audit = build_manifest(self.data_root)
        self.manifest_audit = manifest_audit
        raw = build_wsm_multitask_segment_index(self.data_root)
        raw_lookup = {(str(r.task), str(r.video_id), str(r.segment_file)): r for r in raw.itertuples(index=False)}
        if len(raw_lookup) != len(raw):
            raise WSMAVFusionDataError("duplicate raw canonical identity")
        video_index = _load_video_index(self.video_cache_root / "cache_index.jsonl")
        samples_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], **{p: [] for p in PROTOCOLS}}
        audio_dim: int | None = None
        seen: set[str] = set()
        missing_audio = missing_video = 0
        for original_row in rows:
            corpus, video_id, segment_file = json.loads(original_row["segment_id"])
            row = {**original_row, "corpus": corpus, "video_id": video_id, "segment_file": segment_file}
            if row["segment_id"] in seen:
                raise WSMAVFusionDataError(f"duplicate canonical join: {row['segment_id']}")
            seen.add(row["segment_id"])
            audio_path = _audio_path(self.audio_feature_cache_root, row)
            try:
                detected = _validate_audio(audio_path, row, audio_dim)
            except WSMAVFusionDataError:
                missing_audio += 1; raise
            audio_dim = detected if audio_dim is None else audio_dim
            record = video_index.get(row["segment_id"])
            if record is None:
                missing_video += 1; raise WSMAVFusionDataError(f"missing video cache index record: {row['segment_id']}")
            video_meta = _validate_video(record, row)
            targets, observed = _targets(row)
            meta = {"segment_id": row["segment_id"], "video_id": video_id, "corpus": corpus, "split": row["split"], "segment_file": segment_file, "audio_cache_path": str(audio_path), "video_cache_path": str(video_meta["path"]), "video_cache_fingerprint": video_meta["fingerprint"]}
            sample = {"audio_path": audio_path, "video_path": video_meta["path"], "targets": targets, "observed_mask": observed, "meta": meta}
            if row["split"] in {"train", "dev"}: samples_by_split[row["split"]].append(sample)
            else:
                samples_by_split["test_none"].append(sample)
                raw_row = raw_lookup.get((corpus, video_id, segment_file))
                if raw_row is None: raise WSMAVFusionDataError(f"missing raw Test membership: {row['segment_id']}")
                if _filter_flag(raw_row.soft_filter, "soft_filter"): samples_by_split["test_soft"].append({**sample, "meta": {**meta, "evaluation_protocol": "test_soft"}})
                if _filter_flag(raw_row.hard_filter, "hard_filter"): samples_by_split["test_hard"].append({**sample, "meta": {**meta, "evaluation_protocol": "test_hard"}})
                samples_by_split["test_none"][-1]["meta"]["evaluation_protocol"] = "test_none"
        self.audio_feature_dim = int(audio_dim or 0)
        self.video_feature_dim = FEATURE_DIM
        self.train_dataset = WSMAVFusionDataset(samples_by_split["train"])
        self.val_dataset = WSMAVFusionDataset(samples_by_split["dev"])
        self.test_dataset = {p: WSMAVFusionDataset(samples_by_split[p]) for p in PROTOCOLS}
        self.collate_fn = collate_wsm_av_fusion
        self.audit = {"canonical_total": len(rows), "joined_total": len(seen), "unique_join": len(seen), "missing_audio": missing_audio, "missing_video": missing_video, "duplicate_join": 0, "audio_cache_files_found": len(rows), "video_cache_records_found": len(rows), "audio_feature_dim": self.audio_feature_dim, "video_feature_dim": FEATURE_DIM, "no_dropped_rows": True, "counts": {"train": len(self.train_dataset), "dev": len(self.val_dataset), **{p: len(self.test_dataset[p]) for p in PROTOCOLS}}}
        if self.audit["canonical_total"] != 8622 or self.audit["counts"] != EXPECTED_COUNTS:
            raise WSMAVFusionDataError(f"unexpected canonical/join counts: {self.audit}")
        super().__init__(train_dataset=self.train_dataset, val_dataset=self.val_dataset, test_dataset=self.test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, shuffle_train=shuffle_train, drop_last_train=drop_last_train, collate_fn=self.collate_fn)

    def val_dataloader(self) -> dict[str, Any]:
        return {p: self._make_loader(self.val_dataset if p == "dev" else self.test_dataset[p], shuffle=False, drop_last=False) for p in ("dev", *PROTOCOLS)}

    def describe_context(self, context: Any) -> None:
        context.set("data.num_tasks", 2); context.set("data.task_names", list(TASK_NAMES)); context.set("data.modality_names", ["audio", "video"])
        context.set("data.audio_feature_dim", self.audio_feature_dim); context.set("data.video_feature_dim", FEATURE_DIM); context.set("data.audio_video_fusion", True)
        context.set("data.train_rows", len(self.train_dataset)); context.set("data.dev_rows", len(self.val_dataset))
        for p in PROTOCOLS: context.set(f"data.{p}_rows", len(self.test_dataset[p]))
        context.set("data.audio_unavailable", 0); context.set("data.video_unavailable", 0); context.set("data.test_protocols", list(PROTOCOLS))


@DATAMODULES.register("wsm_av_fusion_datamodule")
def wsm_av_fusion_datamodule(context: Any | None = None, **params: Any) -> WSMAVFusionDataModule:
    del context
    return WSMAVFusionDataModule(**params)
