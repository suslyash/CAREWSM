"""Registered A+V DataModule for the frozen RAMPS R2 TRAIN cache."""
from __future__ import annotations

import hashlib
from functools import partial
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import DATAMODULES
from common.data.wsm_manifest import build_manifest
from fusion.data.wsm_av_fusion_datamodule import (
    EXPECTED_COUNTS,
    PROTOCOLS,
    TASK_NAMES,
    WSMAVFusionDataError,
    WSMAVFusionDataModule,
    collate_wsm_av_fusion,
)

CACHE_VERSION = "ramps-r2-semantic-v1"
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "main"
AUDIO_SHA256 = "0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2"
VIDEO_SHA256 = "3e39778126db401a2fe17bb472a172191616e8a7b5265e982233c53dcd4af2f6"
PROMPT_BANK_SHA256 = "19428db58f91f73ca26ce9c4354b5731431e14ec524fc1a47e7070e1b32f447e"
EXPECTED_ACCEPTED = (376, 1801)
EXPECTED_MISSING = (2665, 3660)
EXPECTED_POSITIVE = (376, 212)
EXPECTED_NEGATIVE = (0, 1589)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_tensor(cache: dict[str, Any], name: str, rows: int) -> torch.Tensor:
    value = cache.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (rows, 2):
        raise WSMAVFusionDataError(f"pseudo cache field {name!r} must have shape [{rows}, 2]")
    return value.detach().cpu()


class _PseudoDataset(Dataset):
    def __init__(self, base: Dataset, fields: dict[str, torch.Tensor], train: bool) -> None:
        self.base = base
        self.fields = fields
        self.train = train

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        if self.train:
            for name, values in self.fields.items():
                sample[name] = values[index].clone()
        else:
            sample["pseudo_accept_mask"] = torch.zeros(2, dtype=torch.bool)
            sample["pseudo_targets"] = torch.full((2,), float("nan"), dtype=torch.float32)
            sample["pseudo_reliability"] = torch.zeros(2, dtype=torch.float32)
            sample["pseudo_class"] = torch.full((2,), -1, dtype=torch.long)
        return sample


def collate_wsm_ramps_semantic(samples: list[dict[str, Any]], modality_available_override: tuple[bool, bool] | None = None) -> Batch:
    batch = collate_wsm_av_fusion(samples)
    if modality_available_override is not None:
        batch.masks["modality_available"] = torch.tensor(modality_available_override, dtype=torch.bool).expand(len(samples), -1).clone()
    batch.masks["pseudo_accept_mask"] = torch.stack([s["pseudo_accept_mask"].bool() for s in samples])
    batch.inputs["pseudo_targets"] = torch.stack([s["pseudo_targets"].float() for s in samples])
    batch.inputs["pseudo_reliability"] = torch.stack([s["pseudo_reliability"].float() for s in samples])
    batch.inputs["pseudo_class"] = torch.stack([s["pseudo_class"].long() for s in samples])
    return batch


class WSMRampsSemanticDataModule(WSMAVFusionDataModule):
    """Base A+V datasets with immutable accepted RAMPS pseudo fields on TRAIN."""

    def __init__(self, pseudo_cache_path: str,
                 modality_available_override: Any = None,
                 expected_accepted_counts: Any = EXPECTED_ACCEPTED,
                 expected_positive_counts: Any = EXPECTED_POSITIVE,
                 expected_negative_counts: Any = EXPECTED_NEGATIVE,
                 **kwargs: Any) -> None:
        if modality_available_override is not None:
            if (not isinstance(modality_available_override, (list, tuple)) or len(modality_available_override) != 2
                    or any(not isinstance(value, bool) for value in modality_available_override)
                    or not any(modality_available_override)):
                raise WSMAVFusionDataError("modality_available_override must be exactly two booleans with at least one true value")
            modality_available_override = (modality_available_override[0], modality_available_override[1])
        super().__init__(**kwargs)
        self.modality_available_override = modality_available_override
        def contract(value: Any, name: str) -> tuple[int, int]:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise WSMAVFusionDataError(f"{name} must contain exactly two non-negative integers")
            if any(isinstance(x, bool) or not isinstance(x, Integral) or int(x) < 0 for x in value):
                raise WSMAVFusionDataError(f"{name} must contain exactly two non-negative integers")
            return (int(value[0]), int(value[1]))
        expected_accepted = contract(expected_accepted_counts, "expected_accepted_counts")
        expected_positive = contract(expected_positive_counts, "expected_positive_counts")
        expected_negative = contract(expected_negative_counts, "expected_negative_counts")
        cache_path = Path(pseudo_cache_path).expanduser().resolve()
        if not cache_path.is_file():
            raise WSMAVFusionDataError(f"pseudo cache is missing: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict):
            raise WSMAVFusionDataError("pseudo cache must contain a mapping")
        rows = len(self.train_dataset)
        if cache.get("version") != CACHE_VERSION or cache.get("task_names") != list(TASK_NAMES):
            raise WSMAVFusionDataError("pseudo cache version/task_names mismatch")
        if rows != 6325 or len(cache.get("segment_ids", [])) != 6325 or len(set(cache["segment_ids"])) != 6325:
            raise WSMAVFusionDataError("pseudo cache/canonical TRAIN row count or ID uniqueness mismatch")
        canonical_ids = [s["meta"]["segment_id"] for s in self.train_dataset._samples]
        if cache["segment_ids"] != canonical_ids:
            raise WSMAVFusionDataError("pseudo cache segment IDs do not match canonical TRAIN order")
        observed = _task_tensor(cache, "observed_mask", rows).bool()
        targets = _task_tensor(cache, "observed_targets", rows).float()
        canonical_observed = torch.stack([s["observed_mask"] for s in self.train_dataset._samples])
        canonical_targets = torch.stack([s["targets"] for s in self.train_dataset._samples])
        target_nan_match = torch.equal(torch.isnan(targets), torch.isnan(canonical_targets))
        target_values_match = torch.equal(targets[torch.isfinite(targets)], canonical_targets[torch.isfinite(canonical_targets)])
        if not torch.equal(observed, canonical_observed) or not target_nan_match or not target_values_match:
            raise WSMAVFusionDataError("pseudo cache observed truth does not match canonical TRAIN")
        if cache.get("clip_model_name") != CLIP_MODEL or cache.get("clip_model_revision") != CLIP_REVISION:
            raise WSMAVFusionDataError("pseudo cache CLIP identity mismatch")
        if cache.get("audio_checkpoint_sha256") != AUDIO_SHA256 or cache.get("video_checkpoint_sha256") != VIDEO_SHA256:
            raise WSMAVFusionDataError("pseudo cache teacher checkpoint SHA mismatch")
        if cache.get("semantic_prompt_bank_sha256") != PROMPT_BANK_SHA256:
            raise WSMAVFusionDataError("pseudo cache prompt-bank SHA mismatch")
        accept = _task_tensor(cache, "pseudo_accept_mask", rows).bool()
        pseudo_targets = _task_tensor(cache, "pseudo_targets", rows)
        reliability = _task_tensor(cache, "pseudo_reliability", rows)
        pseudo_class = _task_tensor(cache, "pseudo_class", rows).long()
        calibrated = _task_tensor(cache, "calibrated_audio_probs", rows)
        if bool((accept & observed).any()):
            raise WSMAVFusionDataError("pseudo acceptance overlaps observed truth")
        accepted = accept
        if not bool(torch.isfinite(pseudo_targets[accepted]).all()) or not bool(((pseudo_targets[accepted] >= 0) & (pseudo_targets[accepted] <= 1)).all()):
            raise WSMAVFusionDataError("accepted pseudo targets are invalid")
        if not torch.equal(pseudo_targets[accepted], calibrated[accepted]):
            raise WSMAVFusionDataError("accepted pseudo targets differ from calibrated audio probabilities")
        if not bool(torch.isfinite(reliability[accepted]).all()) or not bool(((reliability[accepted] >= 0) & (reliability[accepted] <= 1)).all()):
            raise WSMAVFusionDataError("accepted pseudo reliability is invalid")
        rejected_or_observed = ~accepted
        if bool((reliability[rejected_or_observed] != 0).any()) or not bool(torch.isnan(pseudo_targets[rejected_or_observed]).all()):
            raise WSMAVFusionDataError("rejected/observed pseudo fields are not neutral")
        if bool((pseudo_class[rejected_or_observed] != -1).any()) or not set(torch.unique(pseudo_class).tolist()).issubset({-1, 0, 1}):
            raise WSMAVFusionDataError("pseudo classes violate the frozen contract")
        accepted_counts = tuple(int(accepted[:, i].sum()) for i in range(2))
        missing_counts = tuple(int((~observed[:, i]).sum()) for i in range(2))
        positive_counts = tuple(int((accepted[:, i] & (pseudo_class[:, i] == 1)).sum()) for i in range(2))
        negative_counts = tuple(int((accepted[:, i] & (pseudo_class[:, i] == 0)).sum()) for i in range(2))
        if (accepted_counts, missing_counts, positive_counts, negative_counts) != (expected_accepted, EXPECTED_MISSING, expected_positive, expected_negative):
            raise WSMAVFusionDataError("pseudo accepted/missing/class counts mismatch")
        self.pseudo_accept_mask = accept.clone()
        self.pseudo_targets = pseudo_targets.clone()
        self.pseudo_reliability = reliability.clone()
        self.pseudo_class = pseudo_class.clone()
        self.train_segment_ids = tuple(canonical_ids)
        self.train_observed_mask = observed.clone()
        self.audit.update({
            "pseudo_cache_path": str(cache_path),
            "pseudo_cache_sha256": _sha256(cache_path),
            "pseudo_cache_version": CACHE_VERSION,
            "pseudo_cache_clip_model": CLIP_MODEL,
            "pseudo_cache_clip_revision": CLIP_REVISION,
            "pseudo_cache_audio_checkpoint_sha256": AUDIO_SHA256,
            "pseudo_cache_video_checkpoint_sha256": VIDEO_SHA256,
            "pseudo_cache_prompt_bank_sha256": PROMPT_BANK_SHA256,
            "pseudo_missing_counts": list(EXPECTED_MISSING),
            "pseudo_accepted_counts": list(expected_accepted),
            "pseudo_accepted_positive_counts": list(expected_positive),
            "pseudo_accepted_negative_counts": list(expected_negative),
        })
        self.train_dataset = _PseudoDataset(self.train_dataset, {
            "pseudo_accept_mask": self.pseudo_accept_mask,
            "pseudo_targets": self.pseudo_targets,
            "pseudo_reliability": self.pseudo_reliability,
            "pseudo_class": self.pseudo_class,
        }, train=True)
        self.val_dataset = _PseudoDataset(self.val_dataset, {}, train=False)
        self.test_dataset = {name: _PseudoDataset(dataset, {}, train=False) for name, dataset in self.test_dataset.items()}
        self.collate_fn = partial(collate_wsm_ramps_semantic, modality_available_override=self.modality_available_override)

    def describe_context(self, context: Any) -> None:
        super().describe_context(context)
        context.set("data.pseudo_cache_path", self.audit["pseudo_cache_path"])
        context.set("data.pseudo_cache_sha256", self.audit["pseudo_cache_sha256"])
        context.set("data.pseudo_accepted_counts", self.audit["pseudo_accepted_counts"])


class WSMRampsSemanticDevOnlyDataModule(WSMRampsSemanticDataModule):
    """RAMPS data contract with validation restricted to canonical DEV.

    The base module deliberately exposes the three Test protocols alongside DEV.
    This narrow extension is for explicitly DEV-only experiments: it preserves the
    canonical TRAIN and DEV datasets while making Test loaders unreachable from
    the normal trainer validation loop.
    """

    def val_dataloader(self) -> dict[str, Any]:
        return {"dev": self._make_loader(self.val_dataset, shuffle=False, drop_last=False)}


@DATAMODULES.register("wsm_ramps_semantic_datamodule")
def wsm_ramps_semantic_datamodule(context: Any | None = None, **params: Any) -> WSMRampsSemanticDataModule:
    del context
    return WSMRampsSemanticDataModule(**params)


@DATAMODULES.register("wsm_ramps_semantic_dev_only_datamodule")
def wsm_ramps_semantic_dev_only_datamodule(
    context: Any | None = None, **params: Any
) -> WSMRampsSemanticDevOnlyDataModule:
    del context
    return WSMRampsSemanticDevOnlyDataModule(**params)
