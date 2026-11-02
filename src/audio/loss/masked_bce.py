"""Observed-label-only multitask binary loss."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss


class MaskedMultitaskBCE(BaseLoss):
    """Equal mean of Depression/Parkinson BCE over observed labels."""

    def __call__(self, output: ModelOutput, batch: Any) -> torch.Tensor:
        observed = batch.get_masks("observed_targets")
        if observed is None:
            observed = batch.inputs.get("observed_targets")
        if observed is None and batch.meta is not None:
            observed = batch.meta.get("observed_targets")
            if observed is None and "sample_meta" in batch.meta:
                observed = torch.stack([item["observed_targets"] for item in batch.meta["sample_meta"]])
        if observed is None:
            raise ValueError("Batch does not contain observed-label masks")
        return self.compute(output.preds, batch.targets, observed)

    @staticmethod
    def compute(logits: torch.Tensor, targets: torch.Tensor,
                observed: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for task in range(2):
            mask = observed[:, task].bool()
            if mask.any():
                weight = None if pos_weight is None else pos_weight[task]
                losses.append(F.binary_cross_entropy_with_logits(
                    logits[mask, task], targets[mask, task].float(), pos_weight=weight
                ))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


@LOSSES.register("carewsm_masked_bce")
def carewsm_masked_bce(context: Any | None = None, **params: Any) -> MaskedMultitaskBCE:
    return MaskedMultitaskBCE(**params)
