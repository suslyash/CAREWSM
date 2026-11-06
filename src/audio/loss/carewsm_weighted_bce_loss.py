"""TRAIN-only Parkinson-weighted observed-label multitask BCE loss."""

from __future__ import annotations

import math
from typing import Any

import torch

from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss

from audio.loss.carewsm_masked_bce_loss import CAREWSMMaskedBCELoss


class CAREWSMWeightedParkinsonBCELoss(BaseLoss):
    """Keep Depression unweighted and apply one frozen Parkinson weight."""

    def __init__(self, parkinson_pos_weight: float) -> None:
        if not math.isfinite(parkinson_pos_weight) or parkinson_pos_weight <= 0:
            raise ValueError("parkinson_pos_weight must be finite and positive")
        self.parkinson_pos_weight = float(parkinson_pos_weight)

    def __call__(self, output: ModelOutput, batch: Any) -> torch.Tensor:
        observed = batch.get_masks("observed_targets")
        if observed is None:
            observed = batch.inputs.get("observed_targets")
        if observed is None and batch.meta is not None:
            observed = batch.meta.get("observed_targets")
            if observed is None and "sample_meta" in batch.meta:
                observed = torch.stack(
                    [item["observed_targets"] for item in batch.meta["sample_meta"]]
                )
        if observed is None:
            raise ValueError("Batch does not contain observed-label masks")

        pos_weight = torch.tensor(
            [1.0, self.parkinson_pos_weight],
            dtype=output.preds.dtype,
            device=output.preds.device,
        )
        return CAREWSMMaskedBCELoss.compute(
            output.preds, batch.targets, observed, pos_weight=pos_weight
        )


@LOSSES.register("carewsm_weighted_parkinson_bce")
def carewsm_weighted_parkinson_bce(
    context: Any | None = None,
    **params: Any,
) -> CAREWSMWeightedParkinsonBCELoss:
    del context
    return CAREWSMWeightedParkinsonBCELoss(**params)
