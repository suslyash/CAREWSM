"""Batch 01 corpus-balanced and two-task weighted BCE losses."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss

from audio.loss.carewsm_masked_bce_loss import CAREWSMMaskedBCELoss


def _observed_targets(batch: Any) -> torch.Tensor:
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
    return observed


def _corpora(batch: Any) -> list[str]:
    if batch.meta is None or "sample_meta" not in batch.meta:
        raise ValueError("Batch does not contain sample corpus metadata")
    return [item["corpus"] for item in batch.meta["sample_meta"]]


class CAREWSMCorpusBalancedWeightedBCELoss(BaseLoss):
    """Equalize available CARE and WSM BCE contributions within each task."""

    def __init__(self, parkinson_pos_weight: float) -> None:
        if not math.isfinite(parkinson_pos_weight) or parkinson_pos_weight <= 0:
            raise ValueError("parkinson_pos_weight must be finite and positive")
        self.parkinson_pos_weight = float(parkinson_pos_weight)

    def __call__(self, output: ModelOutput, batch: Any) -> torch.Tensor:
        logits = output.preds
        targets = batch.targets.to(device=logits.device, dtype=logits.dtype)
        observed = _observed_targets(batch).to(device=logits.device, dtype=torch.bool)
        corpora = _corpora(batch)
        if len(corpora) != logits.shape[0]:
            raise ValueError("sample corpus metadata does not match batch size")
        weights = torch.tensor(
            [1.0, self.parkinson_pos_weight],
            device=logits.device,
            dtype=logits.dtype,
        )

        task_losses: list[torch.Tensor] = []
        for task in range(2):
            corpus_losses: list[torch.Tensor] = []
            for corpus in ("CARE", "WSM"):
                mask = observed[:, task] & torch.tensor(
                    [value == corpus for value in corpora],
                    device=logits.device,
                )
                if mask.any():
                    corpus_losses.append(
                        F.binary_cross_entropy_with_logits(
                            logits[mask, task],
                            targets[mask, task],
                            pos_weight=weights[task],
                        )
                    )
            if corpus_losses:
                task_losses.append(torch.stack(corpus_losses).mean())
        if not task_losses:
            return logits.sum() * 0.0
        return torch.stack(task_losses).mean()


class CAREWSMWeightedDepressionParkinsonBCELoss(BaseLoss):
    """Apply TRAIN-derived positive weights to both observed-label tasks."""

    def __init__(
        self,
        depression_pos_weight: float,
        parkinson_pos_weight: float,
    ) -> None:
        for name, value in (
            ("depression_pos_weight", depression_pos_weight),
            ("parkinson_pos_weight", parkinson_pos_weight),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.depression_pos_weight = float(depression_pos_weight)
        self.parkinson_pos_weight = float(parkinson_pos_weight)

    def __call__(self, output: ModelOutput, batch: Any) -> torch.Tensor:
        observed = _observed_targets(batch)
        pos_weight = torch.tensor(
            [self.depression_pos_weight, self.parkinson_pos_weight],
            dtype=output.preds.dtype,
            device=output.preds.device,
        )
        return CAREWSMMaskedBCELoss.compute(
            output.preds,
            batch.targets,
            observed,
            pos_weight=pos_weight,
        )


@LOSSES.register("carewsm_corpus_balanced_weighted_bce")
def carewsm_corpus_balanced_weighted_bce(
    context: Any | None = None,
    **params: Any,
) -> CAREWSMCorpusBalancedWeightedBCELoss:
    del context
    return CAREWSMCorpusBalancedWeightedBCELoss(**params)


@LOSSES.register("carewsm_weighted_depression_parkinson_bce")
def carewsm_weighted_depression_parkinson_bce(
    context: Any | None = None,
    **params: Any,
) -> CAREWSMWeightedDepressionParkinsonBCELoss:
    del context
    return CAREWSMWeightedDepressionParkinsonBCELoss(**params)
