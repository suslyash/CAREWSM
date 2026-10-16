"""Observed-label-only sparse BCE loss for the WSM two-task contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss


class WSMMaskedSparseLoss(BaseLoss):
    """Mean independent BCE-with-logits over observed task elements only."""

    def __init__(self, eps: float = 1e-8) -> None:
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        self.eps = float(eps)

    def __call__(self, output: ModelOutput, batch: Any) -> torch.Tensor:
        targets = _batch_value(batch, "targets")
        observed_mask = _batch_value(batch, "observed_mask")
        return self.compute_from_tensors(output.preds, targets, observed_mask)

    def compute_from_tensors(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("logits must be a tensor with shape [B, 2]")
        if not isinstance(targets, torch.Tensor) or targets.shape != logits.shape:
            raise ValueError("targets must be a tensor with shape [B, 2]")
        if not isinstance(observed_mask, torch.Tensor) or observed_mask.shape != logits.shape:
            raise ValueError("observed_mask must be a tensor with shape [B, 2]")

        mask = observed_mask.to(device=logits.device, dtype=torch.bool)
        observed_count = int(mask.sum().item())
        if observed_count == 0:
            raise ValueError("masked sparse loss requires at least one observed target")

        # Select before validating or evaluating targets so masked NaN placeholders
        # cannot enter BCE or contaminate the scalar loss/gradients.
        selected_logits = logits[mask]
        selected_targets = targets.to(device=logits.device, dtype=logits.dtype)[mask]
        if not torch.isfinite(selected_targets).all():
            raise ValueError("observed targets must be finite")
        if not torch.logical_or(selected_targets == 0, selected_targets == 1).all():
            raise ValueError("observed targets must be binary 0/1")

        elementwise = F.binary_cross_entropy_with_logits(
            selected_logits,
            selected_targets,
            reduction="none",
        )
        return elementwise.sum() / (observed_count + self.eps)


def _batch_value(batch: Any, name: str) -> torch.Tensor:
    if isinstance(batch, Mapping):
        value = batch.get(name)
    elif name == "targets":
        value = getattr(batch, "targets", None)
    else:
        value = None
        get_masks = getattr(batch, "get_masks", None)
        if callable(get_masks):
            value = get_masks(name)
        if value is None:
            inputs = getattr(batch, "inputs", None)
            if isinstance(inputs, Mapping):
                value = inputs.get(name)
        if value is None:
            value = getattr(batch, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch must provide tensor {name!r}")
    return value


@LOSSES.register("wsm_masked_sparse_loss")
def wsm_masked_sparse_loss(
    context: Any | None = None,
    **params: Any,
) -> WSMMaskedSparseLoss:
    del context
    return WSMMaskedSparseLoss(**params)
