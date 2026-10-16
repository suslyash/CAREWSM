"""Observed plus detached reliability-weighted pseudo-label loss for RAMPS."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from chimera_ml.core.registry import LOSSES
from chimera_ml.losses.base import BaseLoss


def _value(batch: Any, name: str) -> torch.Tensor:
    if isinstance(batch, Mapping):
        value = batch.get(name)
    elif name == "targets":
        value = getattr(batch, "targets", None)
    else:
        value = None
        getter = getattr(batch, "get_masks", None)
        if callable(getter):
            value = getter(name)
        if value is None and isinstance(getattr(batch, "inputs", None), Mapping):
            value = batch.inputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch must provide tensor {name!r}")
    return value


class WSMRampsObservedPseudoLoss(BaseLoss):
    def __init__(self, pseudo_scale: float = 0.0, eps: float = 1e-8) -> None:
        if not 0.0 <= float(pseudo_scale) <= 1.0:
            raise ValueError("pseudo_scale must be in [0, 1]")
        if float(eps) < 0.0:
            raise ValueError("eps must be non-negative")
        self.pseudo_scale = float(pseudo_scale)
        self.eps = float(eps)

    def compute_components(self, logits: torch.Tensor, targets: torch.Tensor, observed_mask: torch.Tensor, pseudo_accept_mask: torch.Tensor, pseudo_targets: torch.Tensor, pseudo_reliability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or tuple(logits.shape[1:]) != (2,):
            raise ValueError("logits must have shape [B, 2]")
        tensors = {"targets": targets, "observed_mask": observed_mask, "pseudo_accept_mask": pseudo_accept_mask, "pseudo_targets": pseudo_targets, "pseudo_reliability": pseudo_reliability}
        if any(not isinstance(v, torch.Tensor) or tuple(v.shape) != tuple(logits.shape) for v in tensors.values()):
            raise ValueError("targets and pseudo fields must have shape [B, 2]")
        device, dtype = logits.device, logits.dtype
        observed = observed_mask.to(device=device, dtype=torch.bool)
        accept = pseudo_accept_mask.to(device=device, dtype=torch.bool)
        target = targets.to(device=device, dtype=dtype)
        pseudo = pseudo_targets.to(device=device, dtype=dtype).detach()
        reliability = pseudo_reliability.to(device=device, dtype=dtype).detach()
        if bool((accept & observed).any()):
            raise ValueError("pseudo acceptance overlaps observed truth")
        if bool(observed.any()) and (not bool(torch.isfinite(target[observed]).all()) or not bool(torch.logical_or(target[observed] == 0, target[observed] == 1).all())):
            raise ValueError("observed targets must be finite binary 0/1")
        rejected_or_observed = ~accept
        if bool(torch.isfinite(pseudo[rejected_or_observed]).any()):
            raise ValueError("rejected/observed pseudo targets must be NaN")
        if bool((reliability[rejected_or_observed] != 0).any()):
            raise ValueError("rejected/observed pseudo reliability must be zero")
        pseudo_mask = accept & ~observed
        if bool(pseudo_mask.any()):
            if not bool(torch.isfinite(pseudo[pseudo_mask]).all()) or not bool(((pseudo[pseudo_mask] >= 0) & (pseudo[pseudo_mask] <= 1)).all()):
                raise ValueError("accepted pseudo targets must be finite in [0, 1]")
            if not bool(torch.isfinite(reliability[pseudo_mask]).all()) or not bool(((reliability[pseudo_mask] >= 0) & (reliability[pseudo_mask] <= 1)).all()):
                raise ValueError("accepted pseudo reliability must be finite in [0, 1]")
        if bool(observed.any()):
            observed_loss = F.binary_cross_entropy_with_logits(logits[observed], target[observed], reduction="sum") / (int(observed.sum()) + self.eps)
        else:
            observed_loss = logits.sum() * 0.0
        if bool(pseudo_mask.any()):
            weighted = reliability[pseudo_mask] * F.binary_cross_entropy_with_logits(logits[pseudo_mask], pseudo[pseudo_mask].detach(), reduction="none")
            pseudo_loss = weighted.sum() / (reliability[pseudo_mask].detach().sum() + self.eps)
        else:
            pseudo_loss = logits.sum() * 0.0
        return observed_loss, pseudo_loss

    def __call__(self, output: Any, batch: Any) -> torch.Tensor:
        components = self.compute_components(output.preds, _value(batch, "targets"), _value(batch, "observed_mask"), _value(batch, "pseudo_accept_mask"), _value(batch, "pseudo_targets"), _value(batch, "pseudo_reliability"))
        return components[0] + self.pseudo_scale * components[1]


@LOSSES.register("wsm_ramps_observed_pseudo_loss")
def wsm_ramps_observed_pseudo_loss(context: Any | None = None, **params: Any) -> WSMRampsObservedPseudoLoss:
    del context
    return WSMRampsObservedPseudoLoss(**params)
