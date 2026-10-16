"""Frozen R3 observed sparse loss with unimodal auxiliary agreement."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss
from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss


class R3AuxAgreementLoss(BaseLoss):
    """Observed main BCE plus fixed, availability-aware auxiliary terms."""

    def __init__(self, aux_weight: float = 0.25, agreement_weight: float = 0.10, eps: float = 1e-8) -> None:
        for name, value in (("aux_weight", aux_weight), ("agreement_weight", agreement_weight), ("eps", eps)):
            if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        self.aux_weight = float(aux_weight)
        self.agreement_weight = float(agreement_weight)
        self.eps = float(eps)
        self._main = WSMMaskedSparseLoss(eps=eps)

    def __call__(self, output: ModelOutput, batch: Any) -> torch.Tensor:
        targets = batch.targets
        observed = batch.get_masks("observed_mask").to(dtype=torch.bool, device=output.preds.device)
        targets = targets.to(device=output.preds.device, dtype=output.preds.dtype)
        main = self._main.compute_from_tensors(output.preds, targets, observed)
        audio_logits = output.aux["audio_aux_logits"]
        video_logits = output.aux["video_aux_logits"]
        audio_valid = output.aux["audio_aux_valid"].to(dtype=torch.bool, device=audio_logits.device)
        video_valid = output.aux["video_aux_valid"].to(dtype=torch.bool, device=video_logits.device)
        audio_mask = observed & audio_valid
        video_mask = observed & video_valid
        aux_sum = output.preds.sum() * 0.0
        count = int(audio_mask.sum().item()) + int(video_mask.sum().item())
        if count:
            audio_targets = targets.to(device=audio_logits.device, dtype=audio_logits.dtype)
            aux_sum = F.binary_cross_entropy_with_logits(audio_logits[audio_mask], audio_targets[audio_mask], reduction="sum")
            aux_sum = aux_sum + F.binary_cross_entropy_with_logits(video_logits[video_mask], audio_targets.to(video_logits.device)[video_mask], reduction="sum")
            aux = aux_sum / (count + self.eps)
        else:
            aux = aux_sum
        agreement_mask = observed & audio_valid & video_valid
        if int(agreement_mask.sum().item()):
            pa = torch.sigmoid(audio_logits[agreement_mask])
            pv = torch.sigmoid(video_logits[agreement_mask.to(video_logits.device)])
            agreement = ((pa - pv) ** 2).mean()
        else:
            agreement = output.preds.sum() * 0.0
        return main + self.aux_weight * aux + self.agreement_weight * agreement


@LOSSES.register("wsm_r3_aux_agreement_loss")
def wsm_r3_aux_agreement_loss(context: Any | None = None, **params: Any) -> R3AuxAgreementLoss:
    del context
    return R3AuxAgreementLoss(**params)
