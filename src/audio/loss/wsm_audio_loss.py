from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss


class WSMAudioLoss(BaseLoss):
    def __init__(
        self,
        class_weights: list[list[float]] | None = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        aux_weight: float = 0.15,
    ) -> None:
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.aux_weight = float(aux_weight)

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        targets = batch.targets.long().view(-1)
        task_ids = batch.inputs["task_ids"].long().view(-1)
        loss = _classification_loss(
            output.preds,
            targets,
            task_ids,
            class_weights=self.class_weights,
            label_smoothing=self.label_smoothing,
            focal_gamma=self.focal_gamma,
        )

        aux = output.aux or {}
        if self.aux_weight > 0.0 and "aux_logits" in aux:
            loss = loss + self.aux_weight * _classification_loss(
                aux["aux_logits"],
                targets,
                task_ids,
                class_weights=self.class_weights,
                label_smoothing=self.label_smoothing,
                focal_gamma=self.focal_gamma,
            )

        return loss


@LOSSES.register("wsm_audio_loss")
def wsm_audio_loss(context: Any | None = None, **params: Any) -> WSMAudioLoss:
    params = dict(params)
    if context is not None:
        params.setdefault("class_weights", context.get("data.class_weights"))

    return WSMAudioLoss(**params)


def _classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    focal_gamma: float,
) -> torch.Tensor:
    logits = logits.float()
    losses: list[torch.Tensor] = []
    for task_id in torch.unique(task_ids).tolist():
        mask = task_ids == int(task_id)
        task_logits = logits[mask]
        task_targets = targets[mask]
        ce = F.cross_entropy(task_logits, task_targets, reduction="none", label_smoothing=float(label_smoothing))

        if float(focal_gamma) > 0.0:
            probs = torch.softmax(task_logits, dim=-1)
            pt = probs.gather(dim=1, index=task_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
            ce = ce * (1.0 - pt).pow(float(focal_gamma))

        if class_weights is not None:
            weights = class_weights.to(device=logits.device, dtype=torch.float32)
            weights = weights[task_targets] if weights.ndim == 1 else weights[int(task_id), task_targets]
            ce = (ce * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            ce = ce.mean()

        losses.append(ce)

    return torch.stack(losses).mean()
