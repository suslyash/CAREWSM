"""Project optimizers for models with intentionally frozen submodules."""
from __future__ import annotations

from typing import Any

import torch

from chimera_ml.core.registry import OPTIMIZERS


@OPTIMIZERS.register("wsm_trainable_adamw_optimizer")
def wsm_trainable_adamw_optimizer(
    *,
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> torch.optim.AdamW:
    """Build AdamW from exactly the parameters marked trainable."""
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("wsm_trainable_adamw_optimizer requires trainable parameters")
    return torch.optim.AdamW(
        trainable_parameters,
        lr=float(lr),
        weight_decay=float(weight_decay),
        **kwargs,
    )
