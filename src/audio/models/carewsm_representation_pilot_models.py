"""Experimental DEEP-008 representation-pilot models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput


def _shared_components(
    input_dim: int, hidden_dim: int, dropout: float
) -> tuple[nn.Module, nn.Module]:
    from audio.models.carewsm_audio_models import CAREWSMSequenceModel

    shared = CAREWSMSequenceModel(input_dim, hidden_dim, dropout)
    return shared.projection, shared.heads


def _unpack(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from audio.models.carewsm_audio_models import _unpack as unpack

    return unpack(batch)


class MaskedAttentionPool(nn.Module):
    """Learn a scalar attention weight for each valid input position."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(values).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (values * weights).sum(dim=1)


class CAREWSMAttentiveMLPModel(nn.Module):
    """Reference MLP with learned mask-safe attentive sequence pooling."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.projection, self.heads = _shared_components(
            input_dim, hidden_dim, dropout
        )
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(self, batch: Any) -> ModelOutput:
        values, mask = _unpack(batch)
        features = self.pool(self.projection(values), mask)
        logits = self.heads(features)
        return ModelOutput(preds=logits, aux={"features": features, "logits": logits})


class CAREWSMGatedFusionModel(nn.Module):
    """Native turn-aligned gated fusion for emotion and TRILLsson features."""

    def __init__(
        self,
        input_dim: int,
        family_dims: list[int],
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        if input_dim != 2048 or family_dims != [1024, 1024]:
            raise ValueError("gated fusion requires [1024, 1024] native families")
        super().__init__()
        self.family_dims = family_dims
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU()
                )
                for dim in family_dims
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(dim, hidden_dim) for dim in family_dims]
        )
        _, self.heads = _shared_components(input_dim, hidden_dim, dropout)

    def forward(self, batch: Any) -> ModelOutput:
        values, mask = _unpack(batch)
        family_values = torch.split(values, self.family_dims, dim=-1)
        fused = torch.zeros(
            values.shape[0], values.shape[1], self.projections[0][1].out_features,
            device=values.device,
            dtype=values.dtype,
        )
        for family, projection, gate in zip(
            family_values, self.projections, self.gates
        ):
            fused = fused + torch.sigmoid(gate(family)) * projection(family)
        weights = mask.unsqueeze(-1).to(fused.dtype)
        features = (fused * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.heads(features)
        return ModelOutput(preds=logits, aux={"features": features, "logits": logits})


@MODELS.register("carewsm_attentive_mlp")
def carewsm_attentive_mlp(
    context: Any | None = None, **params: Any
) -> CAREWSMAttentiveMLPModel:
    del context
    return CAREWSMAttentiveMLPModel(**params)


@MODELS.register("carewsm_gated_fusion")
def carewsm_gated_fusion(
    context: Any | None = None, **params: Any
) -> CAREWSMGatedFusionModel:
    del context
    return CAREWSMGatedFusionModel(**params)
