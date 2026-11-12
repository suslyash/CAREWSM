"""Static recording-level models for the audio_turns_combined program."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput

from audio.models.carewsm_audio_models import TaskHeads, _unpack


class CAREWSMStaticMLPModel(nn.Module):
    """Layer-normalized static MLP with independent task heads."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = TaskHeads(hidden_dim, dropout)

    def forward(self, batch: Any) -> ModelOutput:
        values, _ = _unpack(batch)
        features = self.encoder(values[:, 0, :])
        logits = self.heads(features)
        return ModelOutput(preds=logits, aux={"features": features, "logits": logits})


class CAREWSMStaticResidualMLPModel(CAREWSMStaticMLPModel):
    """Static residual alternative used only by the bounded tuning stage."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1
    ) -> None:
        nn.Module.__init__(self)
        self.normalization = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.heads = TaskHeads(hidden_dim, dropout)

    def forward(self, batch: Any) -> ModelOutput:
        values, _ = _unpack(batch)
        features = self.input_projection(self.normalization(values[:, 0, :]))
        features = features + self.block(features)
        logits = self.heads(features)
        return ModelOutput(preds=logits, aux={"features": features, "logits": logits})


class CAREWSMStaticGatedFusionModel(nn.Module):
    """Feature-family gating over one static vector per recording."""

    def __init__(
        self,
        input_dim: int,
        family_dims: list[int],
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if sum(family_dims) != input_dim or not family_dims:
            raise ValueError("family_dims must sum to input_dim")
        self.family_dims = tuple(family_dims)
        self.normalizations = nn.ModuleList(nn.LayerNorm(dim) for dim in family_dims)
        self.projections = nn.ModuleList(
            nn.Linear(dim, hidden_dim) for dim in family_dims
        )
        self.gates = nn.ModuleList(nn.Linear(dim, hidden_dim) for dim in family_dims)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.heads = TaskHeads(hidden_dim, dropout)

    def forward(self, batch: Any) -> ModelOutput:
        values, _ = _unpack(batch)
        families = torch.split(values[:, 0, :], self.family_dims, dim=-1)
        fused = torch.zeros(
            values.shape[0],
            self.projections[0].out_features,
            device=values.device,
            dtype=values.dtype,
        )
        gates = []
        for family, normalization, projection, gate in zip(
            families, self.normalizations, self.projections, self.gates, strict=True
        ):
            normalized = normalization(family)
            projected = self.activation(projection(normalized))
            gate_values = torch.sigmoid(gate(normalized))
            fused = fused + gate_values * projected
            gates.append(gate_values)
        features = self.dropout(fused)
        logits = self.heads(features)
        return ModelOutput(
            preds=logits,
            aux={"features": features, "logits": logits, "family_gates": gates},
        )


@MODELS.register("carewsm_static_mlp")
def carewsm_static_mlp(
    context: Any | None = None, **params: Any
) -> CAREWSMStaticMLPModel:
    del context
    return CAREWSMStaticMLPModel(**params)


@MODELS.register("carewsm_static_residual_mlp")
def carewsm_static_residual_mlp(
    context: Any | None = None, **params: Any
) -> CAREWSMStaticResidualMLPModel:
    del context
    return CAREWSMStaticResidualMLPModel(**params)


@MODELS.register("carewsm_static_gated_fusion")
def carewsm_static_gated_fusion(
    context: Any | None = None, **params: Any
) -> CAREWSMStaticGatedFusionModel:
    del context
    return CAREWSMStaticGatedFusionModel(**params)
