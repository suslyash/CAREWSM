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


class CAREWSMGatedTaskAttentionModel(nn.Module):
    """Gated fusion with independent masked attention for each task."""

    def __init__(
        self,
        input_dim: int,
        family_dims: list[int],
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        if input_dim != 2048 or family_dims != [1024, 1024]:
            raise ValueError(
                "gated task attention requires [1024, 1024] native families"
            )
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
        self.depression_attention = nn.Linear(hidden_dim, 1)
        self.parkinson_attention = nn.Linear(hidden_dim, 1)

    def _fuse(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        family_values = torch.split(values, self.family_dims, dim=-1)
        fused = torch.zeros(
            values.shape[0],
            values.shape[1],
            self.projections[0][1].out_features,
            device=values.device,
            dtype=values.dtype,
        )
        for family, projection, gate in zip(
            family_values, self.projections, self.gates
        ):
            fused = fused + torch.sigmoid(gate(family)) * projection(family)
        return fused * mask.unsqueeze(-1).to(fused.dtype)

    @staticmethod
    def _pool(
        values: torch.Tensor,
        mask: torch.Tensor,
        scorer: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = scorer(values).squeeze(-1).masked_fill(~mask, -torch.inf)
        attention = torch.softmax(scores, dim=1)
        pooled = (values * attention.unsqueeze(-1)).sum(dim=1)
        return pooled, attention

    def forward(self, batch: Any) -> ModelOutput:
        values, mask = _unpack(batch)
        if not mask.any(dim=1).all():
            raise ValueError("gated task attention requires at least one valid turn")

        fused = self._fuse(values, mask)
        depression_features, depression_attention = self._pool(
            fused, mask, self.depression_attention
        )
        parkinson_features, parkinson_attention = self._pool(
            fused, mask, self.parkinson_attention
        )
        logits = torch.cat(
            (
                self.heads.depression(depression_features),
                self.heads.parkinson(parkinson_features),
            ),
            dim=-1,
        )
        features = torch.stack((depression_features, parkinson_features), dim=1)
        attention = torch.stack((depression_attention, parkinson_attention), dim=1)
        return ModelOutput(
            preds=logits,
            aux={
                "features": features,
                "logits": logits,
                "attention_weights": attention,
            },
        )


class CAREWSMGatedSingleTaskModel(CAREWSMGatedFusionModel):
    """Historical gated fusion with one active task output."""

    def __init__(self, active_task: str, **params: Any) -> None:
        if active_task not in {"depression", "parkinson"}:
            raise ValueError("active_task must be depression or parkinson")
        super().__init__(**params)
        self.active_task = active_task
        self.active_index = 0 if active_task == "depression" else 1

    def forward(self, batch: Any) -> ModelOutput:
        output = super().forward(batch)
        logits = output.preds.clone()
        logits[:, 1 - self.active_index] = 0.0
        return ModelOutput(preds=logits, aux=output.aux)


class CAREWSMGatedCorpusHeadsModel(CAREWSMGatedFusionModel):
    """Gated fusion with separate task heads for CARE and WSM samples."""

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        from audio.models.carewsm_audio_models import TaskHeads

        hidden_dim = self.projections[0][1].out_features
        dropout = params.get("dropout", 0.1)
        self.corpus_heads = nn.ModuleDict(
            {
                "CARE": TaskHeads(hidden_dim, dropout),
                "WSM": TaskHeads(hidden_dim, dropout),
            }
        )
        del self.heads

    def forward(self, batch: Any) -> ModelOutput:
        values, mask = _unpack(batch)
        family_values = torch.split(values, self.family_dims, dim=-1)
        fused = torch.zeros(
            values.shape[0],
            values.shape[1],
            self.projections[0][1].out_features,
            device=values.device,
            dtype=values.dtype,
        )
        for family, projection, gate in zip(
            family_values, self.projections, self.gates
        ):
            fused = fused + torch.sigmoid(gate(family)) * projection(family)
        weights = mask.unsqueeze(-1).to(fused.dtype)
        features = (fused * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        sample_meta = batch.meta.get("sample_meta") if batch.meta else None
        if sample_meta is None:
            raise ValueError("corpus-specific heads require sample metadata")
        corpora = [item.get("corpus") for item in sample_meta]
        if any(corpus not in self.corpus_heads for corpus in corpora):
            raise ValueError("each sample must have a CARE or WSM corpus")
        logits = torch.empty(
            features.shape[0], 2, device=features.device, dtype=features.dtype
        )
        for corpus in self.corpus_heads:
            indices = [index for index, item in enumerate(corpora) if item == corpus]
            if indices:
                index_tensor = torch.tensor(indices, device=features.device)
                logits[index_tensor] = self.corpus_heads[corpus](features[index_tensor])
        return ModelOutput(preds=logits, aux={"features": features, "logits": logits})


@MODELS.register("carewsm_attentive_mlp")
def carewsm_attentive_mlp(
    context: Any | None = None, **params: Any
) -> CAREWSMAttentiveMLPModel:
    del context
    return CAREWSMAttentiveMLPModel(**params)


@MODELS.register("carewsm_gated_task_attention")
def carewsm_gated_task_attention(
    context: Any | None = None, **params: Any
) -> CAREWSMGatedTaskAttentionModel:
    del context
    return CAREWSMGatedTaskAttentionModel(**params)


@MODELS.register("carewsm_gated_fusion")
def carewsm_gated_fusion(
    context: Any | None = None, **params: Any
) -> CAREWSMGatedFusionModel:
    del context
    return CAREWSMGatedFusionModel(**params)


@MODELS.register("carewsm_gated_single_task")
def carewsm_gated_single_task(
    context: Any | None = None, **params: Any
) -> CAREWSMGatedSingleTaskModel:
    del context
    return CAREWSMGatedSingleTaskModel(**params)


@MODELS.register("carewsm_gated_corpus_heads")
def carewsm_gated_corpus_heads(
    context: Any | None = None, **params: Any
) -> CAREWSMGatedCorpusHeadsModel:
    del context
    return CAREWSMGatedCorpusHeadsModel(**params)
