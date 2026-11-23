"""Provisional audio/video/text fusion models for the AV-003 contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput

MODALITIES = ("audio", "video", "text")


def _mask_for(batch: Any, modality: str, values: torch.Tensor) -> torch.Tensor:
    """Return a boolean [B,T] mask from AV-002 lengths or valid masks."""
    mask = batch.get_masks(f"{modality}_valid_mask")
    if mask is None and modality == "video":
        mask = batch.get_masks("video_valid_mask")
    if mask is not None:
        return mask.to(device=values.device, dtype=torch.bool)
    lengths = batch.get_masks(f"{modality}_lengths")
    if lengths is None and batch.meta:
        lengths = batch.meta.get(f"{modality}_lengths")
    if lengths is None:
        return torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
    steps = torch.arange(values.shape[1], device=values.device)
    return steps.unsqueeze(0) < lengths.to(device=values.device).unsqueeze(1)


def _inputs(
    batch: Any, dimensions: dict[str, int]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    values = {}
    masks = {}
    for modality in MODALITIES:
        if modality not in batch.inputs:
            raise ValueError(f"batch.inputs is missing modality {modality!r}")
        tensor = batch.inputs[modality].float()
        if tensor.ndim != 3 or tensor.shape[-1] != dimensions[modality]:
            raise ValueError(
                f"inputs[{modality!r}] must have shape [B,T,{dimensions[modality]}]"
            )
        values[modality] = tensor
        masks[modality] = _mask_for(batch, modality, tensor)
    return values, masks


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_tokens(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values * mask.unsqueeze(-1).to(values.dtype)


class _TaskHeads(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.depression = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.parkinson = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.depression(features), self.parkinson(features)), dim=-1)


class _TrimodalBase(nn.Module):
    def __init__(
        self, audio_dim: int, video_dim: int, text_dim: int,
        hidden_dim: int = 64, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dimensions = {"audio": audio_dim, "video": video_dim, "text": text_dim}
        self.projections = nn.ModuleDict({
            modality: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU())
            for modality, dim in self.dimensions.items()
        })
        self.heads = _TaskHeads(hidden_dim, dropout)

    def _project(
        self, batch: Any
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        values, masks = _inputs(batch, self.dimensions)
        projected = {
            modality: self.projections[modality](values[modality])
            for modality in MODALITIES
        }
        return projected, masks

    def _output(self, features: torch.Tensor, aux: dict[str, Any]) -> ModelOutput:
        logits = self.heads(features)
        return ModelOutput(preds=logits, aux={**aux, "logits": logits})


class CAREWSMTextGuidedDualGatedFusion(_TrimodalBase):
    """Text-conditioned channel and inter-modality gated residual fusion."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.projections["audio"][1].out_features
        self.audio_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.video_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.modality_gate = nn.Linear(hidden_dim * 3, 3)
        self.refine = nn.ModuleDict({
            modality: nn.Linear(hidden_dim * 2, hidden_dim)
            for modality in MODALITIES
        })

    def forward(self, batch: Any) -> ModelOutput:
        projected, masks = self._project(batch)
        pooled = {m: _masked_mean(projected[m], masks[m]) for m in MODALITIES}
        text = pooled["text"]
        audio_condition = torch.cat((pooled["audio"], text), dim=-1)
        video_condition = torch.cat((pooled["video"], text), dim=-1)
        audio_gate = torch.sigmoid(self.audio_gate(audio_condition))
        video_gate = torch.sigmoid(self.video_gate(video_condition))
        audio = pooled["audio"] + audio_gate * torch.tanh(self.refine["audio"](audio_condition))
        video = pooled["video"] + video_gate * torch.tanh(self.refine["video"](video_condition))
        components = torch.stack((audio, video, text), dim=1)
        weights = torch.softmax(self.modality_gate(torch.cat((audio, video, text), dim=-1)), dim=-1)
        fused = (components * weights.unsqueeze(-1)).sum(dim=1) + (audio + video + text) / 3
        return self._output(
            fused,
            {
                "audio_gate": audio_gate.mean(-1),
                "video_gate": video_gate.mean(-1),
                "modality_gate": weights,
            },
        )


class CAREWSMTriModalCrossAttentionResidualFusion(_TrimodalBase):
    """Mask-safe cross-modal attention with an explicit singleton fallback."""

    def __init__(self, num_heads: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.projections["audio"][1].out_features
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.static_fallback = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.residual = nn.ModuleDict({m: nn.Linear(hidden_dim, hidden_dim) for m in MODALITIES})
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

    def _attend_to_others(
        self, query: torch.Tensor, query_mask: torch.Tensor,
        others: list[torch.Tensor], other_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        keys = torch.cat(others, dim=1)
        valid = torch.cat(other_masks, dim=1)
        attended, _ = self.attention(query, keys, keys, key_padding_mask=~valid)
        return _masked_mean(attended, query_mask)

    def forward(self, batch: Any) -> ModelOutput:
        projected, masks = self._project(batch)
        pooled = {m: _masked_mean(projected[m], masks[m]) for m in MODALITIES}
        singleton = all(projected[m].shape[1] == 1 for m in MODALITIES)
        if singleton:
            fused = self.static_fallback(torch.cat(tuple(pooled[m] for m in MODALITIES), dim=-1))
            path = "static_fallback"
        else:
            attended = {
                modality: self._attend_to_others(
                    projected[modality], masks[modality],
                    [projected[other] for other in MODALITIES if other != modality],
                    [masks[other] for other in MODALITIES if other != modality],
                )
                for modality in MODALITIES
            }
            residuals = [pooled[m] + self.residual[m](attended[m]) for m in MODALITIES]
            fused = self.fusion(torch.cat(residuals, dim=-1))
            path = "cross_attention"
        return self._output(fused, {"fusion_path": path})


class CAREWSMSharedPrivateTrimodalFusion(_TrimodalBase):
    """Six-component shared/private fusion with learned component gates."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.projections["audio"][1].out_features
        self.shared = nn.ModuleDict({m: nn.Linear(hidden_dim, hidden_dim) for m in MODALITIES})
        self.private = nn.ModuleDict({m: nn.Linear(hidden_dim, hidden_dim) for m in MODALITIES})
        self.component_gate = nn.Linear(hidden_dim, 1)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    def forward(self, batch: Any) -> ModelOutput:
        projected, masks = self._project(batch)
        components = []
        for modality in MODALITIES:
            pooled = _masked_mean(projected[modality], masks[modality])
            components.extend((self.shared[modality](pooled), self.private[modality](pooled)))
        stacked = torch.stack(components, dim=1)
        weights = torch.softmax(self.component_gate(torch.tanh(stacked)).squeeze(-1), dim=1)
        fused = self.fusion((stacked * weights.unsqueeze(-1)).sum(dim=1) + stacked.mean(dim=1))
        return self._output(fused, {"component_gate": weights})


@MODELS.register("carewsm_text_guided_dual_gated")
def carewsm_text_guided_dual_gated(
    context: Any | None = None, **params: Any
) -> CAREWSMTextGuidedDualGatedFusion:
    del context
    return CAREWSMTextGuidedDualGatedFusion(**params)


@MODELS.register("carewsm_trimodal_cross_attention")
def carewsm_trimodal_cross_attention(
    context: Any | None = None, **params: Any
) -> CAREWSMTriModalCrossAttentionResidualFusion:
    del context
    return CAREWSMTriModalCrossAttentionResidualFusion(**params)


@MODELS.register("carewsm_shared_private_trimodal")
def carewsm_shared_private_trimodal(
    context: Any | None = None, **params: Any
) -> CAREWSMSharedPrivateTrimodalFusion:
    del context
    return CAREWSMSharedPrivateTrimodalFusion(**params)
