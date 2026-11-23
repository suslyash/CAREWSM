"""Registered CAREWSM bimodal and trimodal fusion adaptations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput

ALL_MODALITIES = ("audio", "video", "text")


def _mask_for(batch: Any, modality: str, values: torch.Tensor) -> torch.Tensor:
    mask = batch.get_masks(f"{modality}_valid_mask")
    if mask is None:
        lengths = batch.get_masks(f"{modality}_lengths")
        if lengths is not None:
            steps = torch.arange(values.shape[1], device=values.device)
            mask = steps.unsqueeze(0) < lengths.to(values.device).unsqueeze(1)
    if mask is None:
        mask = torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
    return mask.to(device=values.device, dtype=torch.bool)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class _TaskHeads(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(2)
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(values) for head in self.heads], dim=-1)


class _FusionBase(nn.Module):
    def __init__(
        self,
        modalities=None,
        dimensions=None,
        audio_dim=None,
        video_dim=None,
        text_dim=None,
        hidden_dim=64,
        dropout=0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        if dimensions is None:
            dimensions = {"audio": audio_dim, "video": video_dim, "text": text_dim}
        self.modalities = tuple(modalities or ALL_MODALITIES)
        if len(self.modalities) not in (2, 3) or len(set(self.modalities)) != len(
            self.modalities
        ):
            raise ValueError(
                "fusion models require exactly two or three distinct modalities"
            )
        if any(
            modality not in dimensions or dimensions[modality] is None
            for modality in self.modalities
        ):
            raise ValueError(f"missing dimensions for {self.modalities}")
        self.dimensions = {
            modality: int(dimensions[modality]) for modality in self.modalities
        }
        self.hidden_dim = hidden_dim
        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.LayerNorm(dimension), nn.Linear(dimension, hidden_dim), nn.GELU()
                )
                for modality, dimension in self.dimensions.items()
            }
        )
        self.heads = _TaskHeads(hidden_dim, dropout)

    def _project(self, batch: Any) -> dict[str, torch.Tensor]:
        projected = {}
        for modality in self.modalities:
            if modality not in batch.inputs:
                raise ValueError(f"batch.inputs is missing modality {modality!r}")
            values = batch.inputs[modality].float()
            if values.ndim != 3 or values.shape[-1] != self.dimensions[modality]:
                raise ValueError(
                    f"inputs[{modality!r}] must have shape [B,T,{self.dimensions[modality]}]"
                )
            projected[modality] = self.projections[modality](values)
        return projected

    def _pooled(self, batch: Any) -> dict[str, torch.Tensor]:
        projected = self._project(batch)
        return {
            modality: _masked_mean(
                values, _mask_for(batch, modality, batch.inputs[modality])
            )
            for modality, values in projected.items()
        }

    def _output(self, features: torch.Tensor, aux: dict[str, Any]) -> ModelOutput:
        logits = self.heads(features)
        return ModelOutput(preds=logits, aux={**aux, "logits": logits})


class CAREWSMBimodalTaskWiseGatedLateFusion(_FusionBase):
    """F0 adaptation: task-specific gates fuse two modality logits."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        first, second = self.modalities
        del self.heads
        self.modality_heads = nn.ModuleDict(
            {
                modality: _TaskHeads(self.hidden_dim, kwargs.get("dropout", 0.1))
                for modality in (first, second)
            }
        )
        self.gates = nn.ModuleList(
            [nn.Linear(self.hidden_dim * 2, 1) for _ in range(2)]
        )

    def forward(self, batch: Any) -> ModelOutput:
        pooled = self._pooled(batch)
        first, second = self.modalities
        first_logits = self.modality_heads[first](pooled[first])
        second_logits = self.modality_heads[second](pooled[second])
        gates = torch.cat(
            [
                torch.sigmoid(gate(torch.cat((pooled[first], pooled[second]), -1)))
                for gate in self.gates
            ],
            dim=-1,
        )
        logits = gates * first_logits + (1 - gates) * second_logits
        return ModelOutput(preds=logits, aux={"task_gates": gates})


class CAREWSMBimodalSharedMTLFusion(_FusionBase):
    """F1 adaptation: shared trunk with residual modality information paths."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        count = len(self.modalities)
        self.shared = nn.Sequential(
            nn.Linear(self.hidden_dim * count, self.hidden_dim), nn.GELU()
        )
        self.residual = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in self.modalities]
        )
        self.fuse = nn.Linear(self.hidden_dim * (count + 1), self.hidden_dim)

    def forward(self, batch: Any) -> ModelOutput:
        pooled = self._pooled(batch)
        values = [pooled[modality] for modality in self.modalities]
        shared = self.shared(torch.cat(values, -1))
        fused = self.fuse(
            torch.cat(
                [shared]
                + [layer(value) for layer, value in zip(self.residual, values)],
                -1,
            )
        )
        return self._output(fused, {})


class CAREWSMBimodalTaskAwareDirectedFusion(_FusionBase):
    """F2 adaptation: task-aware weights combine both directed relation paths."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if len(self.modalities) != 2:
            raise ValueError("task-aware directed fusion is bimodal")
        self.relations = nn.ModuleList(
            [nn.Linear(self.hidden_dim * 2, self.hidden_dim) for _ in range(2)]
        )
        self.shared = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.task_weights = nn.ModuleList(
            [nn.Linear(self.hidden_dim * 3, 2) for _ in range(2)]
        )
        self.task_features = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(2)]
        )

    def forward(self, batch: Any) -> ModelOutput:
        pooled = self._pooled(batch)
        first, second = (pooled[modality] for modality in self.modalities)
        relation12 = torch.tanh(self.relations[0](torch.cat((first, second), -1)))
        relation21 = torch.tanh(self.relations[1](torch.cat((second, first), -1)))
        shared = torch.tanh(self.shared(torch.cat((first, second), -1)))
        features = []
        for weight, transform in zip(self.task_weights, self.task_features):
            weights = torch.softmax(
                weight(torch.cat((shared, relation12, relation21), -1)), -1
            )
            features.append(
                shared
                + weights[:, :1] * transform(relation12)
                + weights[:, 1:] * transform(relation21)
            )
        logits = torch.cat(
            [self.heads.heads[index](features[index]) for index in range(2)], -1
        )
        return ModelOutput(
            preds=logits,
            aux={
                "directed_weights": torch.stack(
                    [
                        torch.softmax(
                            weight(torch.cat((shared, relation12, relation21), -1)), -1
                        )
                        for weight in self.task_weights
                    ],
                    1,
                )
            },
        )


class CAREWSMTextGuidedDualGatedFusion(_FusionBase):
    """Text-conditioned channel gates with residual paths."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.modalities != ALL_MODALITIES:
            raise ValueError("text-guided fusion requires audio, video, and text")
        self.audio_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.video_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.refine = nn.ModuleDict(
            {
                m: nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                for m in ("audio", "video")
            }
        )
        self.modality_gate = nn.Linear(self.hidden_dim * 3, 3)

    def forward(self, batch: Any) -> ModelOutput:
        pooled = self._pooled(batch)
        text = pooled["text"]
        audio_input = torch.cat((pooled["audio"], text), -1)
        video_input = torch.cat((pooled["video"], text), -1)
        audio = pooled["audio"] + torch.sigmoid(
            self.audio_gate(audio_input)
        ) * torch.tanh(self.refine["audio"](audio_input))
        video = pooled["video"] + torch.sigmoid(
            self.video_gate(video_input)
        ) * torch.tanh(self.refine["video"](video_input))
        weights = torch.softmax(
            self.modality_gate(torch.cat((audio, video, text), -1)), -1
        )
        fused = (torch.stack((audio, video, text), 1) * weights.unsqueeze(-1)).sum(
            1
        ) + (audio + video + text) / 3
        return self._output(fused, {"modality_gate": weights})


class CAREWSMAudioGuidedDualGatedFusion(_FusionBase):
    """Project-specific audio-conditioned comparison, not a published reproduction."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.modalities != ALL_MODALITIES:
            raise ValueError("audio-guided fusion requires audio, video, and text")
        self.video_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.text_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.refine = nn.ModuleDict(
            {
                m: nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                for m in ("video", "text")
            }
        )
        self.modality_gate = nn.Linear(self.hidden_dim * 3, 3)

    def forward(self, batch: Any) -> ModelOutput:
        pooled = self._pooled(batch)
        audio = pooled["audio"]
        video_input = torch.cat((pooled["video"], audio), -1)
        text_input = torch.cat((pooled["text"], audio), -1)
        video = pooled["video"] + torch.sigmoid(
            self.video_gate(video_input)
        ) * torch.tanh(self.refine["video"](video_input))
        text = pooled["text"] + torch.sigmoid(self.text_gate(text_input)) * torch.tanh(
            self.refine["text"](text_input)
        )
        weights = torch.softmax(
            self.modality_gate(torch.cat((audio, video, text), -1)), -1
        )
        fused = (torch.stack((audio, video, text), 1) * weights.unsqueeze(-1)).sum(
            1
        ) + (audio + video + text) / 3
        return self._output(fused, {"modality_gate": weights})


class CAREWSMTriModalCrossAttentionResidualFusion(_FusionBase):
    """Mask-safe cross-modal attention with a static pooled path."""

    def __init__(self, num_heads: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * len(self.modalities), self.hidden_dim),
            nn.GELU(),
        )

    def forward(self, batch: Any) -> ModelOutput:
        projected = self._project(batch)
        singleton = all(values.shape[1] == 1 for values in projected.values())
        pooled = self._pooled(batch)
        if singleton:
            fused = self.fusion(torch.cat([pooled[m] for m in self.modalities], -1))
            path = "static_fallback"
        else:
            tokens = torch.cat(list(projected.values()), 1)
            masks = torch.cat(
                [_mask_for(batch, m, batch.inputs[m]) for m in self.modalities], 1
            )
            attended, _ = self.attention(
                tokens, tokens, tokens, key_padding_mask=~masks
            )
            fused = self.fusion(
                torch.cat(
                    [
                        _masked_mean(
                            attended[:, start : start + projected[m].shape[1]],
                            _mask_for(batch, m, batch.inputs[m]),
                        )
                        for start, m in self._slices(projected)
                    ],
                    -1,
                )
            )
            path = "cross_attention"
        return self._output(fused, {"fusion_path": path})

    def _slices(self, projected):
        start = 0
        for modality in self.modalities:
            length = projected[modality].shape[1]
            yield start, modality
            start += length


class CAREWSMSharedPrivateTrimodalFusion(_FusionBase):
    """Shared/private component fusion for the selected modalities."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.shared = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in self.modalities]
        )
        self.private = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in self.modalities]
        )
        self.gate = nn.Linear(self.hidden_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU()
        )

    def forward(self, batch: Any) -> ModelOutput:
        pooled = self._pooled(batch)
        components = []
        for shared, private, modality in zip(
            self.shared, self.private, self.modalities
        ):
            components.extend((shared(pooled[modality]), private(pooled[modality])))
        stacked = torch.stack(components, 1)
        weights = torch.softmax(self.gate(torch.tanh(stacked)).squeeze(-1), 1)
        return self._output(
            self.fusion((stacked * weights.unsqueeze(-1)).sum(1) + stacked.mean(1)),
            {"component_gate": weights},
        )


@MODELS.register("carewsm_bimodal_gated_late")
def carewsm_bimodal_gated_late(context: Any | None = None, **params: Any):
    del context
    return CAREWSMBimodalTaskWiseGatedLateFusion(**params)


@MODELS.register("carewsm_bimodal_shared_mtl")
def carewsm_bimodal_shared_mtl(context: Any | None = None, **params: Any):
    del context
    return CAREWSMBimodalSharedMTLFusion(**params)


@MODELS.register("carewsm_bimodal_task_aware")
def carewsm_bimodal_task_aware(context: Any | None = None, **params: Any):
    del context
    return CAREWSMBimodalTaskAwareDirectedFusion(**params)


@MODELS.register("carewsm_text_guided_dual_gated")
def carewsm_text_guided_dual_gated(context: Any | None = None, **params: Any):
    del context
    return CAREWSMTextGuidedDualGatedFusion(**params)


@MODELS.register("carewsm_audio_guided_dual_gated")
def carewsm_audio_guided_dual_gated(context: Any | None = None, **params: Any):
    del context
    return CAREWSMAudioGuidedDualGatedFusion(**params)


@MODELS.register("carewsm_trimodal_cross_attention")
def carewsm_trimodal_cross_attention(context: Any | None = None, **params: Any):
    del context
    return CAREWSMTriModalCrossAttentionResidualFusion(**params)


@MODELS.register("carewsm_shared_private_trimodal")
def carewsm_shared_private_trimodal(context: Any | None = None, **params: Any):
    del context
    return CAREWSMSharedPrivateTrimodalFusion(**params)
