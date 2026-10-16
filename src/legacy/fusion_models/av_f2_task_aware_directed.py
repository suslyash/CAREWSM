"""Availability-aware task-specific directed relation-bank F2 model."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


class DirectedRelationExpert(nn.Module):
    """One ordered cross-modal relation expert shared across disease tasks."""

    def __init__(self, hidden_dim: int, relation_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.relation_mlp = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, relation_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.output_norm(query + self.relation_mlp(torch.cat([query, context], dim=1)))


class WSMAVTaskAwareDirectedF2Model(BaseModel):
    """F1 shared representation plus shared directed experts and task gates."""

    def __init__(self, audio_feature_dim: int = 768, video_feature_dim: int = 512,
                 hidden_dim: int = 192, fusion_hidden_dim: int = 192,
                 relation_hidden_dim: int = 192, dropout: float = 0.2,
                 num_tasks: int = 2) -> None:
        super().__init__()
        if min(audio_feature_dim, video_feature_dim, hidden_dim, fusion_hidden_dim, relation_hidden_dim) <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        if num_tasks != 2:
            raise ValueError("wsm_av_f2_task_aware_directed_model requires num_tasks=2")
        self.audio_feature_dim = int(audio_feature_dim)
        self.video_feature_dim = int(video_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.relation_hidden_dim = int(relation_hidden_dim)
        self.num_tasks = int(num_tasks)
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(self.audio_feature_dim), nn.Linear(self.audio_feature_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.video_projection = nn.Sequential(
            nn.LayerNorm(self.video_feature_dim), nn.Linear(self.video_feature_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.shared_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim), nn.Linear(2 * self.hidden_dim, self.fusion_hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(self.fusion_hidden_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.audio_to_video_expert = DirectedRelationExpert(self.hidden_dim, self.relation_hidden_dim, dropout)
        self.video_to_audio_expert = DirectedRelationExpert(self.hidden_dim, self.relation_hidden_dim, dropout)
        self.task_gates = nn.ModuleList([self._make_gate(dropout) for _ in range(self.num_tasks)])
        self.task_relation_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_tasks)])
        self.task_heads = nn.ModuleList([self._make_head(dropout) for _ in range(self.num_tasks)])

    def _make_gate(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim, 1),
        )

    def _make_head(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, batch: Any) -> ModelOutput:
        audio_cls = batch.inputs["audio_cls"]
        video = batch.inputs["video"]
        video_mask = batch.get_masks("video_mask")
        available = batch.get_masks("modality_available")
        if not isinstance(audio_cls, torch.Tensor) or audio_cls.ndim != 2 or tuple(audio_cls.shape[1:]) != (self.audio_feature_dim,):
            raise ValueError(f"audio_cls must have shape [B,{self.audio_feature_dim}]")
        if not torch.is_floating_point(audio_cls) or not bool(torch.isfinite(audio_cls).all()):
            raise ValueError("audio_cls must be finite floating point")
        if not isinstance(video, torch.Tensor) or video.ndim != 3 or video.shape[2] != self.video_feature_dim:
            raise ValueError(f"video must have shape [B,T,{self.video_feature_dim}]")
        if not torch.is_floating_point(video) or not bool(torch.isfinite(video).all()):
            raise ValueError("video must be finite floating point")
        if not isinstance(video_mask, torch.Tensor) or video_mask.ndim != 2 or tuple(video_mask.shape) != tuple(video.shape[:2]):
            raise ValueError("video_mask must have shape [B,T]")
        if not isinstance(available, torch.Tensor) or available.ndim != 2 or tuple(available.shape) != (audio_cls.shape[0], 2):
            raise ValueError("modality_available must have shape [B,2] ordered [audio,video]")
        video_mask = video_mask.to(device=video.device, dtype=torch.bool)
        available = available.to(device=video.device, dtype=torch.bool)
        if not bool(available.any(dim=1).all()):
            raise ValueError("every sample must have at least one available modality")
        if bool((available[:, 1] & ~video_mask.any(dim=1)).any()):
            raise ValueError("every available video sample must have at least one valid video position")

        audio_available = available[:, 0:1].to(device=audio_cls.device, dtype=audio_cls.dtype)
        video_available = available[:, 1:2].to(device=video.device, dtype=video.dtype)
        effective_audio_input = audio_cls * audio_available
        weights = video_mask.unsqueeze(-1).to(dtype=video.dtype)
        video_raw = (video * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        video_raw = video_raw * video_available
        features_audio = self.audio_projection(effective_audio_input)
        features_video = self.video_projection(video_raw)
        effective_audio_features = features_audio * audio_available
        effective_video_features = features_video * video_available
        features_shared = self.shared_fusion(torch.cat([effective_audio_features, effective_video_features], dim=1))

        both_available = available[:, 0] & available[:, 1]
        relation_valid = both_available.unsqueeze(1).expand(-1, 2)
        audio_to_video = self.audio_to_video_expert(effective_audio_features, effective_video_features)
        video_to_audio = self.video_to_audio_expert(effective_video_features, effective_audio_features)
        relation_experts = torch.stack([audio_to_video, video_to_audio], dim=1)
        relation_experts = relation_experts * relation_valid.unsqueeze(-1).to(dtype=relation_experts.dtype)

        task_weights = []
        task_relation_features = []
        task_features = []
        for gate, norm in zip(self.task_gates, self.task_relation_norms):
            scores = torch.cat([gate(relation_experts[:, 0]), gate(relation_experts[:, 1])], dim=1)
            current_weights = torch.softmax(scores, dim=-1) * relation_valid.to(dtype=scores.dtype)
            relation = (current_weights.unsqueeze(-1) * relation_experts).sum(dim=1)
            task_weights.append(current_weights)
            task_relation_features.append(relation)
            task_features.append(norm(features_shared + relation))
        task_expert_weights = torch.stack(task_weights, dim=1)
        task_relation_features = torch.stack(task_relation_features, dim=1)
        task_features = torch.stack(task_features, dim=1)
        preds = torch.cat([head(task_features[:, i]) for i, head in enumerate(self.task_heads)], dim=1)
        return ModelOutput(preds=preds, aux={
            "features_audio": features_audio,
            "features_video": features_video,
            "effective_audio_features": effective_audio_features,
            "effective_video_features": effective_video_features,
            "features_shared": features_shared,
            "relation_experts": relation_experts,
            "relation_valid": relation_valid,
            "task_expert_weights": task_expert_weights,
            "task_relation_features": task_relation_features,
            "task_features": task_features,
            "task_logits": {"depression": preds[:, 0], "parkinson": preds[:, 1]},
        })


@MODELS.register("wsm_av_f2_task_aware_directed_model")
def wsm_av_f2_task_aware_directed_model(context: Any | None = None, **params: Any) -> WSMAVTaskAwareDirectedF2Model:
    params = dict(params)
    if context is not None:
        audio_dim = context.get("data.audio_feature_dim")
        video_dim = context.get("data.video_feature_dim")
        context_tasks = context.get("data.num_tasks")
        if audio_dim is not None:
            params.setdefault("audio_feature_dim", audio_dim)
        if video_dim is not None:
            params.setdefault("video_feature_dim", video_dim)
        if context_tasks is not None and context_tasks != 2:
            raise ValueError("wsm_av_f2_task_aware_directed_model requires data.num_tasks=2")
        params.setdefault("num_tasks", 2)
    params.setdefault("audio_feature_dim", 768)
    params.setdefault("video_feature_dim", 512)
    params.setdefault("num_tasks", 2)
    return WSMAVTaskAwareDirectedF2Model(**params)
