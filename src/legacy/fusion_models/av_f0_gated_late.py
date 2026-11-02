"""Availability-aware task-wise gated late-fusion F0 model."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


class WSMAVGatedLateFusionF0Model(BaseModel):
    """Simple sparse two-task late fusion over frozen audio/video features."""

    def __init__(self, audio_feature_dim: int = 768, video_feature_dim: int = 512,
                 hidden_dim: int = 192, gate_hidden_dim: int = 64,
                 dropout: float = 0.2, num_tasks: int = 2) -> None:
        super().__init__()
        if audio_feature_dim <= 0 or video_feature_dim <= 0 or hidden_dim <= 0 or gate_hidden_dim <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        if num_tasks != 2:
            raise ValueError("wsm_av_f0_gated_late_model requires num_tasks=2")
        self.audio_feature_dim = int(audio_feature_dim)
        self.video_feature_dim = int(video_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tasks = int(num_tasks)
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(self.audio_feature_dim), nn.Linear(self.audio_feature_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.video_projection = nn.Sequential(
            nn.LayerNorm(self.video_feature_dim), nn.Linear(self.video_feature_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.audio_heads = nn.ModuleList([self._make_head(dropout) for _ in range(self.num_tasks)])
        self.video_heads = nn.ModuleList([self._make_head(dropout) for _ in range(self.num_tasks)])
        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(2 * self.hidden_dim), nn.Linear(2 * self.hidden_dim, gate_hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(gate_hidden_dim, 1), nn.Sigmoid(),
            ) for _ in range(self.num_tasks)
        ])

    def _make_head(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Dropout(dropout), nn.Linear(self.hidden_dim, self.hidden_dim),
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

        weights = video_mask.unsqueeze(-1).to(dtype=video.dtype)
        video_raw = (video * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        audio_features = self.audio_projection(audio_cls)
        video_features = self.video_projection(video_raw)
        audio_logits = torch.cat([head(audio_features) for head in self.audio_heads], dim=1)
        video_logits = torch.cat([head(video_features) for head in self.video_heads], dim=1)
        joined = torch.cat([audio_features, video_features], dim=1)
        raw_audio_gates = torch.cat([head(joined) for head in self.gate_heads], dim=1)
        audio_available = available[:, 0:1]
        video_available = available[:, 1:2]
        audio_weight = torch.where(audio_available & ~video_available, torch.ones_like(raw_audio_gates), raw_audio_gates)
        audio_weight = torch.where(~audio_available & video_available, torch.zeros_like(audio_weight), audio_weight)
        video_weight = torch.where(audio_available & ~video_available, torch.zeros_like(raw_audio_gates), 1.0 - raw_audio_gates)
        video_weight = torch.where(~audio_available & video_available, torch.ones_like(video_weight), video_weight)
        fusion_weights = torch.stack([audio_weight, video_weight], dim=-1)
        logits = audio_weight * audio_logits + video_weight * video_logits
        return ModelOutput(preds=logits, aux={
            "features_audio": audio_features, "features_video": video_features,
            "audio_logits": audio_logits, "video_logits": video_logits,
            "raw_audio_gates": raw_audio_gates, "fusion_weights": fusion_weights,
            "task_logits": {"depression": logits[:, 0], "parkinson": logits[:, 1]},
        })


@MODELS.register("wsm_av_f0_gated_late_model")
def wsm_av_f0_gated_late_model(context: Any | None = None, **params: Any) -> WSMAVGatedLateFusionF0Model:
    params = dict(params)
    if context is not None:
        audio_dim = context.get("data.audio_feature_dim")
        video_dim = context.get("data.video_feature_dim")
        context_tasks = context.get("data.num_tasks")
        if audio_dim is not None: params.setdefault("audio_feature_dim", audio_dim)
        if video_dim is not None: params.setdefault("video_feature_dim", video_dim)
        if context_tasks is not None and context_tasks != 2:
            raise ValueError("wsm_av_f0_gated_late_model requires data.num_tasks=2")
        params.setdefault("num_tasks", 2)
    params.setdefault("audio_feature_dim", 768)
    params.setdefault("video_feature_dim", 512)
    params.setdefault("num_tasks", 2)
    return WSMAVGatedLateFusionF0Model(**params)
