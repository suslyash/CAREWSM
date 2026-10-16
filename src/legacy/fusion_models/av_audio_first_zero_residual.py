"""Audio-anchored zero-initialized residual fusion candidate A."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel
from .frozen_audio_temporal_adapter import FrozenAudioTemporalAdapter


class WSMAudioFirstZeroResidualModel(BaseModel):
    def __init__(self, audio_checkpoint_path: str, audio_feature_dim: int = 768,
                 audio_hidden_dim: int = 192, video_feature_dim: int = 512,
                 video_hidden_dim: int = 128, residual_hidden_dim: int = 128,
                 dropout: float = 0.2, num_tasks: int = 2) -> None:
        super().__init__()
        if (audio_feature_dim, audio_hidden_dim, video_feature_dim, num_tasks) != (768, 192, 512, 2):
            raise ValueError("audio-first candidate A requires audio=768/192, video=512, tasks=2")
        self.audio_adapter = FrozenAudioTemporalAdapter(audio_checkpoint_path)
        self.video_projection = nn.Sequential(
            nn.LayerNorm(video_feature_dim), nn.Linear(video_feature_dim, video_hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.residual_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(audio_hidden_dim + video_hidden_dim),
                nn.Linear(audio_hidden_dim + video_hidden_dim, residual_hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(residual_hidden_dim, 1),
            ) for _ in range(num_tasks)
        ])
        for head in self.residual_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, batch: Any) -> ModelOutput:
        video = batch.inputs["video"]
        video_mask = batch.get_masks("video_mask")
        available = batch.get_masks("modality_available")
        if not isinstance(video, torch.Tensor) or video.ndim != 3 or video.shape[-1] != 512:
            raise ValueError("video must have shape [B,T,512]")
        if not isinstance(video_mask, torch.Tensor) or tuple(video_mask.shape) != tuple(video.shape[:2]):
            raise ValueError("video_mask must have shape [B,T]")
        if not isinstance(available, torch.Tensor) or tuple(available.shape) != (video.shape[0], 2):
            raise ValueError("modality_available must have shape [B,2]")
        video_mask = video_mask.to(device=video.device, dtype=torch.bool)
        available = available.to(device=video.device, dtype=torch.bool)
        if not bool(available[:, 0].all()):
            raise ValueError("audio availability is required")
        if bool((available[:, 1] & ~video_mask.any(dim=1)).any()):
            raise ValueError("available video requires a valid video frame")
        audio_out = self.audio_adapter(batch)
        weights = video_mask.unsqueeze(-1).to(video.dtype)
        pooled = (video * weights).sum(1) / weights.sum(1).clamp_min(1)
        video_features = self.video_projection(pooled)
        video_features = video_features * available[:, 1:2].to(video.dtype)
        residuals = []
        for task, head in enumerate(self.residual_heads):
            joined = torch.cat((audio_out["task_features"][:, task], video_features), dim=1)
            residuals.append(head(joined))
        residual_logits = torch.cat(residuals, dim=1)
        residual_logits = residual_logits * available[:, 1:2].to(residual_logits.dtype)
        preds = audio_out["base_logits"] + residual_logits
        return ModelOutput(preds=preds, aux={
            "audio_base_logits": audio_out["base_logits"],
            "audio_task_features": audio_out["task_features"],
            "video_features": video_features,
            "residual_logits": residual_logits,
            "task_logits": {"depression": preds[:, 0], "parkinson": preds[:, 1]},
        })


@MODELS.register("wsm_av_audio_first_zero_residual_model")
def wsm_av_audio_first_zero_residual_model(context: Any | None = None, **params: Any) -> WSMAudioFirstZeroResidualModel:
    params = dict(params)
    params.setdefault("audio_feature_dim", 768)
    params.setdefault("audio_hidden_dim", 192)
    params.setdefault("video_feature_dim", 512)
    params.setdefault("num_tasks", 2)
    if "audio_checkpoint_path" not in params:
        raise ValueError("audio_checkpoint_path is required")
    return WSMAudioFirstZeroResidualModel(**params)
