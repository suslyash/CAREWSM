"""Strong temporal-audio F1 residual A+V fusion contract."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel
from .frozen_audio_temporal_adapter import FrozenAudioTemporalAdapter


class WSMAVF1TemporalAudioResidualModel(BaseModel):
    """Frozen task-conditioned audio logits plus shared video residual fusion."""

    def __init__(self, audio_checkpoint_path: str, audio_feature_dim: int = 768,
                 audio_hidden_dim: int = 192, video_feature_dim: int = 512,
                 hidden_dim: int = 192, fusion_hidden_dim: int = 192,
                 dropout: float = 0.2, num_tasks: int = 2) -> None:
        super().__init__()
        if audio_feature_dim != 768 or audio_hidden_dim != 192 or video_feature_dim != 512:
            raise ValueError("strong temporal-audio F1 requires audio=768, audio_hidden=192, video=512")
        if hidden_dim != 192 or fusion_hidden_dim != 192 or num_tasks != 2:
            raise ValueError("strong temporal-audio F1 requires hidden=192, fusion_hidden=192, num_tasks=2")
        self.audio_feature_dim = int(audio_feature_dim)
        self.audio_hidden_dim = int(audio_hidden_dim)
        self.video_feature_dim = int(video_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.num_tasks = int(num_tasks)
        self.audio_adapter = FrozenAudioTemporalAdapter(audio_checkpoint_path)
        self.video_projection = nn.Sequential(
            nn.LayerNorm(self.video_feature_dim), nn.Linear(self.video_feature_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.shared_fusion = nn.Sequential(
            nn.LayerNorm(self.audio_hidden_dim + self.hidden_dim),
            nn.Linear(self.audio_hidden_dim + self.hidden_dim, self.fusion_hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(self.fusion_hidden_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.residual_heads = nn.ModuleList([self._make_head(dropout) for _ in range(self.num_tasks)])

    def _make_head(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, batch: Any) -> ModelOutput:
        audio = batch.inputs["audio"]
        video = batch.inputs["video"]
        audio_mask = batch.get_masks("audio_mask")
        video_mask = batch.get_masks("video_mask")
        available = batch.get_masks("modality_available")
        if not isinstance(available, torch.Tensor) or available.ndim != 2 or tuple(available.shape) != (audio.shape[0], 2):
            raise ValueError("modality_available must have shape [B,2] ordered [audio,video]")
        available = available.to(device=audio.device, dtype=torch.bool)
        if not bool(available[:, 0].all()):
            raise ValueError("strong temporal-audio F1 requires audio availability")
        if not isinstance(video, torch.Tensor) or video.ndim != 3 or video.shape[2] != self.video_feature_dim:
            raise ValueError(f"video must have shape [B,T,{self.video_feature_dim}]")
        if not torch.is_floating_point(video) or not bool(torch.isfinite(video).all()):
            raise ValueError("video must be finite floating point")
        if not isinstance(video_mask, torch.Tensor) or video_mask.ndim != 2 or tuple(video_mask.shape) != tuple(video.shape[:2]):
            raise ValueError("video_mask must have shape [B,T]")
        video_mask = video_mask.to(device=video.device, dtype=torch.bool)
        if bool((available[:, 1] & ~video_mask.any(dim=1)).any()):
            raise ValueError("every available video sample must have at least one valid video position")
        audio_out = self.audio_adapter(batch)
        weights = video_mask.unsqueeze(-1).to(dtype=video.dtype)
        video_raw = (video * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        video_features = self.video_projection(video_raw)
        effective_video_features = video_features * available[:, 1:2].to(dtype=video_features.dtype)
        task_fused = []
        residuals = []
        for task in range(self.num_tasks):
            joined = torch.cat([audio_out["task_features"][:, task], effective_video_features], dim=1)
            fused = self.shared_fusion(joined)
            task_fused.append(fused)
            residuals.append(self.residual_heads[task](fused))
        task_fused_features = torch.stack(task_fused, dim=1)
        residual_logits = torch.cat(residuals, dim=1)
        preds = audio_out["base_logits"] + residual_logits * available[:, 1:2].to(dtype=residual_logits.dtype)
        return ModelOutput(preds=preds, aux={
            "audio_base_logits": audio_out["base_logits"],
            "legacy_audio_class_logits": audio_out["legacy_class_logits"],
            "audio_task_features": audio_out["task_features"],
            "video_features": video_features,
            "effective_video_features": effective_video_features,
            "task_fused_features": task_fused_features,
            "residual_logits": residual_logits,
            "task_logits": {"depression": preds[:, 0], "parkinson": preds[:, 1]},
        })


@MODELS.register("wsm_av_f1_temporal_audio_residual_model")
def wsm_av_f1_temporal_audio_residual_model(context: Any | None = None, **params: Any) -> WSMAVF1TemporalAudioResidualModel:
    params = dict(params)
    if context is not None:
        audio_dim = context.get("data.audio_feature_dim")
        video_dim = context.get("data.video_feature_dim")
        tasks = context.get("data.num_tasks")
        if audio_dim is not None and audio_dim != 768:
            raise ValueError("strong temporal-audio F1 requires data.audio_feature_dim=768")
        if video_dim is not None and video_dim != 512:
            raise ValueError("strong temporal-audio F1 requires data.video_feature_dim=512")
        if tasks is not None and tasks != 2:
            raise ValueError("strong temporal-audio F1 requires data.num_tasks=2")
    params.setdefault("audio_feature_dim", 768)
    params.setdefault("audio_hidden_dim", 192)
    params.setdefault("video_feature_dim", 512)
    params.setdefault("hidden_dim", 192)
    params.setdefault("fusion_hidden_dim", 192)
    params.setdefault("num_tasks", 2)
    if "audio_checkpoint_path" not in params:
        raise ValueError("audio_checkpoint_path is required for strong temporal-audio F1")
    return WSMAVF1TemporalAudioResidualModel(**params)
