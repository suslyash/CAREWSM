"""Audio-query temporal-video zero-initialized residual fusion candidate B."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel
from .frozen_audio_temporal_adapter import FrozenAudioTemporalAdapter


class WSMAudioQueryTemporalVideoModel(BaseModel):
    def __init__(self, audio_checkpoint_path: str, audio_feature_dim: int = 768,
                 audio_hidden_dim: int = 192, video_feature_dim: int = 512,
                 hidden_dim: int = 128, num_heads: int = 4,
                 residual_hidden_dim: int = 128, dropout: float = 0.2,
                 num_tasks: int = 2) -> None:
        super().__init__()
        if (audio_feature_dim, audio_hidden_dim, video_feature_dim, num_tasks) != (768, 192, 512, 2):
            raise ValueError("audio-query candidate B requires audio=768/192, video=512, tasks=2")
        if hidden_dim > 256 or hidden_dim % num_heads:
            raise ValueError("candidate B hidden_dim must be <=256 and divisible by num_heads")
        self.audio_adapter = FrozenAudioTemporalAdapter(audio_checkpoint_path)
        self.video_projection = nn.Sequential(
            nn.LayerNorm(video_feature_dim), nn.Linear(video_feature_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.query_projection = nn.Sequential(
            nn.LayerNorm(audio_hidden_dim), nn.Linear(audio_hidden_dim, hidden_dim),
        )
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.residual_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, residual_hidden_dim),
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
        keys = self.video_projection(video)
        safe_mask = video_mask.clone()
        safe_mask[~available[:, 1]] = False
        if safe_mask.shape[1] == 0:
            raise ValueError("video sequence must have at least one position")
        # A missing-video row is masked to a zero key/value position; its correction is
        # then explicitly zeroed below, preserving the frozen audio fallback exactly.
        if safe_mask.shape[1] > 0:
            safe_mask[~available[:, 1], 0] = True
        keys = keys * safe_mask.unsqueeze(-1).to(keys.dtype)
        padding = ~safe_mask
        contexts = []
        queries = []
        for task in range(2):
            query = self.query_projection(audio_out["task_features"][:, task]).unsqueeze(1)
            context, _ = self.temporal_attention(query, keys, keys, key_padding_mask=padding)
            queries.append(query[:, 0])
            contexts.append(context[:, 0])
        queries = torch.stack(queries, dim=1)
        contexts = torch.stack(contexts, dim=1)
        residuals = [head(torch.cat((queries[:, task], contexts[:, task]), 1))
                     for task, head in enumerate(self.residual_heads)]
        residual_logits = torch.cat(residuals, dim=1)
        residual_logits = residual_logits * available[:, 1:2].to(residual_logits.dtype)
        preds = audio_out["base_logits"] + residual_logits
        return ModelOutput(preds=preds, aux={
            "audio_base_logits": audio_out["base_logits"],
            "audio_task_features": audio_out["task_features"],
            "video_sequence": keys,
            "audio_queries": queries,
            "temporal_video_context": contexts,
            "residual_logits": residual_logits,
            "task_logits": {"depression": preds[:, 0], "parkinson": preds[:, 1]},
        })


@MODELS.register("wsm_av_audio_query_temporal_video_model")
def wsm_av_audio_query_temporal_video_model(context: Any | None = None, **params: Any) -> WSMAudioQueryTemporalVideoModel:
    params = dict(params)
    params.setdefault("audio_feature_dim", 768)
    params.setdefault("audio_hidden_dim", 192)
    params.setdefault("video_feature_dim", 512)
    params.setdefault("num_tasks", 2)
    if "audio_checkpoint_path" not in params:
        raise ValueError("audio_checkpoint_path is required")
    return WSMAudioQueryTemporalVideoModel(**params)
