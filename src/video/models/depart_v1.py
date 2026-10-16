"""Registered V1 DEPART-like temporal video model."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel

class WSMVideoDepartV1Model(BaseModel):
    def __init__(self, video_feature_dim: int = 512, hidden_dim: int = 192,
                 num_layers: int = 2, num_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.2, sequence_steps: int = 60, num_tasks: int = 2) -> None:
        super().__init__()
        if video_feature_dim <= 0 or hidden_dim <= 0 or sequence_steps <= 0:
            raise ValueError("video_feature_dim, hidden_dim, and sequence_steps must be positive")
        if num_tasks != 2:
            raise ValueError("wsm_video_depart_v1_model requires num_tasks=2")
        self.video_feature_dim = int(video_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.sequence_steps = int(sequence_steps)
        self.num_tasks = int(num_tasks)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.video_feature_dim),
            nn.Linear(self.video_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.sequence_steps, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=self.hidden_dim * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, 1),
            )
            for _ in range(self.num_tasks)
        ])
        nn.init.normal_(self.position_embeddings, std=0.02)

    def forward(self, batch: Any) -> ModelOutput:
        video = batch.inputs["video"]
        mask = batch.get_masks("video_mask")
        if not isinstance(video, torch.Tensor) or video.ndim != 3:
            raise ValueError("video must be a float tensor with shape [B, T, D]")
        if not torch.is_floating_point(video):
            raise ValueError("video must be a floating-point tensor")
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
            raise ValueError("video_mask must be a tensor with shape [B, T]")
        if mask.shape[0] != video.shape[0] or mask.shape[1] != video.shape[1]:
            raise ValueError("video and video_mask batch/time dimensions must match")
        if video.shape[2] != self.video_feature_dim:
            raise ValueError(f"video feature dimension must be {self.video_feature_dim}, got {video.shape[2]}")
        if video.shape[1] > self.sequence_steps:
            raise ValueError(f"video sequence length must be <= {self.sequence_steps}, got {video.shape[1]}")
        mask = mask.to(device=video.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every video sample must have at least one valid frame")
        encoded = self.input_projection(video) + self.position_embeddings[:, :video.shape[1]]
        encoded = self.temporal_encoder(encoded, src_key_padding_mask=~mask)
        encoded = self.output_norm(encoded)
        weights = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        logits = torch.cat([head(pooled) for head in self.heads], dim=1)
        return ModelOutput(
            preds=logits,
            aux={
                "features": pooled,
                "task_logits": {"depression": logits[:, 0], "parkinson": logits[:, 1]},
            },
        )

@MODELS.register("wsm_video_depart_v1_model")
def wsm_video_depart_v1_model(context: Any | None = None, **params: Any) -> WSMVideoDepartV1Model:
    params = dict(params)
    if context is not None:
        feature_dim = context.get("data.video_feature_dim")
        if feature_dim is not None:
            params.setdefault("video_feature_dim", feature_dim)
        context_tasks = context.get("data.num_tasks")
        if context_tasks is not None and context_tasks != 2:
            raise ValueError("wsm_video_depart_v1_model requires data.num_tasks=2")
        params.setdefault("num_tasks", 2)
    params.setdefault("video_feature_dim", 512)
    params.setdefault("num_tasks", 2)
    return WSMVideoDepartV1Model(**params)
