from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel
from fusion.models.av_sync_mamba_segment import (
    TemporalEncoder,
    _autocast_enabled,
    _head,
    _projection,
    masked_mean,
    resample_valid_sequence,
)


class AudioMambaSegmentModel(BaseModel):
    def __init__(
        self,
        audio_feature_dim: int,
        num_tasks: int = 2,
        num_classes: int = 2,
        hidden_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.2,
        encoder_type: str = "mamba",
        sequence_steps: int = 128,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_required: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_steps = int(sequence_steps)
        self.audio_projection = _projection(int(audio_feature_dim), hidden_dim, dropout)
        self.audio_encoder = TemporalEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            encoder_type=encoder_type,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_required=mamba_required,
        )
        self.segment_encoder = TemporalEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            encoder_type=encoder_type,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_required=mamba_required,
        )

        self.projected_audio_norm = nn.LayerNorm(hidden_dim)
        self.audio_output_norm = nn.LayerNorm(hidden_dim)
        self.segment_input_norm = nn.LayerNorm(hidden_dim)
        self.segment_output_norm = nn.LayerNorm(hidden_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.task_embeddings = nn.Embedding(num_tasks, hidden_dim)
        self.position_embeddings = nn.Parameter(torch.empty(1, sequence_steps, hidden_dim))
        self.heads = nn.ModuleList([_head(hidden_dim, hidden_dim, num_classes, dropout) for _ in range(num_tasks)])
        self.aux_heads = nn.ModuleList([_head(hidden_dim, hidden_dim, num_classes, dropout) for _ in range(num_tasks)])
        self._reset_parameters()

    def forward(self, batch: Batch) -> ModelOutput:
        if self.audio_encoder.encoder_type == "transformer" and _autocast_enabled(batch.inputs["audio"]):
            with torch.amp.autocast(device_type=batch.inputs["audio"].device.type, enabled=False):
                return self.forward(batch)

        task_ids = batch.inputs["task_ids"].long()
        audio_mask = batch.get_masks("audio_mask").to(device=task_ids.device, dtype=torch.bool)

        audio = self.audio_projection(batch.inputs["audio"])
        audio = self.projected_audio_norm(audio)
        audio = self.audio_encoder(audio, audio_mask)
        audio = self.audio_output_norm(audio)
        audio, audio_mask = resample_valid_sequence(audio, audio_mask, self.sequence_steps)
        audio = audio + self.position_embeddings[:, : self.sequence_steps]
        audio = self.segment_input_norm(audio)

        task_embedding = self.task_embeddings(task_ids)
        cls = self.cls_token.expand(audio.shape[0], -1, -1) + task_embedding.unsqueeze(1)
        tokens = torch.cat([cls, audio], dim=1)
        token_mask = torch.cat(
            [torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device), audio_mask],
            dim=1,
        )
        encoded = self.segment_encoder(tokens, token_mask)
        encoded = self.segment_output_norm(encoded)
        segment_features = encoded[:, 0]
        pooled_features = masked_mean(audio, audio_mask)

        all_logits = torch.stack([head(segment_features) for head in self.heads], dim=1)
        aux_logits = torch.stack([head(pooled_features) for head in self.aux_heads], dim=1)
        index = torch.arange(all_logits.shape[0], device=all_logits.device)
        logits = all_logits[index, task_ids]

        return ModelOutput(
            preds=logits,
            aux={
                "logits": logits,
                "aux_logits": aux_logits[index, task_ids],
                "features": segment_features,
                "task_ids": task_ids,
            },
        )

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.task_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.02)


@MODELS.register("audio_mamba_segment_model")
def audio_mamba_segment_model(context: Any | None = None, **params: Any) -> AudioMambaSegmentModel:
    params = dict(params)
    if context is not None:
        params.setdefault("audio_feature_dim", context.get("data.audio_feature_dim"))
        params.setdefault("num_tasks", context.get("data.num_tasks", 2))
        params.setdefault("num_classes", context.get("data.num_classes", 2))

    return AudioMambaSegmentModel(**params)
