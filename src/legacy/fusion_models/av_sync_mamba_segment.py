from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:  # pragma: no cover
    Mamba = None


class AVSyncMambaSegmentModel(BaseModel):
    def __init__(
        self,
        audio_feature_dim: int,
        video_feature_dim: int,
        num_tasks: int = 2,
        num_classes: int = 2,
        hidden_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.2,
        encoder_type: str = "mamba",
        sync_steps: int = 96,
        lag_radius: int = 6,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_required: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_tasks = int(num_tasks)
        self.num_classes = int(num_classes)
        self.sync_steps = int(sync_steps)

        self.audio_projection = _projection(int(audio_feature_dim), hidden_dim, dropout)
        self.video_projection = _projection(int(video_feature_dim), hidden_dim, dropout)

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
        self.video_encoder = TemporalEncoder(
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
        self.synchrony = LaggedSynchronyBlock(
            hidden_dim=hidden_dim,
            lag_radius=lag_radius,
            dropout=dropout,
        )
        self.fusion_encoder = TemporalEncoder(
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

        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.task_embeddings = nn.Embedding(num_tasks, hidden_dim)
        self.type_embeddings = nn.Parameter(torch.empty(4, hidden_dim))
        self.position_embeddings = nn.Parameter(torch.empty(1, sync_steps, hidden_dim))
        self.sync_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([_head(hidden_dim, hidden_dim, num_classes, dropout) for _ in range(num_tasks)])
        self.audio_heads = nn.ModuleList([_head(hidden_dim, hidden_dim, num_classes, dropout) for _ in range(num_tasks)])
        self.video_heads = nn.ModuleList([_head(hidden_dim, hidden_dim, num_classes, dropout) for _ in range(num_tasks)])
        self.audio_contrast = nn.Linear(hidden_dim, hidden_dim)
        self.video_contrast = nn.Linear(hidden_dim, hidden_dim)

        self._reset_parameters()

    def forward(self, batch: Batch) -> ModelOutput:
        if self.audio_encoder.encoder_type == "transformer" and _autocast_enabled(batch.inputs["audio"]):
            with torch.amp.autocast(device_type=batch.inputs["audio"].device.type, enabled=False):
                return self.forward(batch)

        task_ids = batch.inputs["task_ids"].long()
        audio_mask = batch.get_masks("audio_mask").to(device=task_ids.device, dtype=torch.bool)
        video_mask = batch.get_masks("video_mask").to(device=task_ids.device, dtype=torch.bool)

        audio = self.audio_projection(batch.inputs["audio"])
        video = self.video_projection(batch.inputs["video"])
        audio = self.audio_encoder(audio, audio_mask)
        video = self.video_encoder(video, video_mask)

        audio_sync, audio_sync_mask = resample_valid_sequence(audio, audio_mask, self.sync_steps)
        video_sync, video_sync_mask = resample_valid_sequence(video, video_mask, self.sync_steps)
        sync_embedding, lag_probs = self.synchrony(audio_sync, video_sync, audio_sync_mask, video_sync_mask)

        task_embedding = self.task_embeddings(task_ids)
        cls = self.cls_token.expand(audio.shape[0], -1, -1) + task_embedding.unsqueeze(1)
        sync_token = self.sync_projection(sync_embedding).unsqueeze(1)
        sync_token = sync_token + self.type_embeddings[1].view(1, 1, -1) + task_embedding.unsqueeze(1)

        positions = self.position_embeddings[:, : self.sync_steps]
        audio_tokens = audio_sync + positions + self.type_embeddings[2].view(1, 1, -1)
        video_tokens = video_sync + positions + self.type_embeddings[3].view(1, 1, -1)
        interleaved = torch.stack([audio_tokens, video_tokens], dim=2).flatten(1, 2)
        interleaved_mask = torch.stack([audio_sync_mask, video_sync_mask], dim=2).flatten(1, 2)

        tokens = torch.cat([cls + self.type_embeddings[0].view(1, 1, -1), sync_token, interleaved], dim=1)
        token_mask = torch.cat(
            [
                torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device),
                (audio_sync_mask.any(dim=1) & video_sync_mask.any(dim=1)).unsqueeze(1),
                interleaved_mask,
            ],
            dim=1,
        )
        fused = self.fusion_encoder(tokens, token_mask)
        segment_features = fused[:, 0]

        audio_features = masked_mean(audio, audio_mask)
        video_features = masked_mean(video, video_mask)
        all_logits = torch.stack([head(segment_features) for head in self.heads], dim=1)
        audio_logits = torch.stack([head(audio_features) for head in self.audio_heads], dim=1)
        video_logits = torch.stack([head(video_features) for head in self.video_heads], dim=1)
        index = torch.arange(all_logits.shape[0], device=all_logits.device)
        logits = all_logits[index, task_ids]

        return ModelOutput(
            preds=logits,
            aux={
                "logits": logits,
                "task_logits": {
                    "depression": all_logits[:, 0],
                    "parkinson": all_logits[:, 1],
                },
                "audio_logits": audio_logits[index, task_ids],
                "video_logits": video_logits[index, task_ids],
                "features": segment_features,
                "audio_embedding": self.audio_contrast(audio_features),
                "video_embedding": self.video_contrast(video_features),
                "sync_embedding": sync_embedding,
                "lag_probs": lag_probs,
                "task_ids": task_ids,
            },
        )

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.task_embeddings.weight, std=0.02)
        nn.init.normal_(self.type_embeddings, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.02)


class LaggedSynchronyBlock(nn.Module):
    def __init__(self, hidden_dim: int, lag_radius: int, dropout: float) -> None:
        super().__init__()
        self.lags = tuple(range(-int(lag_radius), int(lag_radius) + 1))
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lag_embeddings: list[torch.Tensor] = []
        lag_scores: list[torch.Tensor] = []
        for lag in self.lags:
            shifted_video, shifted_mask = shift_sequence(video, video_mask, lag)
            pair_mask = audio_mask & shifted_mask
            pair = torch.cat([audio, shifted_video, audio - shifted_video, audio * shifted_video], dim=-1)
            pair = self.pair_encoder(pair)
            pooled = masked_mean(pair, pair_mask)
            lag_embeddings.append(pooled)
            lag_scores.append(self.score(pooled).squeeze(-1))

        embeddings = torch.stack(lag_embeddings, dim=1)
        scores = torch.stack(lag_scores, dim=1)
        lag_probs = torch.softmax(scores, dim=-1)
        sync_embedding = torch.sum(embeddings * lag_probs.unsqueeze(-1), dim=1)
        return sync_embedding, lag_probs


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        encoder_type: str,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_required: bool,
    ) -> None:
        super().__init__()
        encoder_type = str(encoder_type).lower()
        if encoder_type == "mamba" and Mamba is None:
            if mamba_required:
                raise ModuleNotFoundError(
                    "encoder_type='mamba' requires the 'mamba_ssm' package. "
                    "Install project dependencies or set encoder_type='transformer'."
                )
            encoder_type = "transformer"

        self.encoder_type = encoder_type
        if encoder_type == "mamba":
            self.layers = nn.ModuleList(
                [
                    BiMambaBlock(
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        d_state=mamba_d_state,
                        d_conv=mamba_d_conv,
                        expand=mamba_expand,
                    )
                    for _ in range(int(num_layers))
                ]
            )
            return

        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * int(ff_mult),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(num_layers),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=x.device, dtype=torch.bool)
        if self.encoder_type == "mamba":
            for layer in self.layers:
                x = layer(x, mask)
            return x * mask.unsqueeze(-1).to(dtype=x.dtype)

        safe_mask = safe_valid_mask(mask)
        if _autocast_enabled(x):
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                x = self.layers(x.float(), src_key_padding_mask=~safe_mask)
            return x * mask.unsqueeze(-1).to(dtype=x.dtype)

        x = self.layers(x, src_key_padding_mask=~safe_mask)
        return x * mask.unsqueeze(-1).to(dtype=x.dtype)


class BiMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        assert Mamba is not None
        self.norm = nn.LayerNorm(hidden_dim)
        self.forward_mamba = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        y = self.norm(x) * mask_f
        y_forward = self.forward_mamba(y.contiguous())
        y_backward = reverse_valid_sequence(y, mask).contiguous()
        y_backward = self.backward_mamba(y_backward)
        y_backward = reverse_valid_sequence(y_backward, mask)
        y = self.out_proj(torch.cat([y_forward, y_backward], dim=-1))
        x = (residual + self.dropout(y)) * mask_f
        x = (x + self.ffn(x)) * mask_f
        return x


def _projection(input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Dropout(dropout),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=x.device, dtype=torch.bool)
    denom = mask.sum(dim=1).clamp_min(1).to(dtype=x.dtype)
    return (x * mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom.unsqueeze(-1)


def resample_valid_sequence(x: torch.Tensor, mask: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = x.new_zeros(x.shape[0], int(steps), x.shape[-1])
    out_mask = torch.zeros(x.shape[0], int(steps), dtype=torch.bool, device=x.device)
    for index in range(x.shape[0]):
        valid = x[index, mask[index]]
        if valid.shape[0] == 0:
            continue
        if valid.shape[0] == 1:
            out[index] = valid.expand(int(steps), -1)
        else:
            out[index] = F.interpolate(
                valid.transpose(0, 1).unsqueeze(0),
                size=int(steps),
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
        out_mask[index] = True

    return out, out_mask


def shift_sequence(x: torch.Tensor, mask: torch.Tensor, lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    if lag == 0:
        return x, mask

    shifted = torch.zeros_like(x)
    shifted_mask = torch.zeros_like(mask)
    if lag > 0:
        shifted[:, lag:] = x[:, :-lag]
        shifted_mask[:, lag:] = mask[:, :-lag]
    else:
        lag_abs = abs(lag)
        shifted[:, :-lag_abs] = x[:, lag_abs:]
        shifted_mask[:, :-lag_abs] = mask[:, lag_abs:]

    return shifted, shifted_mask


def reverse_valid_sequence(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size, time_steps = mask.shape
    positions = torch.arange(time_steps, device=x.device).unsqueeze(0).expand(batch_size, -1)
    lengths = mask.sum(dim=1).clamp(max=time_steps)
    valid_positions = positions.masked_fill(~mask, time_steps).sort(dim=1).values
    ranks = mask.cumsum(dim=1) - 1
    reverse_ranks = (lengths.unsqueeze(1) - 1 - ranks).clamp_min(0)
    reverse_positions = valid_positions.gather(dim=1, index=reverse_ranks)
    gather_idx = torch.where(mask, reverse_positions, positions)
    return x.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def safe_valid_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.clone()
    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty, 0] = True
    return mask


def _autocast_enabled(x: torch.Tensor) -> bool:
    try:
        return bool(torch.is_autocast_enabled(x.device.type))
    except TypeError:
        return bool(torch.is_autocast_enabled())


@MODELS.register("av_sync_mamba_segment_model")
def av_sync_mamba_segment_model(context: Any | None = None, **params: Any) -> AVSyncMambaSegmentModel:
    params = dict(params)
    if context is not None:
        params.setdefault("audio_feature_dim", context.get("data.audio_feature_dim"))
        params.setdefault("video_feature_dim", context.get("data.video_feature_dim"))
        params.setdefault("num_tasks", context.get("data.num_tasks", 2))
        params.setdefault("num_classes", context.get("data.num_classes", 2))

    return AVSyncMambaSegmentModel(**params)
