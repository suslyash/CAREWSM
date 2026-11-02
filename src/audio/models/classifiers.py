from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


def _unpack(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    values = batch.inputs["native"].float()
    if values.ndim != 3:
        raise ValueError("inputs['native'] must have shape [batch, time, features]")
    mask = batch.get_masks("sequence_mask")
    if mask is None:
        mask = batch.get_masks("native_mask")
    if mask is None:
        mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
    return values, mask.to(device=values.device, dtype=torch.bool)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class TaskHeads(nn.Module):
    """Independent Depression and Parkinson binary heads."""

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


class SequenceClassifier(BaseModel):
    """Shared projection, architecture-specific encoder and task heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU()
        )
        self.heads = TaskHeads(hidden_dim, dropout)

    def encode(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: Any) -> ModelOutput:
        values, mask = _unpack(batch)
        encoded = self.encode(self.projection(values), mask)
        features = _masked_mean(encoded, mask)
        logits = self.heads(features)
        return ModelOutput(preds=logits, aux={"features": features, "logits": logits})


class TSCMambaClassifier(SequenceClassifier):
    """TSCMamba-inspired temporal/spectral multiview SSM classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, num_layers: int = 2,
                 dropout: float = 0.1, mamba_required: bool = True) -> None:
        super().__init__(input_dim, hidden_dim, dropout)
        try:
            from mamba_ssm import Mamba
        except ImportError:
            Mamba = None
        if Mamba is None and mamba_required:
            raise ModuleNotFoundError("TSCMamba requires mamba_ssm; no fallback is used")
        self.spectral_projection = nn.Linear(hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.mamba_layers = nn.ModuleList(
            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ) if Mamba is not None else nn.ModuleList()

    def encode(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.mamba_layers:
            return values
        valid = values * mask.unsqueeze(-1)
        spectrum = torch.fft.rfft(valid, dim=1).abs()
        spectral_view = torch.fft.irfft(spectrum, n=values.shape[1], dim=1)
        encoded = self.normalization(valid + self.spectral_projection(spectral_view))
        for layer in self.mamba_layers:
            encoded = layer(encoded) * mask.unsqueeze(-1)
        return encoded


class MPTSNetClassifier(SequenceClassifier):
    """MPTSNet-inspired periodic multiscale local/global classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 96,
                 dropout: float = 0.1) -> None:
        super().__init__(input_dim, hidden_dim, dropout)
        self.periodic_block = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.global_block = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fusion = nn.Linear(hidden_dim * 4, hidden_dim)

    def encode(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = values * mask.unsqueeze(-1)
        power = torch.fft.rfft(valid, dim=1).abs().mean(dim=-1)
        if power.shape[1] > 1:
            power[:, 0] = 0
            period = int(power.argmax(dim=1).median().clamp_min(1).item())
        else:
            period = 1
        length = valid.shape[1]
        pad = (-length) % period
        folded = F.pad(valid, (0, 0, 0, pad)).reshape(
            valid.shape[0], (length + pad) // period, period, valid.shape[2]
        ).permute(0, 3, 1, 2)
        periodic = F.gelu(self.periodic_block(folded))
        periodic = periodic.permute(0, 2, 3, 1).reshape(valid.shape[0], -1, valid.shape[2])[:, :length]
        global_view, _ = self.global_block(valid)
        return self.fusion(torch.cat((valid, periodic, periodic, global_view), dim=-1))


class MDFormerClassifier(SequenceClassifier):
    """MD-Former-inspired independent temporal and cross-channel branches."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__(input_dim, hidden_dim, dropout)
        temporal_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True,
            norm_first=True
        )
        channel_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True,
            norm_first=True
        )
        self.temporal_branch = nn.TransformerEncoder(temporal_layer, num_layers)
        self.channel_branch = nn.TransformerEncoder(channel_layer, num_layers)
        self.channel_projection = nn.Linear(1, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def encode(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_mask = mask.clone()
        safe_mask[:, 0] = True
        temporal = self.temporal_branch(values, src_key_padding_mask=~safe_mask)
        channel_input = _masked_mean(values, mask).unsqueeze(-1)
        channel = self.channel_branch(self.channel_projection(channel_input))
        channel = channel.mean(dim=1, keepdim=True).expand(-1, values.shape[1], -1)
        return self.fusion(torch.cat((temporal, channel), dim=-1))


class PooledMLP(SequenceClassifier):
    """Reference pooled classifier."""

    def encode(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        return values


@MODELS.register("carewsm_tscmamba")
def carewsm_tscmamba(context: Any | None = None, **params: Any) -> TSCMambaClassifier:
    del context
    return TSCMambaClassifier(**params)


@MODELS.register("carewsm_mptsnet")
def carewsm_mptsnet(context: Any | None = None, **params: Any) -> MPTSNetClassifier:
    del context
    return MPTSNetClassifier(**params)


@MODELS.register("carewsm_md_former")
def carewsm_md_former(context: Any | None = None, **params: Any) -> MDFormerClassifier:
    del context
    return MDFormerClassifier(**params)


@MODELS.register("carewsm_pooled_mlp")
def carewsm_pooled_mlp(context: Any | None = None, **params: Any) -> PooledMLP:
    del context
    return PooledMLP(**params)
