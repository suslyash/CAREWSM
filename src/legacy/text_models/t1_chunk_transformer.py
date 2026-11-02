from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


class T1ChunkTransformer(BaseModel):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 192, num_heads: int = 4, ff_dim: int = 384, dropout: float = 0.20, num_tasks: int = 2) -> None:
        super().__init__()
        if (input_dim, hidden_dim, num_heads, ff_dim, num_tasks) != (768, 192, 4, 384, 2):
            raise ValueError("T1 architecture is frozen")
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, ff_dim, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)) for _ in range(2)])

    @staticmethod
    def _positions(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
        result = torch.zeros(length, dim, device=device, dtype=dtype)
        result[:, 0::2] = torch.sin(position * div)
        result[:, 1::2] = torch.cos(position * div)
        return result.unsqueeze(0)

    def forward(self, batch: Any) -> ModelOutput:
        values = batch.inputs["text"]
        mask = batch.get_masks("text_mask").to(device=values.device, dtype=torch.bool)
        if values.ndim != 3 or values.shape[-1] != 768 or mask.shape != values.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("T1 text input must be [B,C,768] with at least one valid chunk per sample")
        encoded = self.input_projection(values) + self._positions(values.shape[1], 192, values.device, encoded_dtype(values))
        encoded = self.output_norm(self.encoder(encoded, src_key_padding_mask=~mask))
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1)
        logits = torch.cat([head(pooled) for head in self.heads], dim=1)
        return ModelOutput(preds=logits, aux={"features": pooled})


def encoded_dtype(batch_values: torch.Tensor) -> torch.dtype:
    return batch_values.dtype if batch_values.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16) else torch.float32


@MODELS.register("wsm_text_t1_chunk_transformer")
def wsm_text_t1_chunk_transformer(context: Any | None = None, **params: Any) -> T1ChunkTransformer:
    del context
    return T1ChunkTransformer(**params)
