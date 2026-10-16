from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "FacebookAI/xlm-roberta-base"
MODEL_REVISION = "e73636d"
EXPECTED_MODEL_SHA256 = "6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb"
CONTENT_TOKENS_PER_CHUNK = 510
FEATURE_DIM = 768


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_token_ids(tokenizer: Any, text: str) -> list[list[int]]:
    ids = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]
    if not ids:
        return []
    return [ids[start : start + CONTENT_TOKENS_PER_CHUNK] for start in range(0, len(ids), CONTENT_TOKENS_PER_CHUNK)]


def _special_mask(tokenizer: Any, ids: list[int]) -> list[int]:
    mask = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
    if len(mask) != len(ids):
        raise RuntimeError("tokenizer special-token mask length mismatch")
    return mask


@torch.no_grad()
def extract_chunk_features(encoder: "XLMRVideoTranscriptEncoder", text: str) -> torch.Tensor:
    chunks = chunk_token_ids(encoder.tokenizer, text)
    if not chunks:
        raise ValueError("empty decoded transcript is unavailable")
    vectors: list[torch.Tensor] = []
    for content in chunks:
        if encoder.tokenizer.num_special_tokens_to_add(pair=False) != 2:
            raise RuntimeError("T1 requires the frozen single-sequence two-special-token contract")
        model_ids = [int(encoder.tokenizer.cls_token_id), *content, int(encoder.tokenizer.sep_token_id)]
        if len(model_ids) > 512:
            raise RuntimeError("special-tokenized chunk exceeds 512 tokens")
        attention = torch.ones(1, len(model_ids), dtype=torch.long, device=encoder.device)
        input_ids = torch.tensor([model_ids], dtype=torch.long, device=encoder.device)
        hidden = encoder.model(input_ids=input_ids, attention_mask=attention).last_hidden_state[0]
        special = torch.tensor(_special_mask(encoder.tokenizer, model_ids), dtype=torch.bool, device=encoder.device)
        valid = attention[0].bool() & ~special
        if not bool(valid.any()):
            raise RuntimeError("chunk has no valid non-special tokens")
        vectors.append(hidden[valid].mean(dim=0).float().cpu())
    result = torch.stack(vectors).to(dtype=torch.float32)
    if result.ndim != 2 or result.shape[1] != FEATURE_DIM or not bool(torch.isfinite(result).all()):
        raise RuntimeError("invalid extracted feature tensor")
    return result


class XLMRVideoTranscriptEncoder:
    def __init__(self, *, revision: str = MODEL_REVISION, device: str = "cpu") -> None:
        if revision != MODEL_REVISION:
            raise ValueError(f"only frozen revision {MODEL_REVISION} is allowed")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision, use_fast=True)
        self.model = AutoModel.from_pretrained(MODEL_NAME, revision=revision)
        self.model.eval()
        self.model.to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = torch.device(device)
        self.resolved_commit = str(getattr(self.model.config, "_commit_hash", None) or revision)

    def encode(self, text: str) -> torch.Tensor:
        return extract_chunk_features(self, text)

    def fingerprint(self) -> dict[str, Any]:
        tokenizer_json = getattr(self.tokenizer, "init_kwargs", {})
        return {
            "model_name": MODEL_NAME,
            "requested_revision": MODEL_REVISION,
            "resolved_commit": self.resolved_commit,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "tokenizer_init_fingerprint": hashlib.sha256(json.dumps(tokenizer_json, sort_keys=True, default=str).encode()).hexdigest(),
            "max_model_length": int(self.tokenizer.model_max_length),
            "content_tokens_per_chunk": CONTENT_TOKENS_PER_CHUNK,
            "overlap_stride": 0,
            "pooling": "mean final hidden state over valid non-special tokens",
            "feature_dtype": "float32",
            "feature_dim": FEATURE_DIM,
        }
