"""Frozen adapter for the exact historical temporal audio checkpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from audio.models.audio_mamba_segment import AudioMambaSegmentModel
from chimera_ml.core.batch import Batch


class FrozenAudioTemporalAdapter(nn.Module):
    """Strictly loaded, task-conditioned, non-trainable historical audio model."""

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        path = Path(self.checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"historical audio checkpoint is missing: {path}")
        self.audio_model = AudioMambaSegmentModel(
            audio_feature_dim=768, num_tasks=2, num_classes=2, hidden_dim=192,
            num_layers=3, num_heads=4, ff_mult=4, dropout=0.25,
            encoder_type="transformer", sequence_steps=128,
            mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_required=True,
        )
        # The archived epoch-4 state predates four later source normalizations.
        # Identity replacement reconstructs the archived forward graph exactly,
        # allowing strict loading without editing the frozen src/audio provider.
        for name in ("projected_audio_norm", "audio_output_norm", "segment_input_norm", "segment_output_norm"):
            setattr(self.audio_model, name, nn.Identity())
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("epoch") != 4:
            raise ValueError("checkpoint payload is not the required selected epoch-4 run")
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint has no model_state_dict")
        self.audio_model.load_state_dict(state, strict=True)
        for parameter in self.audio_model.parameters():
            parameter.requires_grad_(False)
        self.audio_model.eval()

    def train(self, mode: bool = True) -> "FrozenAudioTemporalAdapter":
        super().train(mode)
        self.audio_model.eval()
        return self

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        audio = batch.inputs["audio"]
        audio_mask = batch.get_masks("audio_mask")
        if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or tuple(audio.shape[2:]) != (768,):
            raise ValueError("audio must have shape [B,T,768]")
        if not torch.is_floating_point(audio) or not bool(torch.isfinite(audio).all()):
            raise ValueError("audio must be finite floating point")
        if not isinstance(audio_mask, torch.Tensor) or audio_mask.ndim != 2 or tuple(audio_mask.shape) != tuple(audio.shape[:2]):
            raise ValueError("audio_mask must have shape [B,T]")
        audio_mask = audio_mask.to(device=audio.device, dtype=torch.bool)
        if not bool(audio_mask.any(dim=1).all()):
            raise ValueError("every audio sample must have at least one valid position")
        outputs = []
        with torch.no_grad():
            for task in (0, 1):
                task_ids = torch.full((audio.shape[0],), task, device=audio.device, dtype=torch.long)
                internal = Batch(inputs={"audio": audio, "task_ids": task_ids}, targets=None, masks={"audio_mask": audio_mask}, meta={})
                outputs.append(self.audio_model(internal))
        legacy = torch.stack([output.preds for output in outputs], dim=1)
        features = torch.stack([output.aux["features"] for output in outputs], dim=1)
        base_logits = legacy[:, :, 1] - legacy[:, :, 0]
        return {"base_logits": base_logits, "task_features": features, "legacy_class_logits": legacy}
