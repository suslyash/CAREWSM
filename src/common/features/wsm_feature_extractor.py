from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


@dataclass
class WSMSegmentFeatureExtractor:
    audio_model_name: str = "microsoft/wavlm-base-plus"
    video_model_name: str = "google/vit-base-patch16-224-in21k"
    sample_rate: int = 16000
    audio_layer: int = -1
    video_layer: int = -1
    audio_temporal_pool: int = 8
    video_fps: float = 2.0
    max_video_frames: int = 128
    video_batch_size: int = 16
    device: str = "cuda"
    load_audio_model: bool = True
    load_video_model: bool = True

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.dtype = torch.float16
        if self.load_audio_model:
            self.init_audio_model()
        if self.load_video_model:
            self.init_video_model()

    @torch.no_grad()
    def extract(self, segment_path: str | Path, wav_path: str | Path) -> dict[str, torch.Tensor]:
        segment_path = Path(segment_path)
        wav_path = Path(wav_path)
        audio_temporal = self.extract_audio(wav_path)
        video_temporal = self.extract_video(segment_path)
        return {
            "audio_temporal": audio_temporal.cpu().float(),
            "video_temporal": video_temporal.cpu().float(),
            "audio_cls": audio_temporal.mean(dim=0).cpu().float(),
            "video_cls": video_temporal.mean(dim=0).cpu().float(),
        }

    @torch.no_grad()
    def extract_audio(self, wav_path: Path) -> torch.Tensor:
        waveform = self.read_audio(wav_path)
        if waveform.numel() == 0:
            waveform = torch.zeros(self.min_audio_samples, dtype=torch.float32)

        waveform = waveform.float()
        if torch.any(waveform):
            waveform = (waveform - waveform.mean()) / waveform.std(unbiased=False).clamp_min(1e-5)

        window_samples = 30 * self.sample_rate
        chunks: list[torch.Tensor] = []
        for start in range(0, int(waveform.numel()), window_samples):
            chunk = waveform[start : start + window_samples]
            if chunk.numel() == 0:
                continue
            if chunk.numel() < self.min_audio_samples:
                chunk = F.pad(chunk, (0, self.min_audio_samples - chunk.numel()))

            inputs = self.audio_processor(
                chunk.numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            inputs["input_values"] = inputs["input_values"].half()
            with torch.amp.autocast(device_type=self.device.type, enabled=True):
                output = self.audio_model(**inputs, output_hidden_states=self.audio_layer != -1)

            hidden_states = getattr(output, "hidden_states", None)
            hidden = hidden_states[self.audio_layer] if hidden_states is not None else output.last_hidden_state
            chunks.append(hidden.squeeze(0).detach().cpu())

        features = torch.cat(chunks, dim=0) if chunks else torch.zeros(1, self.audio_model.config.hidden_size)
        if self.audio_temporal_pool > 1:
            features = _temporal_average_pool(features, self.audio_temporal_pool)

        return features

    @torch.no_grad()
    def extract_video(self, segment_path: Path) -> torch.Tensor:
        frames = _read_video_frames(segment_path, fps=self.video_fps, max_frames=self.max_video_frames)
        if not frames:
            hidden_size = int(getattr(self.video_model.config, "hidden_size", 1))
            return torch.zeros(1, hidden_size)

        chunks: list[torch.Tensor] = []
        for start in range(0, len(frames), self.video_batch_size):
            batch_frames = frames[start : start + self.video_batch_size]
            inputs = self.video_processor(images=batch_frames, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.amp.autocast(device_type=self.device.type, enabled=True):
                output = self.video_model(**inputs, output_hidden_states=self.video_layer != -1)

            hidden_states = getattr(output, "hidden_states", None)
            hidden = hidden_states[self.video_layer] if hidden_states is not None else output.last_hidden_state
            if hidden.ndim == 3:
                hidden = hidden[:, 0]

            chunks.append(hidden.detach().cpu())

        return torch.cat(chunks, dim=0)

    def init_audio_model(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        self.audio_processor = AutoFeatureExtractor.from_pretrained(self.audio_model_name)
        self.audio_model = AutoModel.from_pretrained(self.audio_model_name).to(device=self.device, dtype=self.dtype)
        self.audio_model.eval()
        for param in self.audio_model.parameters():
            param.requires_grad = False
        self.min_audio_samples = 1
        for kernel, stride in zip(
            reversed(self.audio_model.config.conv_kernel),
            reversed(self.audio_model.config.conv_stride),
            strict=True,
        ):
            self.min_audio_samples = (self.min_audio_samples - 1) * int(stride) + int(kernel)

    def init_video_model(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.video_processor = AutoImageProcessor.from_pretrained(self.video_model_name)
        self.video_model = AutoModel.from_pretrained(self.video_model_name).to(device=self.device, dtype=self.dtype)
        self.video_model.eval()
        for param in self.video_model.parameters():
            param.requires_grad = False

    def read_audio(self, wav_path: Path) -> torch.Tensor:
        waveform, source_rate = torchaudio.load(str(wav_path))
        waveform = waveform.mean(dim=0)
        if int(source_rate) != int(self.sample_rate):
            waveform = torchaudio.functional.resample(waveform, int(source_rate), int(self.sample_rate))

        return waveform.detach().cpu()


def _read_video_frames(path: Path, *, fps: float, max_frames: int) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps or 1.0)
    step = max(1, int(round(native_fps / max(float(fps), 1e-6))))

    frames: list[np.ndarray] = []
    index = 0
    while len(frames) < int(max_frames):
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        index += 1

    capture.release()
    return frames


def _temporal_average_pool(features: torch.Tensor, pool: int) -> torch.Tensor:
    pad = (-features.shape[0]) % int(pool)
    if pad:
        features = F.pad(features, (0, 0, 0, pad))

    return features.view(-1, int(pool), features.shape[-1]).mean(dim=1)
