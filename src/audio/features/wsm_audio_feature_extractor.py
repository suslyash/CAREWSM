from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from audio.features.hf_audio_wrappers import (
    KintsugiDAMExtractor,
    SpeechBrainEmotionExtractor,
    TransformersAudioClassificationExtractor,
    VoxProfileExtractor,
)
from common.features.wsm_feature_extractor import WSMSegmentFeatureExtractor


@dataclass
class WSMAudioFeatureExtractor:
    extractor_type: str = "transformers_ssl"
    model_name: str = "microsoft/wavlm-base-plus"
    sample_rate: int = 16000
    layer: int = -1
    temporal_pool: int = 8
    device: str = "cuda"

    def __post_init__(self) -> None:
        extractor_type = str(self.extractor_type).lower()
        if extractor_type == "transformers_ssl":
            self.extractor = WSMSegmentFeatureExtractor(
                audio_model_name=self.model_name,
                sample_rate=self.sample_rate,
                audio_layer=self.layer,
                audio_temporal_pool=self.temporal_pool,
                device=self.device,
                load_audio_model=True,
                load_video_model=False,
            )
        elif extractor_type == "transformers_audio_classification":
            self.extractor = TransformersAudioClassificationExtractor(
                model_name=self.model_name,
                sample_rate=self.sample_rate,
                layer=self.layer,
                temporal_pool=self.temporal_pool,
                device=self.device,
            )
        elif extractor_type == "kintsugi_dam":
            self.extractor = KintsugiDAMExtractor(model_name=self.model_name, device=self.device)
        elif extractor_type == "speechbrain_emotion":
            self.extractor = SpeechBrainEmotionExtractor(model_name=self.model_name, device=self.device)
        elif extractor_type == "vox_profile":
            self.extractor = VoxProfileExtractor(
                model_name=self.model_name,
                sample_rate=self.sample_rate,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown audio feature extractor type: {self.extractor_type}")

    @torch.no_grad()
    def extract(self, wav_path: str | Path) -> dict[str, torch.Tensor]:
        if str(self.extractor_type).lower() == "transformers_ssl":
            audio_temporal = self.extractor.extract_audio(Path(wav_path))
            return {
                "audio_temporal": audio_temporal.cpu().float(),
                "audio_cls": audio_temporal.mean(dim=0).cpu().float(),
            }

        return self.extractor.extract(Path(wav_path))
