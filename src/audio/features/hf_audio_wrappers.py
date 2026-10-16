from __future__ import annotations

import importlib
import importlib.util
from importlib import metadata as importlib_metadata
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio


class TransformersAudioClassificationExtractor:
    def __init__(
        self,
        *,
        model_name: str,
        sample_rate: int = 16000,
        layer: int = -1,
        temporal_pool: int = 8,
        device: str = "cuda",
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.model_name = model_name
        self.sample_rate = int(sample_rate)
        self.layer = int(layer)
        self.temporal_pool = int(temporal_pool)
        self.device = torch.device(device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device, dtype=torch.float16)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.min_audio_samples = 400
        conv_kernel = getattr(self.model.config, "conv_kernel", None)
        conv_stride = getattr(self.model.config, "conv_stride", None)
        if conv_kernel is not None and conv_stride is not None:
            self.min_audio_samples = 1
            for kernel, stride in zip(reversed(conv_kernel), reversed(conv_stride), strict=True):
                self.min_audio_samples = (self.min_audio_samples - 1) * int(stride) + int(kernel)

    @torch.no_grad()
    def extract(self, wav_path: str | Path) -> dict[str, torch.Tensor]:
        waveform = _read_audio(Path(wav_path), self.sample_rate)
        if waveform.numel() == 0:
            waveform = torch.zeros(self.min_audio_samples, dtype=torch.float32)
        if torch.any(waveform):
            waveform = (waveform - waveform.mean()) / waveform.std(unbiased=False).clamp_min(1e-5)

        window_samples = 30 * self.sample_rate
        chunks: list[torch.Tensor] = []
        for start in range(0, int(waveform.numel()), window_samples):
            chunk = waveform[start : start + window_samples]
            if chunk.numel() < self.min_audio_samples:
                chunk = F.pad(chunk, (0, self.min_audio_samples - chunk.numel()))

            inputs = self.processor(chunk.numpy(), sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            inputs["input_values"] = inputs["input_values"].half()
            with torch.amp.autocast(device_type=self.device.type, enabled=True):
                output = self.model(**inputs, output_hidden_states=True)

            hidden_states = getattr(output, "hidden_states", None)
            if hidden_states is not None:
                hidden = hidden_states[self.layer].squeeze(0).detach().cpu()
            else:
                hidden = output.logits.detach().cpu()
            chunks.append(hidden)

        features = torch.cat(chunks, dim=0) if chunks else torch.zeros(1, int(self.model.config.num_labels))
        if self.temporal_pool > 1 and features.shape[0] > 1:
            features = _temporal_average_pool(features, self.temporal_pool)
        return {"audio_temporal": features.float(), "audio_cls": features.mean(dim=0).float()}


class KintsugiDAMExtractor:
    def __init__(self, *, model_name: str, device: str = "cuda", **_: Any) -> None:
        self.model_name = model_name
        self.device = device
        self.pipeline = self._load_pipeline(model_name)

    @torch.no_grad()
    def extract(self, wav_path: str | Path) -> dict[str, torch.Tensor]:
        scores = self.pipeline.run_on_file(str(wav_path), quantize=False)
        if isinstance(scores, dict):
            values = [float(scores[key]) for key in sorted(scores)]
        else:
            values = [float(scores)]
        features = torch.tensor(values, dtype=torch.float32).view(1, -1)
        return {"audio_temporal": features, "audio_cls": features.squeeze(0)}

    def _load_pipeline(self, model_name: str) -> Any:
        from huggingface_hub import snapshot_download

        repo_dir = Path(snapshot_download(model_name))
        pipeline_path = repo_dir / "pipeline.py"
        spec = importlib.util.spec_from_file_location("kintsugi_dam_pipeline", pipeline_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Kintsugi pipeline from {pipeline_path}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(repo_dir))
        spec.loader.exec_module(module)
        return module.Pipeline()


class SpeechBrainEmotionExtractor:
    def __init__(self, *, model_name: str, device: str = "cuda", **_: Any) -> None:
        from speechbrain.inference.interfaces import foreign_class

        self.classifier = foreign_class(
            source=model_name,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": device},
        )

    @torch.no_grad()
    def extract(self, wav_path: str | Path) -> dict[str, torch.Tensor]:
        output = self.classifier.classify_file(str(wav_path))
        tensors = [_as_flat_tensor(item) for item in output if _as_flat_tensor(item) is not None]
        features = torch.cat(tensors).view(1, -1).float()
        return {"audio_temporal": features, "audio_cls": features.squeeze(0)}


class VoxProfileExtractor:
    MODULES = {
        "tiantiaf/wavlm-large-speech-flow": "src.model.fluency.wavlm_fluency",
        "tiantiaf/wavlm-large-voice-quality": "src.model.voice_quality.wavlm_voice_quality",
    }

    def __init__(self, *, model_name: str, sample_rate: int = 16000, device: str = "cuda", **_: Any) -> None:
        self.model_name = model_name
        self.sample_rate = int(sample_rate)
        self.device = torch.device(device)
        module = self._load_vox_module(self.MODULES[model_name])
        self.model = module.WavLMWrapper.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _load_vox_module(self, module_name: str) -> Any:
        try:
            distribution = importlib_metadata.distribution("vox-profile")
        except importlib_metadata.PackageNotFoundError as exc:
            raise ModuleNotFoundError(
                "Vox-Profile is not on PyPI as vox-profile-release. Install it from GitHub without deps:\n"
                "  pip install loralib\n"
                "  pip install --no-deps git+https://github.com/tiantiaf0627/vox-profile-release.git\n"
                "The --no-deps flag is important because Vox-Profile pins older torch/torchaudio versions."
            ) from exc

        package_root = Path(distribution.locate_file(""))
        editable_root = Path("external/vox-profile-release").resolve()
        if editable_root.exists():
            package_root = editable_root

        src_root = package_root / "src"
        module_path = package_root / Path(*module_name.split(".")).with_suffix(".py")
        if not module_path.exists():
            raise FileNotFoundError(
                f"Cannot find Vox-Profile module: {module_path}. Clone it to external/vox-profile-release and install with --no-deps -e."
            )

        old_src = sys.modules.pop("src", None)
        vox_src = types.ModuleType("src")
        vox_src.__path__ = [str(src_root)]
        sys.modules["src"] = vox_src
        try:
            return importlib.import_module(module_name)
        finally:
            sys.modules.pop("src", None)
            if old_src is not None:
                sys.modules["src"] = old_src

    @torch.no_grad()
    def extract(self, wav_path: str | Path) -> dict[str, torch.Tensor]:
        waveform = _read_audio(Path(wav_path), self.sample_rate).to(self.device).float()
        if self.model_name.endswith("speech-flow"):
            features = self._extract_speech_flow(waveform)
        else:
            features = self._extract_voice_quality(waveform)
        return {"audio_temporal": features.cpu().float(), "audio_cls": features.mean(dim=0).cpu().float()}

    def _extract_speech_flow(self, waveform: torch.Tensor) -> torch.Tensor:
        win = 3 * self.sample_rate
        step = self.sample_rate
        if waveform.numel() < win:
            waveform = F.pad(waveform, (0, win - waveform.numel()))
        windows = [waveform[start : start + win] for start in range(0, waveform.numel() - win + 1, step)]
        batch = torch.stack(windows, dim=0)
        fluency_logits, disfluency_logits = self.model(batch)
        return torch.cat([torch.softmax(fluency_logits, dim=-1), torch.sigmoid(disfluency_logits)], dim=-1)

    def _extract_voice_quality(self, waveform: torch.Tensor) -> torch.Tensor:
        win = 15 * self.sample_rate
        windows = []
        lengths = []
        for start in range(0, int(waveform.numel()), win):
            chunk = waveform[start : start + win]
            if chunk.numel() == 0:
                continue
            windows.append(chunk)
            lengths.append(int(chunk.numel()))
        batch = torch.nn.utils.rnn.pad_sequence(windows, batch_first=True)
        length = torch.tensor(lengths, device=self.device, dtype=torch.long)
        logits = self.model(batch, length=length, return_feature=False)
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, device=self.device)
        return torch.sigmoid(logits)


def _read_audio(wav_path: Path, sample_rate: int) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(str(wav_path))
    waveform = waveform.mean(dim=0)
    if int(source_rate) != int(sample_rate):
        waveform = torchaudio.functional.resample(waveform, int(source_rate), int(sample_rate))
    return waveform.detach().cpu()


def _temporal_average_pool(features: torch.Tensor, pool: int) -> torch.Tensor:
    pad = (-features.shape[0]) % int(pool)
    if pad:
        features = F.pad(features, (0, 0, 0, pad))
    return features.view(-1, int(pool), features.shape[-1]).mean(dim=1)


def _as_flat_tensor(item: Any) -> torch.Tensor | None:
    if torch.is_tensor(item):
        return item.detach().cpu().float().flatten()
    if isinstance(item, (int, float)):
        return torch.tensor([float(item)], dtype=torch.float32)
    if hasattr(item, "detach"):
        return item.detach().cpu().float().flatten()
    return None
