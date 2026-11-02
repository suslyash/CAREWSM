#!/usr/bin/env python3
"""Diagnostic speech-content audit for a deterministic WSM sample."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering

# SpeechBrain 1.1 uses torch.amp.custom_fwd, while this project intentionally
# keeps Torch 2.1 for the extracted audio features.
if not hasattr(torch.amp, "custom_fwd"):
    def _custom_fwd_compat(fwd=None, device_type="cuda", cast_inputs=None):
        if fwd is None:
            return lambda fn: _custom_fwd_compat(fn, device_type, cast_inputs)
        if device_type == "cuda":
            return torch.cuda.amp.custom_fwd(fwd, cast_inputs=cast_inputs)
        return fwd

    torch.amp.custom_fwd = _custom_fwd_compat

from speechbrain.inference.speaker import EncoderClassifier


SR = 16_000
ROOT = Path("/media/maxim/Programs/Projects/CAREWSM")
DB = Path("/media/maxim/Databases/WSM_NEW")
STAGING = Path("/tmp/carewsm_registered_wavs")
TORCH_HUB = Path("/media/maxim/Programs/ml_cache/torch/hub/snakers4_silero-vad_master")
TF_CACHE = "/media/maxim/Programs/ml_cache/tensorflow_hub"


def load_vad():
    model, utils = torch.hub.load(str(TORCH_HUB), "silero_vad", source="local")
    return model, utils[0]


def load_yamnet():
    import tensorflow_hub as hub

    model = hub.KerasLayer("https://tfhub.dev/google/yamnet/1")
    class_map = Path(TF_CACHE) / "9616fd04ec2360621642ef9455b84f4b668e219e/assets/yamnet_class_map.csv"
    labels = pd.read_csv(class_map)["display_name"].tolist()
    return model, labels


def audio_tensor(path: Path) -> tuple[torch.Tensor, float]:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.clamp(-1, 1)
    return wav, wav.shape[-1] / SR


def intervals_from_vad(vad_model, get_speech_timestamps, wav_cpu, device):
    x = wav_cpu.squeeze(0)
    # Silero's timestamp helper is more stable on CPU; the heavy speaker model uses CUDA.
    ts = get_speech_timestamps(
        x,
        vad_model,
        sampling_rate=SR,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        speech_pad_ms=100,
        return_seconds=False,
    )
    raw = [(int(t["start"]) / SR, int(t["end"]) / SR) for t in ts]
    merged = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= 0.5:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def gaps(duration, speech):
    if not speech:
        return 0.0, 0.0
    speech = sorted(speech)
    total_long = 0.0
    count = 0
    cursor = 0.0
    for start, end in speech:
        gap = max(0.0, start - cursor)
        if gap >= 2.0:
            total_long += gap
            count += 1
        cursor = max(cursor, end)
    gap = max(0.0, duration - cursor)
    if gap >= 2.0:
        total_long += gap
        count += 1
    return total_long / max(duration, 1e-6), count


def speech_chunks(speech, max_chunks=16, chunk_sec=2.0):
    chunks = []
    for start, end in speech:
        pos = start
        while pos < end:
            stop = min(end, pos + chunk_sec)
            if stop - pos >= 1.0:
                chunks.append((pos, stop))
            pos += chunk_sec
    if len(chunks) <= max_chunks:
        return chunks
    idx = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
    return [chunks[i] for i in idx]


def speaker_stats(ecapa, wav, speech, device):
    chunks = speech_chunks(speech)
    if not chunks:
        return 0, 0.0, np.nan, np.nan
    embeds = []
    with torch.inference_mode():
        for start, end in chunks:
            x = wav[:, int(start * SR): int(end * SR)].to(device)
            if x.shape[-1] < SR:
                continue
            emb = ecapa.encode_batch(x).squeeze().detach().float().cpu().numpy()
            emb /= max(np.linalg.norm(emb), 1e-8)
            embeds.append(emb)
    if len(embeds) < 2:
        return len(embeds), 1.0, np.nan, np.nan
    emb = np.stack(embeds)
    sim = emb @ emb.T
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.55, metric="cosine", linkage="average"
    ).fit(emb)
    labels = clustering.labels_
    counts = np.bincount(labels)
    dominant = float(counts.max() / len(labels))
    within, between = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (within if labels[i] == labels[j] else between).append(float(sim[i, j]))
    within_mean = float(np.mean(within)) if within else np.nan
    between_mean = float(np.mean(between)) if between else np.nan
    separation = within_mean - between_mean if within and between else np.nan
    return int(len(counts)), dominant, separation, float(np.mean(sim[np.triu_indices(len(sim), 1)]))


def yamnet_probs(yamnet, labels, wav):
    x = wav.squeeze(0).numpy().astype(np.float32)
    scores, _, _ = yamnet(x)
    scores = np.asarray(scores)
    labels_l = [str(x).lower() for x in labels]
    music_terms = ("music", "song", "singing", "musical instrument", "choir")
    noise_terms = ("noise", "hiss", "static", "engine", "traffic", "siren", "crowd")
    music = [i for i, s in enumerate(labels_l) if any(t in s for t in music_terms)]
    noise = [i for i, s in enumerate(labels_l) if any(t in s for t in noise_terms)]
    return float(scores[:, music].sum(axis=1).mean()), float(scores[:, noise].sum(axis=1).mean())


def category(row):
    speech = row["speech_ratio"]
    music = row["music_prob"]
    noise = row["noise_prob"]
    clusters = row["speaker_clusters"]
    dominant = row["dominant_cluster_share"]
    if speech < 0.20:
        return "low-speech"
    if max(music, noise) >= 0.45 and speech < 0.55:
        return "music/noise-heavy"
    if row["n_speech_regions"] >= 8 or row["median_speech_duration"] < 0.75:
        return "fragmented"
    if clusters >= 2 and dominant < 0.80:
        return "multi-speaker-like"
    if clusters <= 1 or dominant >= 0.80:
        return "single-speaker-like"
    return "uncertain"


def select_files(n, seed):
    rng = np.random.default_rng(seed)
    files = []
    for dataset in ("depression", "parkinson"):
        candidates = sorted((STAGING / dataset).glob("**/segments/*.wav"))
        if not candidates:
            continue
        take = min(n // 2, len(candidates))
        files.extend(list(rng.choice(candidates, size=take, replace=False)))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", default=str(ROOT / "logs/wsm_speech_content_audit.csv"))
    args = parser.parse_args()
    os.environ.setdefault("TFHUB_CACHE_DIR", TF_CACHE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vad_model, get_speech_timestamps = load_vad()
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/media/maxim/Programs/ml_cache/speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    yamnet, labels = load_yamnet()
    rows = []
    for i, path in enumerate(select_files(args.n, args.seed), 1):
        wav, duration = audio_tensor(path)
        speech = intervals_from_vad(vad_model, get_speech_timestamps, wav, device)
        speech_sec = sum(max(0, b - a) for a, b in speech)
        pause_ratio, long_pauses = gaps(duration, speech)
        n_clusters, dominant, separation, mean_similarity = speaker_stats(ecapa, wav, speech, device)
        music_prob, noise_prob = yamnet_probs(yamnet, labels, wav)
        row = {
            "segment": str(path),
            "dataset": path.parts[path.parts.index("carewsm_registered_wavs") + 1],
            "duration_sec": round(duration, 4),
            "speech_ratio": round(speech_sec / max(duration, 1e-6), 6),
            "n_speech_regions": len(speech),
            "median_speech_duration": round(float(np.median([b - a for a, b in speech])) if speech else 0.0, 4),
            "long_pause_ratio": round(pause_ratio, 6),
            "n_long_pauses": long_pauses,
            "music_prob": round(music_prob, 6),
            "noise_prob": round(noise_prob, 6),
            "speaker_chunks": len(speech_chunks(speech)),
            "speaker_clusters": n_clusters,
            "dominant_cluster_share": round(dominant, 6),
            "speaker_separation": None if not np.isfinite(separation) else round(separation, 6),
            "mean_speaker_similarity": None if not np.isfinite(mean_similarity) else round(mean_similarity, 6),
        }
        row["category"] = category(row)
        rows.append(row)
        print(f"[{i}/{args.n}] {path.name} -> {row['category']}", flush=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
