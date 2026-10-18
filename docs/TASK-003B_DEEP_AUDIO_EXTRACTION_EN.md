# TASK-003B — bounded deep-audio extraction

## Outcome

The offline implementation covers the four CARE-compatible WSM deep-audio families:
`wav2vec_xlsr300m.npz`, `trillsson4_1.npz`, `emotionmsp_trajectories.csv`, and
`emotionmsp_embeds.npz`. It uses manifest-derived canonical destinations and the
CARE turn policy (longest detected speaker provisionally labelled participant,
interviewer hard boundaries, maximum gap 3 seconds, minimum turn 0.3 seconds).

The locally cached XLS-R checkpoint was available. Three `wav2vec_xlsr300m.npz`
files were created on CUDA for the same TRAIN records. DiariZen generated turns
in temporary scratch space because no persistent local `diarizen_diarization.rttm`
was present; no RTTM was retained. The TRILLsson and Emotion MSP families
remained explicitly blocked because their package/model resources were absent.
The run did not overwrite the 21 accepted CSVs or trigger downloads.

Original CARE audio is unavailable. Numerical equality with historical CARE values
remains UNVERIFIED and is not a blocker.

## CARE source contract

The implementation was based on the read-only CARE producers:

- `extract_wav2vec.py`: torchaudio `WAV2VEC2_XLSR_300M`, 16 kHz resampling,
  final feature layer `(T, 1024)`, per-turn mean and population standard
  deviation (`unbiased=False`), NPZ keys `mean`, `std`, `turn_ids`.
- `extract_trillsson.py`: TensorFlow Hub `google/trillsson4/1`, 16 kHz
  librosa input, embedding-frame mean and TensorFlow standard deviation, the
  same NPZ keys and `(n_turns, 1024)` float32 contract.
- `extract_audeering_emotionmsp.py`: local Hugging Face model
  `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`, mean-pooled 1024-D
  hidden embedding, regression order `arousal`, `dominance`, `valence`, and
  one trajectory CSV row per CARE participant turn.
- `turn_utils.py`: interviewer boundaries, participant gaps up to 3 seconds,
  and minimum turn duration 0.3 seconds.

The canonical output is
`WSM_NEW/extracted/<label_id>/<subgroup>/PROFILE_<profile_id>/VIDEO_<video_id>/`;
legacy `feature_dir` is never used as an output. Staging is on the destination
filesystem and publishing uses no-clobber hard links after full validation.

## Local preflight

Observed without installation or download:

- Python environment: `.venv-diarizen`, Python 3.10.
- NumPy 1.26.4; torch 2.1.1+cu121; torchaudio 2.1.1+cu121; CUDA available;
  NVIDIA GeForce RTX 4080.
- TensorFlow 2.15.1 and librosa 0.11.0 are present.
- `transformers` and `tensorflow_hub` are absent.
- The targeted local Hugging Face Emotion MSP snapshot is absent.
- No targeted XLS-R checkpoint was found in the inspected torch/cache paths.
- No targeted TRILLsson SavedModel was found in the inspected cache paths.
- The three canonical target directories contain the accepted seven CSVs and,
  after the bounded run, one valid XLS-R NPZ each; no persistent DiariZen RTTM
  was present.

The read-only CARE representative had exact shapes: XLS-R and TRILLsson
`mean/std` `(1,1024)` float32 plus int64 `turn_ids`; Emotion MSP `embed`
`(1,1024)` float32 plus int64 `turn_ids`; trajectory header exactly
`turn_id,turn_start_sec,turn_end_sec,arousal,dominance,valence`.

## Owner-only provisioning commands (not run)

These commands are options only; the owner must review and execute them. They
are intentionally absent from the implementation path. Expected transfer sizes
are approximate/unknown until the source reports `Content-Length`; allow roughly
1–2 GB for the XLS-R checkpoint, 0.5–2 GB for Emotion MSP plus config files, and
an unknown TensorFlow Hub payload that may be hundreds of MB. Cache locations are
explicit so the existing environment is not altered accidentally.

XLS-R official torchaudio asset (the bundle reports
`wav2vec2_xlsr_300m.pth`; no download was attempted):

```bash
mkdir -p /media/maxim/Programs/ml_cache/torch/hub/checkpoints
curl -L --fail --retry 2 \
  -o /media/maxim/Programs/ml_cache/torch/hub/checkpoints/wav2vec2_xlsr_300m.pth \
  https://download.pytorch.org/torchaudio/models/wav2vec2_xlsr_300m.pth
.venv-diarizen/bin/python - <<'PY'
from pathlib import Path
p=Path("/media/maxim/Programs/ml_cache/torch/hub/checkpoints/wav2vec2_xlsr_300m.pth")
print(p, p.stat().st_size, p.is_file())
PY
```

Emotion MSP official model ID (Hugging Face model card):

```bash
.venv-diarizen/bin/python -m pip install "transformers==4.37.2"
HF_HOME=/media/maxim/Programs/ml_cache/huggingface \
.venv-diarizen/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    local_dir_use_symlinks=False,
))
PY
```

This transfers the model and transitive package dependencies; exact size is not
known locally. Verify the printed snapshot contains `config.json`, processor
files, and model weights before running.

TRILLsson official TensorFlow Hub handle is `https://tfhub.dev/google/trillsson4/1`.
The installed script accepts an extracted local SavedModel directory; the owner
should download/cache it with the TensorFlow Hub tooling and pass that directory
through `--trillsson-model`. Exact payload size is source-dependent and was not
queried by Codex. Verify the directory contains a SavedModel and the `embedding`
output signature.

## Commands and verification

Dry-run/preflight (expected exit 1 with explicit blocked JSON in the current
state; it performs no inference and no writes):

```bash
.venv-diarizen/bin/python scripts/audio/wsm_extract_care_deep_audio.py \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --care-root /media/maxim/Databases/CARE \
  --split train --limit 3 --dry-run
```

After the owner supplies every local model path and the three RTTMs exist, the
bounded command is:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
.venv-diarizen/bin/python scripts/audio/wsm_extract_care_deep_audio.py \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --care-root /media/maxim/Databases/CARE \
  --split train --limit 3 \
  --xlsr-model /media/maxim/Programs/ml_cache/torch/hub/checkpoints/wav2vec2_xlsr_300m.pth \
  --trillsson-model /OWNER/LOCAL/trillsson4_1 \
  --emotion-model /media/maxim/Programs/ml_cache/huggingface/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim/snapshots/<revision> \
  --device cuda:0
```

Read-only validation of all 11 audio families is:

```bash
.venv-diarizen/bin/python scripts/audio/validate_care_audio_parity.py \
  --care-root /media/maxim/Databases/CARE \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --split train --limit 3 --all-audio
```

No tests, training, bulk extraction, video/text work, package installation, or
model download was run.

## Bounded result

| sample | turns | device | wall time | XLS-R bytes | status |
|---|---:|---|---:|---:|---|
| `DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_001` | 2 | `cuda:0` | 2.583 s | 17,148 | created/validated |
| `CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_001` | 1 | `cuda:0` | 1.698 s | 8,948 | created/validated |
| `PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_001` | 1 | `cuda:0` | 1.915 s | 8,948 | created/validated |

Each NPZ has exact keys `mean`, `std`, `turn_ids`, float32 `(n_turns,1024)`
arrays, int64 sequential turn IDs, and finite values. The three other families
remain missing and therefore full 11-family validation is intentionally not PASS.
