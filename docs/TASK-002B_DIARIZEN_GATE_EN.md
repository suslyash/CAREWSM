# TASK-002B — bounded WSM DiariZen feasibility gate

## Outcome

The local-only gate **completed successfully pending manager review**. It
selected three existing TRAIN recordings, measured their stereo channels
before mono conversion, ran DiariZen with explicitly supplied local weights,
and produced runtime, RTF, speaker and segment summaries. No package, model,
or weight was downloaded by the gate.

This is not evidence that DiariZen is usable corpus-wide. The bounded run
provides runtime and detected-turn observations, but respondent/interviewer
identity remains unverified.

## Exact command and bounded selection

```bash
./scripts/audio/wsm_diarizen_gate.py \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --weights /path/to/hf-cache/models--BUT-FIT--diarizen-wavlm-large-s80-md/snapshots/<revision> \
  --embedding-weights /path/to/hf-cache/models--pyannote--wespeaker-voxceleb-resnet34-LM/snapshots/<revision>/pytorch_model.bin
```

The CLI requires an explicit WSM root and an existing local `--weights` model
directory. It reads only `splits/train.csv`, rejects non-TRAIN and unsafe paths,
sorts deterministic existing media candidates, and
selects at most one row from each available stratum in this order:
`DEPRESSION`, `CONTROL`, `PARKINSON`. The observed selection was:

| sample ID | task | stratum | source path | source duration (s) | audio |
|---|---|---|---|---:|---|
| `DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_001` | depression | DEPRESSION | `depression/train_labels/-7UpRmNVJzQ/segments/-7UpRmNVJzQ_001.mp4` | 75.263965 | AAC, 16 kHz, 2 channels |
| `CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_001` | depression | CONTROL | `depression/train_labels/0efS0axfX0E/segments/0efS0axfX0E_001.mp4` | 61.463965 | AAC, 16 kHz, 2 channels |
| `PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_001` | parkinson | PARKINSON | `parkinson/train_labels/-1N99QsfWRk/segments/-1N99QsfWRk_001.mp4` | 100.063965 | AAC, 16 kHz, 2 channels |

Each source duration is below the 120-second gate cap. The source files and
WSM roots were read only.

## Stereo-channel check

For each selected file, `ffmpeg` decoded at most 120 seconds to an ephemeral
16 kHz, two-channel PCM WAV. Channel RMS ratio and correlation were measured
before any potential DiariZen mono input would be made:

| sample ID suffix | left RMS | right RMS | RMS ratio | correlation | conclusion |
|---|---:|---:|---:|---:|---|
| `-7UpRmNVJzQ_001` | 1477.844 | 1479.388 | 0.998956 | 0.952633 | not evidence of separated speakers |
| `0efS0axfX0E_001` | 1401.384 | 1773.100 | 0.790358 | 0.538052 | not evidence of separated speakers |
| `-1N99QsfWRk_001` | 775.250 | 775.248 | 0.999998 | 0.999723 | not evidence of separated speakers |

These channel statistics cannot identify respondent versus interviewer. The
CARE reference policy therefore remains unverified for WSM: CARE labels the
longest-speaking speaker `PARTICIPANT`, while `turn_utils.py` merges gaps up
to three seconds and treats interviewer speech as a boundary. Those semantics
were not treated as verified speaker identity; the reported longest-speaker
mapping remains a diagnostic only.

## Local setup and measured runtime

The successful run used a separate Python 3.10 `.venv-diarizen` environment
with the official DiariZen stack and CUDA GPU execution. The model directory
was `/media/maxim/Programs/ml_cache/huggingface/models--BUT-FIT--diarizen-wavlm-large-s80-md/snapshots/6285693ddd5b38e8229acb93f864f3d04a82bee1`,
and the local WeSpeaker embedding file was
`/media/maxim/Programs/ml_cache/huggingface/blobs/db/dbae20d3c4f8fc381db5d514a7d4069c87d09feba054c2b248f7333b073e08cd`.

| sample ID | audio duration (s) | elapsed (s) | RTF | speakers | segments | longest-speaker coverage |
|---|---:|---:|---:|---:|---:|---:|
| `DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_001` | 75.263965 | 2.158076 | 0.028673 | 1 | 23 | 0.659545 |
| `CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_001` | 61.463965 | 1.429725 | 0.023261 | 1 | 24 | 0.587987 |
| `PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_001` | 100.063965 | 2.495638 | 0.024940 | 1 | 21 | 0.871842 |

All three runs returned one detected speaker. This is a runtime and bounded
feasibility result, not proof that the recordings contain one human speaker or
that speaker `1` is the respondent. `respondent_interviewer_sanity` therefore
remains `unverified`.

Temporary mono WAV and RTTM artifacts were created only under a temporary
directory and cleaned after the run. No files were written under CARE or WSM.

## Recommendation for manager decision

**Retain provisionally / narrow-optimize only if needed**: the bounded run is
fast on the observed GPU (RTF below 0.03 for all three files), and the gate
completed without corpus writes or downloads. The single-speaker outputs and
unverified respondent mapping do not establish participant-turn quality;
manager/owner review is still required before using DiariZen for extraction.

## Manual provisioning commands (not run by Codex)

The verified environment state before manual provisioning was:

```text
torch       2.11.0+cu128
torchaudio 2.11.0+cu128
CUDA build  12.8
CUDA        available: True
GPU         NVIDIA GeForce RTX 4080
tqdm        4.70.1
```

The official DiariZen README specifies Python 3.10, PyTorch/torchaudio 2.1.1
for CUDA 12.1, installation from its `requirements.txt` and editable source,
and the local `pyannote-audio` package. The successful run used the separate
`.venv-diarizen`; the CAREWSM `.venv` was not modified by the gate.

```bash
.venv-diarizen/bin/python - <<'PY'
import torch, torchaudio
print(torch.__version__, torchaudio.__version__, torch.version.cuda, torch.cuda.is_available())
PY

# Official DiariZen CUDA pair for this Python 3.10 environment, if needed.
.venv-diarizen/bin/pip install --upgrade --force-reinstall \
  torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the official source and dependencies into the existing environment:

```bash
DIARIZEN_SRC=/tmp/CAREWSM-DiariZen
git clone --depth 1 --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git "$DIARIZEN_SRC"
.venv-diarizen/bin/pip install -r "$DIARIZEN_SRC/requirements.txt"
.venv-diarizen/bin/pip install -e "$DIARIZEN_SRC"
.venv-diarizen/bin/pip install -e "$DIARIZEN_SRC/pyannote-audio[dev,testing]" -c "$DIARIZEN_SRC/constraints.txt"
```

Cache the official DiariZen model and its additional WeSpeaker embedding
weight. The downloader is intentionally not run by Codex:

```bash
HF_CACHE=/media/maxim/Programs/ml_cache/huggingface
.venv-diarizen/bin/python scripts/audio/cache_diarizen_weights.py --cache-dir "$HF_CACHE"
```

The downloader uses the official repositories
`BUT-FIT/diarizen-wavlm-large-s80-md` and
`pyannote/wespeaker-voxceleb-resnet34-LM` (`pytorch_model.bin`) through
`huggingface_hub`. It prints exact local paths. Verify both paths before running
the gate:

```bash
.venv-diarizen/bin/python - <<'PY'
import json, subprocess
print('Use the JSON paths printed by cache_diarizen_weights.py as --weights and --embedding-weights.')
print('No model download is performed by this check.')
PY
```

Then run the bounded gate with those printed paths:

```bash
.venv-diarizen/bin/python scripts/audio/wsm_diarizen_gate.py \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --weights /exact/model_dir/from/downloader \
  --embedding-weights /exact/pytorch_model.bin/from/downloader
```

The model weights are distributed by the official project under CC BY-NC 4.0
(non-commercial research) terms. Hugging Face access, cache permissions, and
any model-card terms must be accepted by the user before downloading. The gate
keeps converted mono WAVs and RTTM files only in an ephemeral temporary
directory and never writes CARE/WSM roots.
