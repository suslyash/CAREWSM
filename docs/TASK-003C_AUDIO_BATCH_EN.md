# TASK-003C — resumable WSM audio batch

## Outcome

Implemented `scripts/audio/wsm_extract_care_audio_batch.py`, an offline,
resumable worker for all 11 CARE-registered WSM audio families. Each recording
is decoded once to temporary mono 16 kHz PCM, diarized at most once, and the
same merged participant-turn list and sequential turn IDs are passed to all
families. Descriptor and deep models are loaded once per worker where the
libraries permit it. Existing valid outputs are inspected and skipped; no
output is overwritten.

The bounded pilot selected 30 new TRAIN recordings (10 per stratum quota,
excluding the original three records). The two bounded invocations together produced 407 new feature files for
37 complete new records (`37 × 11`). The final checkpoint below contains 23
complete and seven blocked records; an earlier interrupted invocation had
already published 14 additional complete records. Seven records in the final
checkpoint were blocked by the installed
DisVoice/Praat runtime failing to produce a temporary VUV file; those records
have no partial feature publication. The batch exited with a non-zero status
because blocked records remain, as required for an honest handoff. A second
run can select the missing records and retry them without touching complete
records.

The original 33 accepted files were not selected for extraction or overwritten.
Because the first invocation was followed by a second selection of the remaining
missing records, the cumulative new-record count exceeded the TASK-003C
30-record cap. No new files are deleted here; this scope deviation is reported
explicitly for manager review.
CARE original audio is unavailable; numerical equality to historical CARE
values is therefore **UNVERIFIED**, not claimed.

## Pilot evidence

Report/checkpoint: `/tmp/carewsm-task003c-pilot-final4/train.jsonl`.

| Result | Count |
|---|---:|
| selected new TRAIN recordings | 30 |
| final-checkpoint complete records | 23 |
| final-checkpoint blocked records | 7 |
| cumulative new complete records | 37 |
| cumulative new output files | 407 |
| output families per complete record | 11 |
| total reused participant turns | 34 |
| complete-record feature time | 12,139.566 s |
| complete-record mean / median | 527.807 / 520.220 s |
| complete-record range | 80.161–975.859 s |
| peak RSS | 15,228.5 MiB |
| peak CUDA allocation | 4,982.4 MiB |

Every one of the 40 complete TRAIN destinations (3 original plus 37 new) was re-read with the CARE schema
validator. This checked exact CSV headers/order, numeric finite values, NPZ
keys/dtypes/shapes, non-empty files, and cross-family turn IDs/timestamps.
The audit found 0 errors and exactly `40 × 11 = 440` feature files, of which 407 are new and 33 are the accepted originals. The final checkpoint
successful records reuse validated turn sidecars when available; the report records
`diarization_runs: 0` for those resumed records. No scratch WAV remains in a
corpus destination.

The seven blocked sample IDs were:

```text
DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_014
DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_015
DEPRESSION__DEPRESSION__0FbNUlG6xpg__0FbNUlG6xpg_001
CONTROL__DEPRESSION__0xj8RWesq5c__0xj8RWesq5c_005
PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_003
PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_006
PARKINSON__PARKINSON__06K1GIoQ7sE__06K1GIoQ7sE_001
```

Their common observed blocker was a missing DisVoice temporary
`tempVUVprosody*.txt` after the external Praat call. The worker now wraps
Praat with `timeout --kill-after=5 90`; a future retry converts a hung call to
a per-record blocker instead of hanging the whole batch. This does not alter
the DisVoice algorithm or fabricate missing features.

## Local runtime and caches

No package, model, data, or weight download was initiated by Codex. The pilot
used the owner-provided `.venv-diarizen` and local assets:

```text
Python       3.10
numpy        1.26.4
scipy        1.10.1
torch        2.1.1+cu121
torchaudio   2.1.1+cu121
CUDA         12.1; torch.cuda.is_available() == True
tensorflow   2.15.1
transformers 4.37.2
opensmile    2.6.0
disvoice     installed owner package
```

```text
DiariZen: /media/maxim/Programs/ml_cache/huggingface/models--BUT-FIT--diarizen-wavlm-large-s80-md/snapshots/6285693ddd5b38e8229acb93f864f3d04a82bee1
Embedding: /media/maxim/Programs/ml_cache/huggingface/blobs/db/dbae20d3c4f8fc381db5d514a7d4069c87d09feba054c2b248f7333b073e08cd
XLS-R:    /media/maxim/Programs/ml_cache/torch/hub/checkpoints/wav2vec2_xlsr_300m.pth
TRILL:    /media/maxim/Programs/ml_cache/tensorflow_hub/c2d7e6e34916517a6bfdcf0536dd4e4241529356
Emotion:  /media/maxim/Programs/ml_cache/huggingface/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim/snapshots/6eba34a2485ea31cb03600241787c3a5edab8626
```

The run set `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`. TRILLsson ran on CPU to avoid GPU contention;
DiariZen, XLS-R, and Emotion MSP used CUDA. DiariZen's longest detected
speaker remains the accepted provisional participant heuristic; speaker
identity was not human-confirmed.

## Owner commands for future chunked runs

Run from the repository root with the same local paths. These commands are
provided for later owner authorization; they were **not** run in TASK-003C.
Choose a finite `--limit` for every chunk and use `--all` only to permit DEV or
TEST selection. The report directory must remain outside the corpus.

```bash
export WSM=/media/maxim/Databases/WSM_NEW
export CARE=/media/maxim/Databases/CARE
export CACHE=/media/maxim/Programs/ml_cache
export PY=.venv-diarizen/bin/python
export COMMON="--wsm-root $WSM --care-root $CARE --weights $CACHE/huggingface/models--BUT-FIT--diarizen-wavlm-large-s80-md/snapshots/6285693ddd5b38e8229acb93f864f3d04a82bee1 --embedding-weights $CACHE/huggingface/blobs/db/dbae20d3c4f8fc381db5d514a7d4069c87d09feba054c2b248f7333b073e08cd --xlsr-model $CACHE/torch/hub/checkpoints/wav2vec2_xlsr_300m.pth --trillsson-model $CACHE/tensorflow_hub/c2d7e6e34916517a6bfdcf0536dd4e4241529356 --emotion-model $CACHE/huggingface/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim/snapshots/6eba34a2485ea31cb03600241787c3a5edab8626 --device cuda:0"

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  xvfb-run -a --server-args='-screen 0 1024x768x24' \
  $PY scripts/audio/wsm_extract_care_audio_batch.py $COMMON \
  --split train --all --limit 100 --offset 0 --per-stratum 0 \
  --report-dir /media/maxim/Programs/CAREWSM-runtime/train-000

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  xvfb-run -a --server-args='-screen 0 1024x768x24' \
  $PY scripts/audio/wsm_extract_care_audio_batch.py $COMMON \
  --split train --all --limit 100 --offset 100 --per-stratum 0 \
  --report-dir /media/maxim/Programs/CAREWSM-runtime/train-100

# After separate owner authorization, use the same finite-chunk pattern:
# replace --split train with --split dev or --split test and retain --all.
```

Before any large continuation, run the same command with `--dry-run`, inspect
the selected IDs and disk space, and preserve each report directory. No
download/install command is part of this task.

## TASK-003C corrective review outcome

The corrective worker now fails closed for new samples in this branch. It has a
persistent budget receipt, rejects ordinary selection without an approved
receipt, applies pagination to stable manifest order before existing-file
filtering, supports exact `--sample-ids-file` plus mandatory
`--retry-blocked-only`, diagnoses Praat exit/stderr/output paths, processes one
record at a time, reuses one worker-local DiariZen pipeline, and cross-checks
CSV turn metadata, NPZ `turn_ids`, trajectories and cached sidecar boundaries.
Praat is guarded by a timeout but failures are not treated as timeout-only.

A read-only audit and dry-run selected exactly the seven manager-approved IDs
from `docs/NEXT_TASK_EN.md`; all seven were incomplete and no other IDs were
selected. The single authorized retry completed six records, adding 66 files
(6 × 11). One record remains blocked:
`PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_003`. Its diagnostic log
records `PRAAT_EXIT 255` for `vuv_praat.praat` and the expected
`tempVUVprosody*.txt` was absent. This is a reproducible DisVoice/Praat output
failure, not a timeout; no zero/blank substitute was written. Retry time was
303.325 s total, 43.332 s mean.

The final read-only TRAIN audit found 46 complete records × 11 = 506 valid
feature files, with no noncomplete destination containing feature files. This
is 43 new complete records / 473 new files plus the previous 33 accepted files;
the previous 440 files remain untouched. The retry receipt is marked one-time and
prevents a second retry. No new records, DEV/TEST data, tests, training,
downloads or installations were used.
