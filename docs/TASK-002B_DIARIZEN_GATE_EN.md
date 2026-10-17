# TASK-002B — bounded WSM DiariZen feasibility gate

## Outcome

The local-only gate is **blocked pending manager review**. It selected three
existing TRAIN recordings, measured their stereo channels before any mono
conversion, and did not invoke DiariZen because the local environment lacks
both `diarizen` and `torchaudio`. No package, model, or weight was downloaded.

This is not evidence that DiariZen is unusable corpus-wide. The gate provides
no speaker segments, participant-turn coverage, wall-clock runtime, or RTF
because the required local runtime was unavailable.

## Exact command and bounded selection

```bash
./scripts/audio/wsm_diarizen_gate.py \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --weights /absolute/path/to/local/diarizen-wavlm-large-s80-md
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
were not applied without a successful diarization result.

## Local setup and blocker

Read-only checks found `ffprobe`, `ffmpeg`, `torch 2.14.1`, and
`transformers` locally. Python import checks found no `diarizen` and no
`torchaudio`. The corrected CLI now requires an existing local `--weights`
directory, sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, and has no installation or download path. With the
current missing dependencies, no temporary 16 kHz mono DiariZen input was
created and no model weights were accessed. If dependencies and weights become
available, the same CLI creates only ephemeral 16 kHz mono WAVs, invokes the
CARE-compatible `DiariZenPipeline.from_pretrained(local_weights)` path once per
selected file, writes temporary RTTM, and reports elapsed seconds, RTF, speaker
counts, segment durations, and longest-speaker coverage. In the current blocked
run, these DiariZen outputs remain **unmeasured**.

The temporary directory containing the bounded stereo WAVs was removed after
the command (`temporary_artifacts_cleaned: true`). No files were written under
CARE or WSM.

## Recommendation for manager decision

**Narrow-optimize / re-gate**, rather than retain or drop based on this result:
the three-file channel check is useful negative evidence against assuming
stereo speaker separation, but the missing local runtime prevents a DiariZen
feasibility decision. A future task would need explicit owner-approved local
installation/weight provisioning and the same bounded mono-16 kHz runtime
gate; it must not silently download models.
