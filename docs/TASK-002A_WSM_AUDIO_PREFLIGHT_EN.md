# TASK-002A — WSM audio input and CARE feature-contract preflight

## Scope and result

This is a read-only TRAIN-only preflight. It does not extract audio, create
features, invoke DiariZen, enumerate TEST, or write CARE/WSM data. The only new
executable is the standalone standard-library script
[`scripts/audio/wsm_audio_preflight.py`](../scripts/audio/wsm_audio_preflight.py).
It imports no ChimeraML code and uses `ffprobe` only through a bounded
subprocess call.

## WSM input observations

The fixed manifest is `/media/maxim/Databases/WSM_NEW/splits/train.csv`. Its
observed path contract is:

- `split=train` identifies the allowed partition;
- `video_path` is a relative path rooted at `WSM_NEW`, for example
  `depression/train_labels/0efS0axfX0E/segments/0efS0axfX0E_001.mp4`;
- `storage_type=files` and the resolved file exists;
- `feature_dir` is a feature destination path and is not used as a raw-media
  fallback.

The script reads only `splits/train.csv`, requires the observed `video_path`
column, rejects absolute or root-escaping paths, sorts existing candidates by
`(sample_id, video_path)`, and applies a hard `--limit` cap of 5. It fails with
a JSON `blocked` summary rather than guessing if the manifest/path contract is
ambiguous.

## Bounded preflight evidence

Command:

```bash
./scripts/audio/wsm_audio_preflight.py \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --limit 5
```

Exact command stdout summary (full media rows are represented in the table below):

```json
{
  "ffprobe_available": true,
  "limit": 5,
  "selected_count": 5,
  "split": "train",
  "status": "ok",
  "wsm_root": "/media/maxim/Databases/WSM_NEW"
}
```

Observed `ffprobe` output for all five selected TRAIN files:

| sample | duration (s) | audio | video | codec | sample rate | channels |
|---|---:|---|---|---|---:|---:|
| `CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_001` | 61.463965 | yes | yes | AAC | 16000 | 2 |
| `CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_002` | 64.400000 | yes | yes | AAC | 16000 | 2 |
| `CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_003` | 66.112000 | yes | yes | AAC | 16000 | 2 |
| `CONTROL__DEPRESSION__0xj8RWesq5c__0xj8RWesq5c_001` | 79.864000 | yes | yes | AAC | 16000 | 2 |
| `CONTROL__DEPRESSION__0xj8RWesq5c__0xj8RWesq5c_002` | 41.800000 | yes | yes | AAC | 16000 | 2 |

The script selected exactly 5 files, reported `status=ok`, and confirmed
`ffprobe_available=true`. The 16 kHz observation is measured source-media
metadata. It is not yet evidence that future extraction should skip explicit
resampling or mono conversion: CARE reference code still loads with 16 kHz and
`mono=True`.

## CARE preparation and turn policy

Read-only inspection of the CARE reference scripts under
`/media/maxim/Databases/CARE/scripts/` found:

- `audio_descriptors/extract_egemaps.py` uses `librosa.load(..., sr=16000,
  mono=True)`, reads `diarizen_diarization.rttm`, and writes eGeMAPS turn/frame
  outputs.
- `audio_descriptors/extract_disvoice.py` uses the same 16 kHz mono loading,
  writes temporary 16 kHz WAV turn chunks for DisVoice, and emits static and
  dynamic prosody/phonation/articulation outputs.
- `audio_descriptors/extract_wav2vec.py` resamples with the torchaudio bundle,
  uses participant turns, and stores per-turn mean/std embeddings.
- `audio_descriptors/extract_trillsson.py` loads at 16 kHz, uses participant
  turns, and stores per-turn mean/std embeddings from the TRILLsson model.
- `extract_diarizen.py` labels the longest-speaking speaker as `PARTICIPANT`
  and other speakers as `INTERVIEWER`; this is a reference policy only and was
  not executed here.
- `turn_utils.py` merges participant speech across gaps up to 3.0 seconds,
  treats interviewer speech as a hard boundary, and removes turns shorter than
  0.3 seconds. This is source-observed CARE behavior, not a WSM validation.

CARE producer output parent:

```text
<output>/<group>/<subgroup>/PROFILE_<profile_id>/VIDEO_<video_id>/
```

## CARE feature-contract mapping

The accepted TASK-001A audit measured the following representative schemas in
CARE. Column counts below exclude the listed metadata columns.

| CARE producer | output filename | level/storage | measured contract |
|---|---|---|---|
| `extract_egemaps.py` | `egemaps_func.csv` | turn CSV | 3 metadata + 88 numeric columns |
| `extract_egemaps.py` | `egemaps_lld.csv` | frame-in-turn CSV | 5 metadata + 25 numeric columns |
| `extract_disvoice.py` | `disvoice_prosody_static.csv` | turn CSV | 3 metadata + 103 numeric columns |
| `extract_disvoice.py` | `disvoice_prosody_dynamic.csv` | segment-in-turn CSV | 5 metadata + 13 numeric columns |
| `extract_disvoice.py` | `disvoice_phonation_static.csv` | turn CSV | 3 metadata + 28 numeric columns |
| `extract_disvoice.py` | `disvoice_phonation_dynamic.csv` | frame-in-turn CSV | 5 metadata + 7 numeric columns |
| `extract_disvoice.py` | `disvoice_articulation_static.csv` | turn CSV | 3 metadata + 488 numeric columns |
| `extract_wav2vec.py` | `wav2vec_xlsr300m.npz` | pooled turn NPZ | `mean`, `std` shape `(turns,1024)`, `turn_ids` |
| `extract_trillsson.py` | `trillsson4_1.npz` | pooled turn NPZ | `mean`, `std` shape `(turns,1024)`, `turn_ids` |

The registry represents the pooled embedding width as 2048 because mean and
standard deviation are both retained; the sampled NPZ arrays are each width
1024. CARE files occur under the audited reference directory, while WSM has no
matching registered feature files yet. These are contracts to validate during a
later extraction task, not extraction results.

## Dependencies and callable boundaries

The preflight itself uses only Python standard-library modules and an optional
external `ffprobe` executable. It does not import `chimera_ml`, use a plugin,
registry, trainer, callback, DataModule, training config, or model library.

CARE reference producers require the dependencies visible in their source:
librosa/opensmile/pandas/numpy for eGeMAPS; librosa/soundfile/DisVoice/pandas
for DisVoice; torch/torchaudio/numpy for XLS-R; and TensorFlow/TensorFlow Hub
plus librosa/numpy for TRILLsson. Those dependencies were not installed or
executed by TASK-002A.

## Explicit unresolved items

- The five media files are stereo at the container level (`channels=2`), while
  CARE extraction explicitly converts input to mono. WSM extraction must make
  that conversion explicit later.
- The preflight does not measure respondent turns or diarization quality; the
  DiariZen gate remains a separate bounded task.
- WSM does not yet have CARE-compatible feature files: TASK-001A measured 0/19
  registered feature filenames across the audited WSM manifests.
- `linguistic_turnlevel` remains unresolved: the registry declares
  `original_feature_dim=41`, while the sampled CSV has three metadata columns
  plus 39 numeric columns whose names match the registry family list. No
  normalization or silent padding was applied.
- CARE loader return types, normalization, padding, and masking remain
  unverified.

## One next extraction-smoke candidate

After owner approval of the separate extraction task, use the same five stable
TRAIN media paths as a bounded input smoke, first producing only the CARE-style
16 kHz mono audio plus RTTM/turn prerequisite and then validating one audio
family pair: `egemaps_func.csv` and `egemaps_lld.csv`. Record failures, output
paths, turn counts, metadata columns, numeric widths, dtypes, and fingerprints.
Do not treat this preflight as feature parity or as permission to extract the
full corpus.
