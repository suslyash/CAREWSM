# TASK-003A — bounded CARE-exact WSM audio descriptor extraction

## Scope and outcome

TASK-003A is limited to the seven lightweight audio families owned by this team. The five video and three linguistic families are external-team outputs and are not implemented, validated, downloaded, or written here. The four deep-audio families are reserved for TASK-003B.

The bounded extraction implementation and read-only parity validator were corrected and exercised. The three-record smoke reached local DiariZen but is **blocked by a NumPy/Torch ABI mismatch before feature extraction**. No WSM feature files were written.

## Fixed TRAIN selection

The extractor deterministically selects at most one available row from each stratum in `DEPRESSION`, `CONTROL`, `PARKINSON` order, capped at three:

| sample ID | task | source media | observed manifest `feature_dir` |
|---|---|---|---|
| `DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_001` | depression | `depression/train_labels/-7UpRmNVJzQ/segments/-7UpRmNVJzQ_001.mp4` | `depression/train_labels/-7UpRmNVJzQ/features/-7UpRmNVJzQ_001` |
| `CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_001` | depression | `depression/train_labels/0efS0axfX0E/segments/0efS0axfX0E_001.mp4` | `depression/train_labels/0efS0axfX0E/features/0efS0axfX0E_001` |
| `PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_001` | parkinson | `parkinson/train_labels/-1N99QsfWRk/segments/-1N99QsfWRk_001.mp4` | `parkinson/train_labels/-1N99QsfWRk/features/-1N99QsfWRk_001` |

The source paths and destination fields were read from the TRAIN manifest; no sample IDs were split to reconstruct paths. The current manifest `feature_dir` values are treated as source-feature metadata only. They are not used as output paths. The target is built from the real manifest fields `label_id`, `subgroup`, `profile_id`, and `video_id` under `WSM_NEW/extracted/`; the existing profile directories were checked before the smoke, and target VIDEO directories are created only for this authorized run.

## CARE parity contract implemented

The extractor follows the inspected CARE producers:

- 16 kHz mono audio via `ffmpeg`, then `librosa.load(..., sr=16000, mono=True)`;
- eGeMAPSv02 openSMILE Functionals and LowLevelDescriptors;
- CARE metadata order for functionals/turn files: `turn_id,turn_start_sec,turn_end_sec`;
- CARE metadata order for frame/segment files: `turn_id,turn_start_sec,turn_end_sec,frame_start_sec,frame_end_sec`;
- DisVoice `Prosody`, `Phonation`, and `Articulation` producers with `fmt="dataframe"`;
- DisVoice dynamic timing: prosody hop/offset `0.01/0`, phonation `0.02/11`, articulation `0.04/2`;
- CARE turn policy: provisional longest detected speaker as `PARTICIPANT`, interviewer labels as boundaries, merge gaps up to 3 seconds, discard turns shorter than 0.3 seconds;
- existing valid feature files are skipped, invalid existing files stop the run, and new CSVs use a temporary file inside the final destination directory before `os.replace`.

The seven target files and numeric widths are:

| file | level | metadata columns | numeric width |
|---|---|---:|---:|
| `egemaps_func.csv` | turn | 3 | 88 |
| `egemaps_lld.csv` | frame in turn | 5 | 25 |
| `disvoice_prosody_static.csv` | turn | 3 | 103 |
| `disvoice_prosody_dynamic.csv` | segment in turn | 5 | 13 |
| `disvoice_phonation_static.csv` | turn | 3 | 28 |
| `disvoice_phonation_dynamic.csv` | frame in turn | 5 | 7 |
| `disvoice_articulation_static.csv` | turn | 3 | 488 |

Headers are checked against the first matching CARE reference CSV by `validate_care_audio_parity.py`; numeric values are checked without loading large arrays. No placeholder or zero-filled features are produced.

## Commands and verification

Dry-run selection, no dependency import and no data write:

```bash
.venv-diarizen/bin/python scripts/audio/wsm_extract_care_descriptors.py \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --split train --limit 3 --dry-run
```

Read-only parity check for the same three rows:

```bash
python3 scripts/audio/validate_care_audio_parity.py \
  --care-root /media/maxim/Databases/CARE \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --split train --limit 3
```

The owner-installed environment contains `opensmile`, `disvoice`, `numpy`, `pandas`, `librosa`, `soundfile`, `torch`, `torchaudio`, `diarizen`, `tqdm`, Praat and Xvfb. The earlier NumPy/Torch and Praat blockers were resolved. The bounded smoke completed successfully through CUDA and headless Praat execution. All three selected TRAIN records produced all seven requested CSV families (21 files total) at the manifest-derived extracted destinations. DiariZen reported one detected speaker per record; participant labeling remains provisional longest-speaker relabeling. The run used a temporary `praat --run` wrapper to avoid Praat singleton IPC in headless execution; the extractor now creates that wrapper automatically. CARE headers, metadata order, widths, and numeric-value semantics matched the reference contract. Existing CARE files remained read-only.

For provenance, the bounded command used for the already-created smoke outputs was:

```bash
.venv-diarizen/bin/python scripts/audio/wsm_extract_care_descriptors.py \
  --wsm-root /media/maxim/Databases/WSM_NEW \
  --split train --limit 3 \
  --weights /media/maxim/Programs/ml_cache/huggingface/models--BUT-FIT--diarizen-wavlm-large-s80-md/snapshots/6285693ddd5b38e8229acb93f864f3d04a82bee1 \
  --embedding-weights /media/maxim/Programs/ml_cache/huggingface/blobs/db/dbae20d3c4f8fc381db5d514a7d4069c87d09feba054c2b248f7333b073e08cd
```

This command was run for the three authorized rows and completed with status `ok`. Existing valid outputs were not overwritten; temporary audio artifacts were cleaned.

## Owner-only dependency commands and traffic

No package download was initiated by Codex. The owner-installed descriptor packages are present, but two local compatibility issues remain:

```bash
.venv-diarizen/bin/python -m pip install --upgrade numpy==1.26.4 "setuptools<81"
```

These repairs were owner-executed before the successful smoke. No package installation or model download was performed during this corrective review. Official sources: [NumPy 1.26.4 on PyPI](https://pypi.org/project/numpy/1.26.4/) and [setuptools on PyPI](https://pypi.org/project/setuptools/). The pinned setuptools upper bound restored `pkg_resources`, which the installed DisVoice dependency imports.

Official sources: [openSMILE Python](https://audeering.github.io/opensmile/install.html) and [DisVoice](https://github.com/jcvasquezc/DisVoice). DisVoice also requires Praat to be installed and available on `PATH` according to its official instructions. The extractor preflights `praat` and the display environment. If Praat is absent, the owner must install it separately using the official Ubuntu/package source. In a headless SSH session, run the extractor under an owner-installed `xvfb-run` wrapper; no virtual-display package is downloaded by Codex. The installed DisVoice wheel must also contain `disvoice/praat/vuv_praat.praat` and `disvoice/praat/FormantsPraat.praat`; the extractor now reports a blocker when those resources are absent.

The dependency setup is complete for this bounded smoke. Future environment changes remain owner-executed under the repository bandwidth rule. Post-install verification:

```bash
.venv-diarizen/bin/python - <<'PY'
import opensmile
from disvoice.prosody import Prosody
from disvoice.phonation import Phonation
from disvoice.articulation import Articulation
print('opensmile:', opensmile.__version__)
print('disvoice: imports ok')
print('prosody:', type(Prosody()).__name__)
print('phonation:', type(Phonation()).__name__)
print('articulation:', type(Articulation()).__name__)
PY
```

No video/text package, model, or feature extractor is requested by this task.

## CARE same-code comparison

The existing CARE archives contain reference descriptor CSVs and RTTMs but no
raw audio files (`.wav`, `.mp4`, `.flac`, or equivalent) in the available CARE
file inventory. Therefore the same-code CARE numeric extraction could not be
run without inventing or downloading an input source. The validator still
compares the generated-output contract against the existing CARE CSV headers,
metadata order, numeric widths, and numeric-value semantics. CARE reference
files and archives remained read-only.

## Data integrity and limitations

CARE reference files, WSM manifests, raw media and transcripts were not changed. The authorized 21 WSM feature CSVs already existed before this corrective review and were audited read-only; no extraction was rerun and no feature was overwritten. Temporary audit artifacts were not retained. DiariZen respondent identity remains provisional and TASK-002C human listening is deferred; any eventual output must preserve that uncertainty in the extraction report. The seven descriptors are not all 11 audio families, and none of the eight nonaudio families are part of this team's extraction.
