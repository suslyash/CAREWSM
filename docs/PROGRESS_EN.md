# CAREWSM Active Progress

## Project state

Project: **CAREWSM**

Repository bootstrap source: a one-commit copy of the completed WSM codebase.

The previous WSM evidence/task history is intentionally excluded from the active CAREWSM research record. No old WSM metric or claim is an accepted CAREWSM result.

## Current stage

Stage 1 — corpus, feature, and legacy-code audit.

Status: **TASK-001A ACCEPTED (PR #3)**, **TASK-001B ACCEPTED (PR #5)**, **TASK-001C ACCEPTED (PR #9)** and **TASK-001D ACCEPTED (PR #12)**. Stage 2: **TASK-002A manager PASS, PR #15 merged**; **TASK-002B manager PASS for bounded GPU runtime (PR #21)**; TASK-002C playback checklist **accepted and deferred** due to SSH audio limitations. Owner prioritizes **TASK-003A real CARE-compatible WSM AUDIO feature extraction**, targeting only **11 audio families** (7 descriptors, 4 deep audio). Five video and three linguistic families are produced by other teams; CAREWSM integrates their outputs later in multimodal models. No feature extraction or training has run.

No CAREWSM scientific experiment has been run.

No CAREWSM baseline has been accepted.

TASK-001C removed obsolete WSM source, tracked old scripts and copied tests. TASK-001D removed 157 old config files, flattened 14 historical model files into passive `src/legacy/{audio_models,fusion_models,video_models,text_models}/`, and archived two callback sources under `src/legacy/callbacks/`. Direct package dependencies are now only `chimera-ml>=0.2.4`; old WSM methods remain inactive.

## Data roots

- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`

TASK-001A observations: fixed train/dev/test and five TRAIN CV folds were consistent and sample/profile-disjoint in both corpora; CARE `videos.csv` and `care_features.py` were absent; WSM manifest has 34 columns and per-task missing labels; 0/19 CARE feature families currently appear in WSM. The CARE `linguistic_turnlevel` declared-vs-observed feature dimension discrepancy remains unresolved.

## Evaluation policy

CV folds are used for development/hyperparameter selection. DEV is used for final model/configuration selection. TEST is visible and stored as descriptive/reporting evidence for completed DEV-selected checkpoints, but it must not influence model, checkpoint, threshold, hyperparameter, architecture, sweep, or retry decisions.

## Completed task

TASK-001A — read-only corpus/feature/repository audit; manager **PASS**. Evidence: `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`. Merged PR: https://github.com/suslyash/CAREWSM/pull/3 ; merge SHA: `4097bd52adc3d46551f2439dd24c040f06a55e47`. No models or TEST metrics produced.

## Next task

TASK-003A — standalone WSM eGeMAPS/DisVoice extraction for the first **7 of our 11 audio families**, strict WSM `feature_dir` and CARE numeric schema/metadata compatibility, and at most 3 actual TRAIN smoke outputs under WSM `extracted/`. Owner authorized bounded output writes; then TASK-003B deep audio (4 outputs) and TASK-003C audio-only resumable batch. Video/text features (8 of CARE's global 19) are externally provided and never extracted by this project. `docs/NEXT_TASK_EN.md` and `docs/FEATURE_PARITY_TARGET_EN.md` are source of truth.

**Latest TASK-002B owner correction:** Ubuntu reboot resolved NVIDIA driver/NVML mismatch. Cancel attempted CPU `torch==2.11.0+cpu`/`torchaudio==2.11.0+cpu` installation; preserve/recover verified matching CUDA-enabled PyTorch and torchaudio in the existing `.venv`, confirm `torch.cuda.is_available() == True` before bounded DiariZen GPU inference. CPU wheels, CPU fallback and additional environments are prohibited; `pyproject.toml` stays unchanged. Actual DiariZen results still pending.

**Latest TASK-002B workflow:** Owner will install dependencies and download weights personally. Codex is restricted to preparing an exact verified shell-command handoff and an **unexecuted** `scripts/audio/cache_diarizen_weights.py` helper plus bounded gate code/report. Await owner's GPU/CUDA dependencies and model-cache outputs before manager authorizes any DiariZen inference; no Codex installations/downloads/inference in preparation phase. Existing `.venv` only; no CPU-only Torch.

## TASK-001B outcome

Created the inactive `src/carewsm/` namespace, future `configs/carewsm/` root,
legacy-isolation documentation, and bounded scaffold tests. Existing WSM-derived
code, configurations, registrations, scripts, and tests remain untouched.
TASK-001B accepted and merged in PR #5: https://github.com/suslyash/CAREWSM/pull/5 ; merge SHA `b23b686fe4a143d0cd8804e232110ccb8731924a`. Codex reported standalone import PASS and `git diff --check` PASS; `pytest` was unavailable (`No module named pytest`), so a pytest run is not claimed. **Owner corrected the source layout afterward; TASK-001C removes the unwanted scaffold.** Extraction, DiariZen, training, evaluation and CARE/WSM writes were not performed.


## TASK-001C outcome

Removed obsolete WSM source, top-level scripts, copied tests, nested README files, and the TASK-001B scaffold. Archived only historical model implementations under passive `src/legacy/`; preserved `src/common/callbacks/wsm_summary_callback.py` and `src/common/callbacks/wsm_segment_callback.py` as non-activating future candidates. The ChimeraML plugin entry point is retained as a no-op, and the package identity is CAREWSM with archived legacy packages excluded from installation.

Future manager-approved training configurations must document `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `wsm_summary_callback`, `mlflow_logger`, and `console_file_logger`; `wsm_segment_metrics_callback` remains optional pending review. No tests were created or run. **Manager PASS**, merged PR: https://github.com/suslyash/CAREWSM/pull/9 ; merge SHA `d732b7a2e83f50fa1b06018e589cc36012797793`. Remaining nested legacy archive and old configs are the explicitly scoped TASK-001D follow-up.


## TASK-001D outcome

Removed all 157 tracked WSM-era configuration files. Flattened the passive archive into `src/legacy/audio_models/`, `fusion_models/`, `video_models/`, `text_models/`, and `callbacks/`, preserving all 14 model implementation files present in `origin/main` and both historical callback sources without behavior changes or activation; the task wording count of 15 does not match the audited inventory.

Reduced direct project dependencies to `chimera-ml>=0.2.4`; removed legacy Torch/audio/video/ML dependencies including `mamba-ssm`. Existing CAREWSM project identity, no-op plugin entrypoint, and `legacy*` package exclusion remain unchanged. TASK-001D manager **PASS**. Merged PR: https://github.com/suslyash/CAREWSM/pull/12 ; merge SHA `db92aa6eab4f10f4c141c6b730feeee7b0c679bc`. Codex reported `pip install -e . --dry-run` and exact `pip install -e .` PASS; manager independently checked Git changes and representative source blob preservation but did not execute pip on Codex's machine. No tests, extraction, DiariZen, evaluation or training were run.


## TASK-002A outcome

TASK-002A read-only WSM TRAIN audio preflight was **accepted (PR #15)**. A bounded deterministic sample of five existing TRAIN
media files was probed with `ffprobe` according to Codex evidence: all had audio/video streams, AAC audio,
16 kHz sample rate, two channels, and durations from 41.8 to 79.864 seconds.
The CARE 16 kHz mono, participant-turn, and representative eGeMAPS/DisVoice/
XLS-R/TRILLsson contracts are documented in
`docs/TASK-002A_WSM_AUDIO_PREFLIGHT_EN.md`. No feature parity claim is made;
extraction, DiariZen, training, evaluation, tests, and corpus writes were not
performed. Manager independently reviewed the code/report and accepted the preflight; merged PR: https://github.com/suslyash/CAREWSM/pull/15 ; merge SHA `5f97a73fb4bf6f52fb7b7320d794df6a8134b061`. Local external data roots were not independently accessed by the manager. The five samples are all depression/CONTROL and do not demonstrate task-balanced coverage.


## TASK-002B bounded GPU outcome (manager reviewed — bounded GPU execution accepted; turn identities unverified)

Owner-authorized local CUDA DiariZen runtime setup required a separate Python 3.10 `.venv-diarizen` because upstream packages conflicted with the CAREWSM `.venv`. The owner accepted this specific exception; both environments and downloaded weights remain untracked. According to the implementer's reported run, exactly three TRAIN clips spanning DEPRESSION, CONTROL and PARKINSON were converted to temporary 16 kHz mono WAV and processed with locally cached DiariZen weights on GPU: inference-only RTF 0.023261–0.028673, 21–24 diarized segments and exactly one speaker detected in each. Respondent-vs-interviewer labels remain **unverified**; output is runtime feasibility, not turn-quality validation or across-corpus parity. The WSM/CARE files were reported unchanged, scratch WAV/RTTM files cleaned, no TEST access, tests, feature extraction or training. Manager reviewed Git source/report but could not independently rerun the local data or GPU inference.


**TASK-002C manager outcome:** PASS for the documented manual listening checklist, merged PR https://github.com/suslyash/CAREWSM/pull/23 ; merge SHA `9629135645c8f10745e3537ad006c83fb27d69bb`. The owner accesses this machine via SSH and cannot currently hear audio, so the human one-vs-multiple-speaker and respondent-identity checks are DEFERRED; none are claimed as verified. This limitation does not stop supervised, explicit provisional feature extraction; keep semantic parity qualified.

**Latest owner scope — audio extraction only, plus audio/multimodal modelling:** our extraction target is **11 CARE audio features**, not all 19 registry families. TASK-003A immediately implements seven eGeMAPS/DisVoice families with a bounded three-TRAIN-file WSM smoke, exact CARE descriptor content/schema and WSM `feature_dir` paths; TASK-003B handles four deep audio outputs and TASK-003C the audio-only batch after manager approval. The other **8 nonaudio features (5 video, 3 linguistic) are supplied by other teams**; later multimodal models consume those artifacts, but we do not generate, alter or provision their extraction. No CARE/WSM raw/manifest writes, no unapproved bulk run, and all potentially large downloads remain owner-executed. Speaker identity remains unverified.

**Manager outcome, TASK-002B:** PASS for bounded GPU feasibility, merged PR https://github.com/suslyash/CAREWSM/pull/21 ; SHA `6c5056bfe6b65424d386c1dfe0872b160e2433be`. Measured run results were documented by Codex; the manager verified committed source/report and preserved newer manager contracts during the merge, without independently rerunning external data. The three files all received one detected speaker, so DiariZen remains provisional pending qualitative speaker sanity. The owner approved a separate `.venv-diarizen` to resolve real Python/runtime incompatibilities. **Owner's permanent traffic rule:** Codex must supply commands, sources, versions and expected/unknown download sizes for any potentially large dependencies or weights; owner executes those downloads, not Codex.

## TASK-002C outcome

TASK-002C prepared a bounded owner listening checklist for exactly the same
three TRAIN recordings used by TASK-002B. It provides three short audio-only
`ffplay` windows per recording, centered on previously observed DiariZen speech
segments, and a response table distinguishing audible speaker count from
respondent identifiability. The intervals are model-derived and not ground
truth. Manual owner confirmation is **pending**. No new media, downloads,
installations, inference reruns, tests, extraction, training, or dataset writes
were performed. Evidence: `docs/TASK-002C_SPEAKER_SANITY_EN.md`.

## TASK-003A outcome

TASK-003A corrected the standalone seven-family eGeMAPS/DisVoice extractor and read-only CARE audio parity validator, both under `scripts/audio/` and without ChimeraML. The deterministic three-TRAIN selection and manifest-field-derived `WSM_NEW/extracted/<label>/<subgroup>/PROFILE_<profile>/VIDEO_<video>` destinations were verified. The smoke reached local DiariZen but is **blocked by `torch 2.1.1` versus `numpy 2.2.6` ABI incompatibility** before descriptor output; no WSM feature files were written. CARE raw audio is absent from the available archives, so same-code numeric CARE extraction remains unrun; existing CARE CSV schema comparison is implemented. The four remaining audio families and all eight nonaudio families are outside this task. Evidence: `docs/TASK-003A_AUDIO_DESCRIPTOR_EXTRACTION_EN.md`.
