# CAREWSM Research and Engineering Plan

Exact model families and missing-label methods remain **unfrozen** after TASK-001A. Audit findings must be translated into validated data/feature contracts before model implementation.

## Stage 0 — Repository bootstrap and legacy reset

**Status: COMPLETE (bootstrap PR #1)**

- remove old WSM task/progress/evidence/plan/paper artifacts;
- install CAREWSM manager/Codex contracts;
- preserve copied implementation temporarily for audit.

## Stage 1 — Corpus, feature, and legacy-code audit

**Status: COMPLETE — TASK-001A manager PASS, PR #3 merged**

- inspect CARE/WSM split and fold CSVs;
- verify participant/profile disjointness;
- inventory exact label columns, values, missingness, and task availability;
- inspect CARE `features.json`, `care_features.py`, and extraction scripts;
- sample representative CARE feature files for schema/dtype/shape/storage;
- inventory current WSM `extracted/` coverage and raw source organization;
- inventory repository `src/`, `configs/`, `tests/`, retained `scripts/`, and packaging;
- classify old components as KEEP / ADAPT / LEGACY / DELETE;
- identify generic callbacks/losses/metrics/orchestration worth retaining;
- identify model families worth preserving under a separated legacy namespace;
- no training and no dataset writes.

Audit accepted with explicit limitations (missing CARE loader, CARE `videos.csv`, and `linguistic_turnlevel` registry width mismatch). WSM has 0/19 CARE feature families; missing disease labels are masked. Proceed to an independent, narrow repository-cleanup/scaffolding task; do not start extraction/training.

## Stage 1B — Controlled repository scaffolding and legacy isolation

**Status: COMPLETE — TASK-001B manager PASS, PR #5 merged**

- introduce a minimal active CAREWSM namespace without activating copied WSM methods;
- document legacy isolation and new config/test roots without moving or deleting old implementation;
- preserve existing Chimera registration/packaging until a later explicit task;
- no corpus writes, extraction, training, evaluation, or running DiariZen.

## Stage 1C — WSM source and scripts cleanup, model archiving

**Status: COMPLETE — TASK-001C manager PASS, PR #9 merged**

- Removed obsolete WSM pipeline modules, all tracked old scripts and historical tests; preserved **14** historical model implementations in passive `src/legacy/` (corrected after task-base inventory).
- Preserved `wsm_summary_callback` and candidate `wsm_segment_metrics_callback` source without active plugin registration.
- Set project identity CAREWSM and no-op ChimeraML plugin; no new models, extraction, training, tests or datasets changed.
- Source archival hierarchy and legacy configs need one further owner-requested organization cleanup.

## Stage 1D — Legacy archive flattening and stale-config removal

**Status: COMPLETE — TASK-001D manager PASS, PR #12 merged**

- Removed all 157 tracked old WSM-era `configs/`, without inventing or activating new training configs.
- Flattened the **14** historical model implementations from `src/legacy/<domain>/models/` into `src/legacy/{audio_models,fusion_models,video_models,text_models}/`.
- Moved both historical summary and segment-metrics callbacks into `src/legacy/callbacks/` as passive source, unchanged and without plugin activation.
- For later model training, recover/adapt `wsm_summary_callback` as an active ChimeraML callback and use `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `mlflow_logger` and `console_file_logger`. Treat `wsm_segment_metrics_callback` as optional pending contract review.
- Removed obsolete direct WSM package dependencies from `pyproject.toml` including `mamba-ssm`; the remaining project direct dependency is `chimera-ml>=0.2.4`. Codex reports bounded dry-run and exact editable install PASS; independent manager verification covers Git content, not the implementer's local pip environment.
- No tests, extraction, diarization, model loading, TEST evaluation, training or dataset writes.

## Stage 2 — WSM feature extraction parity

**Status: PRIORITY — TASK-003A actual CARE-compatible WSM feature extraction; TASK-002C owner listening DEFERRED.** Target all 19 registry families; begin real WSM TRAIN production on the seven eGeMAPS/DisVoice descriptor outputs with strict schema checks. See `docs/FEATURE_PARITY_TARGET_EN.md`.

- TASK-002A verified bounded WSM TRAIN media and CARE reference script/schema contracts (report in `docs/TASK-002A_WSM_AUDIO_PREFLIGHT_EN.md`), without extraction.
- TASK-002A added `scripts/audio/wsm_audio_preflight.py` and evidence report, independently of ChimeraML. All future standalone audio preparation scripts remain under `scripts/audio/`.
- NEXT TASK-002B manual-setup gate: Codex first drafts verified CUDA DiariZen dependency installation commands plus `scripts/audio/cache_diarizen_weights.py`, without executing any install/download or model inference. The owner executes setup/weight-caching commands directly in the existing `.venv`, shares results, and the manager then separately authorizes the bounded 3-file GPU runtime.
- NEXT TASK-002B GPU correction: the NVIDIA driver/NVML mismatch was resolved by reboot; **use CUDA-enabled GPU `torch`/`torchaudio` in the existing project `.venv` only**, never CPU wheels or CPU inference, and check `torch.cuda.is_available()` after resolving wheel compatibility. Stop and report if GPU runtime remains blocked.
- NEXT TASK-002B: **owner approved installation of all required DiariZen inference packages (including `tqdm`, `torchaudio`, compatible `torch`, other upstream dependencies) plus official weights download and caching** (use **only the existing CAREWSM `.venv`**, no new virtualenvs; install required packages there and verify compatible `torch`/`torchaudio`), then run at most three short WSM TRAIN recordings through the bounded mono-16 kHz diarization gate. Preserve original data, caches outside Git and datasets; report runtime/speaker-turn evidence, or exact unresolved setup blocker. The existing first-pass branch requires a real inference path before merge;
- TASK-002B runtime gate accepted provisionally: three TRAIN files had inference-only RTF 0.023261–0.028673, 21–24 segments each, and one speaker per recording; voice identity not verified. The separate `.venv-diarizen` was required by dependency incompatibility and approved by the owner.
- TASK-002C documentation was accepted and merged (PR #23); the owner connects by SSH and cannot listen now. Human participant-turn confirmation is DEFERRED rather than a blocker to engineering the extractor. Unverified speaker identity remains a documented scientific limitation.
- Owner alone runs potentially large dependency/weight downloads after Codex provides verified commands, estimated transfer sizes and cache details; no hidden downloads or new environments without approval.
- **TASK-003A NOW:** implement standalone `scripts/audio/wsm_extract_care_descriptors.py` for seven CARE eGeMAPS/DisVoice families, with manifest `video_path`/`feature_dir` mapping, provisional speaker/turn provenance, read-only CARE schema comparison and at most 3 genuine TRAIN outputs in WSM `extracted/`. Owner specifically authorizes this narrow feature-output write; no full run before manager acceptance.
- **TASK-003B/003C/003D NEXT IN PARALLEL MODALITY SEQUENCE:** deep audio 4, video 5, linguistic 3 (when observed transcripts/weights available). Match exact registered files, shape/key semantics and producer logic; never fabricate missing features. See `docs/FEATURE_PARITY_TARGET_EN.md`.
- **TASK-003E THEN:** resume batch extraction over fixed WSM TRAIN/DEV/TEST inputs, track per-family coverage/schema/status; TEST preprocessing does not read labels or metrics.
- adapt CARE extraction logic for WSM outside Chimera ML;
- run a small extraction smoke only after separate authorization, before full processing;
- extract the full required feature family;
- write CARE-compatible WSM feature files/layout;
- validate coverage, schemas, fingerprints, and failures.

## Stage 3 — Unified CAREWSM data layer

- define canonical CARE+WSM manifest/schema;
- preserve fixed train/dev/test and five TRAIN CV folds;
- implement masked label availability;
- implement modality availability;
- build Chimera DataModule/collate on extracted features;
- verify participant leakage, counts, IDs, labels, and missing-feature handling.

## Stage 4 — Audio baselines

- establish simple reproducible baselines from extracted audio features;
- compare feature families/representations under a controlled budget;
- use CV for development/hyperparameters;
- use DEV for final selection;
- store TEST descriptively for DEV-selected checkpoints.

## Stage 5 — Multimodal baselines

- choose non-audio modalities from the audited CARE inventory and reproducible WSM coverage;
- build simple fusion baselines before specialized architectures;
- preserve explicit missing-modality masks;
- keep evaluation/selection policy unchanged.

## Stage 6 — Joint multitask/multilabel modelling

- establish sparse/masked two-head baseline;
- study shared versus task-specific representations;
- introduce pseudo-labeling, reliability mechanisms, or advanced MTL only if the observed label-missingness problem and manager-approved evidence justify them;
- legacy WSM models are references/comparators, never automatic defaults.

## Stage 7 — Controlled tuning and ablations

- CV-based hyperparameter development;
- DEV final model selection;
- modality/architecture/MTL/missing-label ablations;
- calibration and negative-transfer diagnostics where relevant;
- TEST remains report-only.

## Stage 8 — Final evidence and report

- lock configurations from CV/DEV decisions;
- aggregate multi-seed evidence;
- preserve already-recorded TEST results rather than repeatedly reevaluating them;
- build claim/evidence traceability;
- produce the final report/paper artifacts.
