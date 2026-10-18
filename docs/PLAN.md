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

## Stage 2 — CARE-compatible WSM **audio feature extraction only**

**CURRENT STAGE — TASK-004A FULL-SEGMENT AUDIO RESET (HIGHEST OWNER PRIORITY):** The owner rejected using DiariZen at this stage. WSM `VIDEO_*` inputs already represent segments; the entire 16kHz mono audio of each clip is the extraction unit. Remove 11 old audio output filenames and stale DiariZen sidecars **ONLY** from WSM extracted directories after old production stops; preserve external modalities, transcripts and source datasets. Re-extract 11 families with one synthetic CARE-layout turn per clip and deliver a single owner-operated resumable background all-splits command. This overrides all historical DiariZen/timebase plans and prior audio-output preservation for the wrong-policy feature files. Only owner full run; Codex at most 3 TRAIN real smoke records, no tests or downloads.

**Status: TASK-003C CORRECTIVE PASS (PR #32 merged): 46 complete TRAIN records × 11 audio = 506 files; one Parkinson VUV/Praat error remains. OWNER OVERRIDE TASK-003D: one tightly scoped correction+attempt of that exact sample, plus a robust single `tmux` owner-run launcher across all WSM TRAIN/DEV/TEST. Codex itself must NOT launch full corpus or read DEV/TEST before owner starts it.** See `docs/FEATURE_PARITY_TARGET_EN.md` for exact filenames, widths, keys and source limitations.

- TASK-002A bounded media preflight accepted (PR #15). CARE references use 16 kHz mono and respondent-aligned turns; standalone extractors must live in `scripts/audio/`, not ChimeraML.
- TASK-002B DiariZen bounded GPU feasibility accepted (PR #21): one speaker detected in each of three clips, inference-only RTF 0.023261–0.028673; respondent identity **unverified**. The owner accepted the exceptional `.venv-diarizen` because of dependency incompatibility.
- TASK-002C playback/listening checklist accepted (PR #23) but **deferred** because the owner connects over SSH and cannot play the remote audio. This does not block clearly labeled provisional descriptor extraction.
- **TASK-003A REVIEW FIX IN PROGRESS:** Codex's first four-file commit prepared a dry-run-only implementation but actual extraction is blocked by owner-missing `opensmile`/`disvoice` and the newly observed mismatch between WSM legacy manifest `feature_dir` (`*/train_labels/*/features/*`) and canonical `extracted/` target. Manager rejected first-pass code pending safe validated canonical output derivation, correct DisVoice `extract_features_file(...,fmt='dataframe')`, reliable existing-RTTM participant turns, exact CARE header checks and atomic no-overwrite semantics. Do not edit manifests or write into legacy `feature_dir`; full run not authorized.
- **TASK-003A NOW:** implement `scripts/audio/wsm_extract_care_descriptors.py` for 7 CARE-compatible eGeMAPS/DisVoice outputs and `scripts/audio/validate_care_audio_parity.py`. Owner authorizes writing new feature files only for at most 3 TRAIN clips under the manifest-provided WSM `extracted/` path in this first smoke. Reuse the actual CARE extraction algorithms/schema and report unresolved participant alignment.
- **TASK-003A manager PASS, PR #27 merged:** the implementer-reported WSM TRAIN smoke produced **21 real CSVs (7 descriptor families × 3 clips)**; read-only audit checked all rows, CARE headers/order, metadata, dimensions and finite numeric values (four blanks documented), no retries or data overwrites. CARE original media are **unavailable** (owner confirmed), so numerical re-extraction/comparison is impossible; report semantic/numeric CARE parity as **UNVERIFIED**, not a prerequisite to remaining audio engineering. PR #27 merge SHA `4f6fadffc29ade18bc252dff43fd9a7d469d7f60`.
- **TASK-003B manager PASS — PR #30:** the implementer reported and validated four additional CARE-layout deep audio outputs for each of the same three TRAIN recordings: XLS-R, TRILLsson, emotion MSP trajectories and embeds. Together with TASK-003A that is 11 audio families × 3 TRAIN = 33 files. Merge commit `595d4ebd871a36592e395320a4d5c982aee5a027`. Numeric CARE parity remains UNVERIFIABLE without original CARE audio and respondent identities remain provisional.
- **TASK-003C corrective PASS (PR #32):** fixed task-wide cap/ledger, stable manifest-order selection, cross-family NPZ/CSV/sidecar turn checks, per-record scratch and a single worker DiariZen pipeline, documented Praat exit codes. Six of seven explicit previously blocked samples recovered, so cumulative completed TRAIN rows rose to **46 / 506 validated audio files**; one Parkinson sample is still VUV-blocked. Prior 30-limit overrun was not authorized retrospectively. No further retry of the remaining blocked ID without separate decision.
- **TASK-003D owner override NOW:** repair root cause of the **one** previously blocked Parkinson TRAIN recording and run **at most one** algorithm-preserving, exact-ID bounded retry; no other Codex output writes. Implement one owner-operated `tmux` launcher that performs internally chunked/resumable complete AUDIO feature extraction across **TRAIN→DEV→TEST** using no-clobber, separate production ledger, validated stable manifest ordering, all 11 families, offline local models and explicit status for failures. Codex only reads TRAIN/preflights and supplies tested-usage single background command, **never** launches the full extraction. Owner personally runs it after manager review. The observed full-record pilot mean 527.8 s, peak RSS ~15.2 GiB and CUDA 4.98 GiB inform uncertain resource estimates; no exact run deadline promised.
- **Not our extraction scope:** other teams produce CARE-compatible video (five) and linguistic (three) features. We do not implement `scripts/video/` or `scripts/text/` extractors or acquire those model weights. Integration/masking of externally supplied features belongs to later multimodal modelling.
- **Permanent owner traffic rule:** potentially large packages, GPU framework wheels and model weights are never downloaded by Codex. Codex gives exact commands, official sources, revisions, size estimates or unknown-size warnings and cache paths; the owner downloads personally.

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

- integrate available **externally supplied** video/text feature artifacts from other teams (we do **not** extract them), with explicit provenance and compatible input schema;
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
