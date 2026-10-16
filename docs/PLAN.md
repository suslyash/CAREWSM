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

## Stage 1C — Active packaging and plugin isolation

**Status: NEXT — TASK-001C**

- identify the installed project and active Chimera plugin as CAREWSM, not WSM;
- expose a minimal empty CAREWSM registration entrypoint while no CAREWSM models/DataModules are authorized;
- preserve old WSM implementation, scripts, configurations, and historical registration module untouched as legacy reference;
- include standard-library-only packaging and import verification, without installing/training/building heavyweight dependencies;
- do not relocate, delete or execute legacy code, or modify corpora.

## Stage 2 — WSM feature extraction parity

- adapt CARE extraction logic for WSM outside Chimera ML;
- run a small extraction smoke before full processing;
- run the small DiariZen runtime/sanity gate;
- manager/owner chooses retain or drop diarization;
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
