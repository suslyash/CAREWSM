# CAREWSM Research and Engineering Plan

Exact model families and missing-label methods are deliberately **not frozen** before the corpus/feature/code audit.

## Stage 0 — Repository bootstrap and legacy reset

**Status: COMPLETE pending manager review**

- remove old WSM task/progress/evidence/plan/paper artifacts;
- install CAREWSM manager/Codex contracts;
- preserve copied implementation temporarily for audit.

## Stage 1 — Corpus, feature, and legacy-code audit

**Status: NEXT**

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

After manager review, perform a separate cleanup task based on this inventory.

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
