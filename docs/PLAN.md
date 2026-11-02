# CAREWSM Audio Research Plan

## Stage 0 — project contracts

**Status: COMPLETE**

- audio-only scope;
- manager/Codex workflow;
- fixed split discipline;
- two-head masked target contract;
- Mantis temporalization requirement;
- three-model shortlist;
- cross-corpus programme.

## Stage 1 — feature integrity and coverage audit

**Status: NEXT**

Codex audits the actual 11 owner-reported extracted audio feature families.

Outputs:

- exact operational registry;
- schemas/shapes/dtypes;
- timestamps/temporal semantics;
- valid/missing/corrupt/empty/non-finite status;
- per-sample availability manifest;
- per-feature and pairwise coverage;
- complete-case retention;
- split/fold/label sanity.

No training or Mantis derivation.

## Stage 2 — Chimera-ML data layer

Implement:

- canonical CARE/WSM sample identity;
- split/fold loading;
- two target values + observed masks;
- per-feature availability;
- native temporal/static loaders;
- normalization fit on training data only;
- unit tests preventing leakage.

All active code under `src/audio/data` and `src/audio/features`.

## Stage 3 — Mantis temporal representation

Freeze:

- supported feature blocks;
- 1.0 s / 0.5 s window construction;
- Mantis checkpoint/version;
- selected layer;
- pooling;
- validity/confidence semantics;
- scaler/PCA contract;
- cache schema/fingerprint.

Build derived sequences without modifying source extracted features.

## Stage 4 — model/loss infrastructure

Implement:

- pooled reference baseline;
- TSCMamba-inspired;
- MPTSNet-inspired;
- MD-Former-inspired;
- shared/common representation;
- task-specific Depression/Parkinson feature layers;
- binary heads;
- masked BCE;
- weighted BCE;
- Equal/PCGrad/CAGrad/STCH/PAGB hooks;
- required metrics/callbacks.

## Stage 5 — within-corpus feature screening

CARE and WSM separately.

Evaluate:

- every usable single family;
- native representation;
- Mantis representation where valid;
- simple baseline;
- three research models under standardized budget.

## Stage 6 — feature combinations

Evaluate structured groups, then greedy forward additions.

Avoid exhaustive 2047-subset search.

Re-compare best few combinations across all three models.

## Stage 7 — multitask loss/balancing study

Compare:

- Equal;
- PCGrad;
- CAGrad;
- STCH;
- PAGB;
- single-task models;
- unweighted vs weighted BCE where appropriate.

Use matched feature/model settings.

## Stage 8 — within-corpus final models

Finalize CARE and WSM separately using TRAIN CV + DEV.

No TEST-driven changes.

## Stage 9 — cross-corpus direct transfer

- CARE -> WSM;
- WSM -> CARE.

No target refitting/tuning.

## Stage 10 — joint CARE + WSM training

Train jointly; evaluate separately on each corpus.

## Stage 11 — external tests

- D-Vlog for Depression;
- YouTubePD for Parkinson.

Requires verified dataset roots/protocols.

## Stage 12 — small-target adaptation

Use only a pre-declared labeled fraction of target TRAIN.

Freeze fractions, participant sampling and update policy before running.

## Stage 13 — Optuna

Tune only the winning model/feature/loss configuration.

TRAIN CV objective only.

## Stage 14 — final evidence

Report:

- within-corpus;
- direct transfer;
- joint training;
- adaptation;
- external tests;
- required metrics;
- coverage;
- seeds;
- runtime/parameters;
- exact reproducibility metadata.

Multimodality remains out of scope.
