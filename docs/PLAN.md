# CAREWSM Audio Research Plan

## Stage 0 — project contracts

**Status: COMPLETE**

- audio-only scope;
- manager/Codex workflow;
- fixed split discipline;
- joint CARE+WSM primary training protocol;
- two-head masked target contract;
- Mantis temporalization requirement;
- three-model shortlist;
- validation-only model selection;
- fold-wise CARE/WSM TEST statistics.

## Stage 1 — feature integrity and coverage audit

**Status: NEXT**

Codex audits the actual extracted audio feature families.

Outputs:

- exact operational registry;
- schemas/shapes/dtypes;
- timestamps/temporal semantics;
- valid/missing/corrupt/empty/non-finite status;
- per-sample availability manifest;
- per-feature and pairwise coverage;
- complete-case retention;
- split/fold/label sanity;
- confirmation that fold `k` can be combined safely across CARE and WSM.

No training or Mantis derivation.

## Stage 2 — joint Chimera-ML data layer

Implement:

- canonical CARE/WSM sample identity;
- joint fold construction `CARE_train_k + WSM_train_k`;
- joint validation construction `CARE_val_k + WSM_val_k`;
- separate CARE/WSM validation views for metrics;
- separate CARE/WSM TEST loaders;
- two target values + observed masks;
- per-feature availability;
- native temporal/static loaders;
- TRAIN-only normalization/PCA/class weights;
- leakage tests.

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

Build derived sequences without modifying source features.

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
- validation and TEST metrics/callbacks.

## Stage 5 — joint single-feature screening

Train every experiment on CARE+WSM jointly.

Evaluate:

- every usable single family;
- native representation;
- Mantis representation where valid;
- reference baseline;
- three research models.

Select/promote using validation only.

For each fold, also log CARE TEST and WSM TEST statistics.

## Stage 6 — joint feature combinations

Evaluate structured groups, then greedy forward additions.

Avoid exhaustive 2047-subset search.

Selection uses validation only.

## Stage 7 — joint multitask loss/balancing study

Compare:

- Equal;
- PCGrad;
- CAGrad;
- STCH;
- PAGB;
- single-task models;
- unweighted vs weighted BCE.

Use matched joint-fold settings.

## Stage 8 — Optuna

Tune only the best joint model/feature/loss configuration.

Objective: joint-fold validation only.

## Stage 9 — final joint CARE+WSM evidence

Report:

- validation results used for selection;
- fold-wise CARE TEST statistics;
- fold-wise WSM TEST statistics;
- metric-wise CARE/WSM Average;
- fold aggregate statistics;
- coverage;
- seeds;
- runtime/parameters;
- reproducibility metadata.

## Stage 10 — deferred secondary studies

Only after the primary joint model is complete:

- CARE-only / WSM-only ablations;
- CARE -> WSM / WSM -> CARE direct transfer;
- D-Vlog / YouTubePD external tests;
- small-target adaptation.

Multimodality remains out of scope.
