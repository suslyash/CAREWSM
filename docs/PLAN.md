# CAREWSM Audio Research Plan

## Stage 0 — documentation and operating contract

**Status: IN PROGRESS**

- establish manager/Codex workflow;
- define audio-only scope;
- define feature-audit requirement;
- define experiment funnel;
- select three recent temporal model families.

## Stage 1 — feature extraction audit and filtering metadata

**Next**

Codex must inspect the actual 11 extracted feature families.

Outputs:

- exact feature registry;
- schema/shape/dtype inventory;
- temporal/static classification;
- per-corpus/per-split coverage;
- missing/corrupt/empty/non-finite/schema-failed counts;
- sample-feature availability manifest;
- retention estimates for complete-case and per-feature/per-combination filtering;
- split/label sanity checks;
- no training.

Priority: complete quickly and produce machine-readable results that unblock modelling.

## Stage 2 — Chimera-ML audio data layer

- canonical sample identity across CARE and WSM;
- label masks;
- feature-availability masks;
- family-specific loading;
- variable-length masks;
- normalization fitted only on TRAIN;
- fold-aware DataModule;
- per-corpus metadata;
- strict leakage tests.

All active code under `src/audio/data` and `src/audio/features`.

## Stage 3 — model infrastructure

Implement:

- pooled MLP reference;
- TSCMamba-inspired;
- MPTSNet-inspired;
- MD-Former-inspired;
- two-head masked classifier contract;
- common model output/metric interface.

All active code under `src/audio/models`, `loss`, `metrics`, and `callbacks`.

## Stage 4 — single-feature benchmark

Run every valid feature family through:

- pooled MLP;
- each of the three research models.

Use a cheap screening budget first, then full CV for promoted candidates.

Produce comparable coverage + performance tables.

## Stage 5 — structured feature combinations

Test meaningful feature groups derived from Stage-1 schemas.

Prioritize temporal information while retaining complementary static descriptors.

Avoid exhaustive 2047-subset brute force.

## Stage 6 — greedy feature search and final model comparison

Use the best one/two architectures for forward feature addition.

Then compare the best few combinations across all three research models under full CV.

DEV selects the final combination/model.

## Stage 7 — Optuna refinement

Tune only the winning model + feature combination.

Use TRAIN CV objective.

Search a narrow set of model/optimizer/regularization parameters.

No TEST feedback.

## Stage 8 — final evidence

- multi-seed confirmation;
- DEV-selected final checkpoint/config;
- one report-only TEST evaluation;
- per-task/per-corpus metrics;
- ablation/evidence tables;
- reproducibility summary.

Multimodality remains out of scope until the owner starts a new stage.
