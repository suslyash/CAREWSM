# CAREWSM Audio Research Plan

## Stage 0 — project contracts

**Status: COMPLETE**

## Stage 1 — feature integrity and coverage audit

**Status: COMPLETE**

## Stage 2 — joint Chimera-ML data layer

**Status: COMPLETE FOR SINGLE-FEATURE NATIVE INPUT**

## Stage 3 — Mantis temporal representation

**Status: DEFERRED**

Mantis remains blocked/unfrozen and is not required for the current static deep-feature combination study.

## Stage 4 — model/loss infrastructure

**Status: COMPLETE FOR CURRENT SCREENING**

Implemented/usable:

- pooled MLP;
- MPTSNet-inspired;
- MD-Former-inspired;
- TSCMamba-inspired code path remains dependency-blocked by `mamba_ssm`;
- masked multitask BCE;
- six required metrics;
- VAL-only checkpointing;
- CARE/WSM TEST statistics.

## Stage 5 — joint single-feature screening

**Status: COMPLETE**

Manager-reviewed matrix:

- 22 configurations;
- 110 fold-runs.

Primary VAL ranking is recorded in the manager report.

Strongest feature families:

1. `emotionmsp_embeds`;
2. `trillsson4_1`;
3. `wav2vec_xlsr300m`.

## Refactor gate — documentation and implementation contract

**Status: ACTIVE BEFORE FURTHER REFACTORING**

REFACTOR-000 establishes the simplicity, two-file data-layer, corpus-specific logging, summary, monitor, path, and cleanup contracts. REFACTOR-001 through REFACTOR-005 must be completed and independently reviewed in order before TASK-011 resumes.

The former Stage 6 feature-combination work is paused. No TASK-011 matrix or new feature-combination experiment is authorized by this document.

## Stage 7 — multitask loss/balancing study

**Status: DEFERRED**

Compare Equal / weighted BCE / PCGrad / CAGrad / STCH / PAGB only on strong feature/model configurations.

## Stage 8 — Optuna

**Status: DEFERRED**

Tune only the final winning model/feature/loss family.

## Stage 9 — final joint CARE+WSM evidence

**Status: DEFERRED**

## Stage 10 — secondary studies

**Status: DEFERRED**

- CARE-only / WSM-only ablations;
- direct transfer;
- D-Vlog / YouTubePD;
- small-target adaptation.

Multimodality remains out of scope.
