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

## Stage 6 — joint feature combinations

**Status: IN PROGRESS**

### Stage 6A — multi-feature static fusion infrastructure

Next: TASK-011.

Implement a clean ordered multi-family T=1 concatenation path, coverage filtering and smoke validation.

### Stage 6B — bounded deep-feature combination matrix

After TASK-011 review, run a manager-defined set such as:

- emotion + TRILLsson;
- emotion + XLS-R;
- TRILLsson + XLS-R;
- emotion + TRILLsson + XLS-R;
- one complementary handcrafted addition if coverage permits.

Selection remains validation-only.

### Stage 6C — mixed static + temporal combinations

Only after the static deep fusion study shows what signal remains to gain.

Do not introduce fake temporal structure for static embeddings.

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
