# DEEP008 — Task-Attention Five-Fold Preparation

Date: 2026-11-08

## Preparation status

This document pre-registers the future validation comparison. No CUDA
training, inference, TEST access, or real-data execution was performed for
this preparation task.

Five task-attention YAMLs were created by copying the historical weighted-F
fold configurations. The only scientific model change is:

```yaml
model:
  name: carewsm_gated_task_attention
```

The task-attention files retain each corresponding historical fold's exact
TRAIN-derived Parkinson positive weight. This preserves strict semantic parity
with the existing weighted-F controls, including their fold-specific weights.
All other data, loss, optimizer, regularization, fold, seed, monitor, logging
and metric settings are unchanged apart from isolated task-attention run and
artifact namespaces.

## Future VAL-only comparison

Baseline: existing weighted-F five-fold validation UAR
`0.8271 +/- 0.0279`.

Current best observed reference: DB-MTL five-fold validation UAR
`0.8317 +/- 0.0371`. This is contextual evidence only and is not a second
control for the task-attention comparison.

The future comparison must report, for each fold and each arm:

- CARE Depression Macro Recall;
- CARE Parkinson Macro Recall;
- WSM Depression Macro Recall;
- WSM Parkinson Macro Recall;
- Patient Recall and Healthy Recall for each component;
- paired task-attention minus weighted-F fold differences;
- five-fold mean and sample standard deviation;
- canonical overall VAL UAR.

The provisional VAL-only acceptance rule is:

1. mean VAL UAR improvement of at least `+0.010` over weighted-F;
2. no deterioration greater than `0.020` absolute in any of the four mean
   corpus/task Macro Recall components;
3. explicit interpretation of CARE results, because valid WSM records have
   T=1 and cannot provide turn-order variation.

These thresholds are decision criteria, not statistical significance claims.
TEST must remain uninspected for architecture selection and cannot change the
acceptance decision.

## Execution gate

The prepared launcher is sequential, deterministic and collision-safe, but is
not authorized for execution in this task. Future execution requires explicit
owner approval after manager review of this branch.
