# CAREWSM Audio Experiment Protocol

## Selection hierarchy

TRAIN CV is used for development and hyperparameter/feature selection.

DEV is used to select the final feature combination and model configuration.

TEST is report-only and must not influence any development decision.

## Metrics

Pending owner override, the default primary score is:

`mean_macro_f1 = mean(Depression macro-F1, Parkinson macro-F1)`

computed only over observed labels for each task.

Secondary metrics:

- per-task macro-F1;
- AUROC;
- AUPRC;
- balanced accuracy;
- sensitivity/specificity;
- per-corpus metrics for CARE and WSM;
- coverage/retained sample count.

Threshold tuning, if used, must be based on TRAIN-CV/DEV only.

## Efficient experiment funnel

The project must explore many features without brute-forcing all `2^11 - 1` subsets.

### Phase A — data/feature audit

No training.

Freeze exact feature registry, availability and filtering behavior.

### Phase B — infrastructure + smoke

Implement the Chimera data layer, pooled MLP sanity baseline and the three research models.

Run tiny smoke tests only.

### Phase C — single-family screen

For every valid feature family:

- pooled MLP reference;
- TSCMamba-inspired;
- MPTSNet-inspired;
- MD-Former-inspired.

Use a cheap standardized budget first. Promote only functioning/reasonable candidates to full 5-fold CV.

This identifies whether temporal modelling actually helps each family.

### Phase D — structured combinations

Do not enumerate all subsets.

Evaluate scientifically meaningful groups discovered from the audit, for example:

- handcrafted temporal;
- handcrafted aggregated/static;
- deep speech representations;
- emotion-related;
- all temporal;
- temporal + complementary static;
- all valid families.

Exact group membership is defined from TASK-001 schemas, not assumed here.

### Phase E — greedy combination search

On the best one or two model families:

1. start from the strongest single/group feature set;
2. add one candidate family at a time;
3. keep additions that improve CV materially and consistently;
4. stop when no addition helps or the configured evaluation budget is exhausted.

Record every rejected addition.

### Phase F — final architecture comparison

Re-run the best few feature combinations on all three research models with full CV and multiple seeds as budget allows.

DEV chooses the final model/configuration.

### Phase G — Optuna

Only the winning model + feature combination enters Optuna.

Optuna objective uses TRAIN CV, never TEST.

Search space must be narrow and architecture-aware rather than a broad blind sweep.

### Phase H — final reporting

Evaluate the DEV-selected checkpoint/configuration on TEST once and persist the result.

Report:

- feature coverage;
- CV;
- DEV;
- TEST;
- per-task;
- per-corpus;
- seed variability;
- parameter count/runtime where practical.

## Reproducibility

Every experiment config must explicitly name:

- feature families;
- availability policy;
- model;
- seed;
- fold;
- learning rate;
- regularization;
- batch size;
- max epochs;
- early stopping;
- primary monitor.

Use MLflow through Chimera-ML for experiment tracking.
