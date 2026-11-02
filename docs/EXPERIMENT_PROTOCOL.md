# CAREWSM Audio Experiment Protocol

## Primary protocol: joint CARE + WSM from the start

All main model/feature experiments use joint CARE+WSM training.

For CV fold `k`:

- `TRAIN_k = CARE fold_k_train ∪ WSM fold_k_train`;
- `VAL_k = CARE fold_k_val ∪ WSM fold_k_val`.

The two corpora keep their own sample/profile identities and fixed partition membership; only the authorized fold training/validation portions are concatenated.

Missing Depression/Parkinson labels remain masked.

Preprocessing, PCA/scalers and `pos_weight` are fitted from `TRAIN_k` only.

The fixed DEV split is not silently added to TRAIN/VAL. Its final role is deferred until the main fold pipeline is stable.

## Model selection

Checkpoint selection, early stopping, feature promotion, model comparison, loss-method comparison and Optuna decisions use **validation only**.

TEST may be evaluated and logged after every fold, but TEST is never a selection signal.

## Validation reporting table

For each Configuration × Task × Fold, log three validation blocks:

- CARE validation;
- WSM validation;
- Average.

Each block contains:

- Macro F1;
- Weighted F1;
- Macro Recall;
- Weighted Recall;
- Patient Recall;
- Healthy Recall.

Definitions:

- Patient Recall = positive/patient class recall;
- Healthy Recall = negative/healthy class recall;
- Macro Recall = mean of Patient Recall and Healthy Recall (UAR);
- Weighted Recall = support-weighted class recall;
- Average metric = arithmetic mean of CARE and WSM values for that metric.

Do not pool predictions across corpora to obtain Average.

The validation table is the scientific decision table.

## TEST reporting table

For the fold-selected checkpoint, evaluate both fixed TEST sets:

- CARE TEST;
- WSM TEST.

Log the same six metrics and the same metric-wise CARE/WSM Average.

TEST values are statistics/evidence only. They must not determine:

- next feature combination;
- architecture;
- loss method;
- checkpoint rule;
- threshold tuning;
- Optuna trial selection;
- retry/continuation of an ablation.

## Stage A — fast integrity audit

No model training.

Freeze:

- exact operational feature registry;
- schemas;
- temporal semantics;
- validity statuses;
- per-sample availability manifest;
- pairwise/complete-case coverage;
- joint-fold compatibility.

## Stage B — joint Chimera-ML data layer

Implement a single fold-aware DataModule that:

- loads CARE and WSM together;
- constructs joint TRAIN/VAL for the requested fold;
- exposes corpus identity for separate metric reporting;
- masks unavailable task labels;
- filters only by features required by the active experiment;
- keeps CARE/WSM TEST loaders separate;
- fits no preprocessing outside TRAIN.

## Stage C — Mantis temporalization contract

Freeze:

- which feature blocks have a defensible 1.0 s / 0.5 s temporalization;
- exact Mantis checkpoint/version;
- selected layer;
- pooling;
- validity/confidence semantics;
- scaler/PCA contract;
- cache schema/fingerprint.

Mantis remains frozen.

## Stage D — joint single-feature screening

For every usable feature family/representation, train jointly on CARE+WSM.

Compare:

- native representation where meaningful;
- Mantis representation where meaningful;
- pooled reference classifier;
- TSCMamba-inspired;
- MPTSNet-inspired;
- MD-Former-inspired.

Use a standardized cheap budget first.

Promote/reject candidates from validation only.

For every fold/configuration also persist TEST statistics on CARE and WSM.

## Stage E — structured feature groups

Still under joint CARE+WSM training, evaluate scientifically meaningful groups derived from TASK-001:

- handcrafted temporal;
- handcrafted aggregated;
- deep speech;
- emotion;
- Mantis-derived temporal;
- temporal + complementary static;
- all high-coverage families.

Avoid exhaustive `2^11 - 1` enumeration.

## Stage F — greedy feature combination search

On the strongest one/two model families:

1. start from the strongest validation feature set;
2. add one candidate family/block at a time;
3. keep additions with stable validation improvement;
4. record rejected additions;
5. stop when additions cease to help or the budget is exhausted.

TEST never controls this search.

Re-compare the best few feature combinations on all three research models.

## Stage G — multitask loss study

On strong joint configurations compare:

- unweighted BCE;
- weighted BCE;
- Equal 0.5/0.5;
- PCGrad;
- CAGrad;
- STCH;
- PAGB;
- Depression-only;
- Parkinson-only.

Use the same fold/data/features/model budget whenever possible.

Selection comes from validation.

## Stage H — Optuna

Only the winning joint model + feature combination + loss family enters the primary Optuna study.

Optuna objective uses validation from the five joint folds.

No TEST/external metric enters trial selection.

## Stage I — final joint model evidence

Consolidate the selected joint configuration across folds.

Report:

- fold-wise validation table;
- fold-wise CARE TEST table;
- fold-wise WSM TEST table;
- fold-wise metric-wise Average;
- aggregate fold statistics;
- per-task results;
- feature coverage;
- seed variability;
- runtime/parameter footprint where practical.

## Deferred secondary studies

Only after the primary joint CARE+WSM model is established:

1. CARE-only / WSM-only ablations;
2. direct transfer CARE -> WSM and WSM -> CARE;
3. external D-Vlog and YouTubePD evaluation;
4. small-target adaptation.

These are not part of the initial model-selection loop.

## MLflow / reproducibility

Every run must record:

- Git commit;
- resolved config;
- fold;
- seed;
- joint CARE+WSM protocol;
- feature families;
- native/Mantis path;
- Mantis checkpoint/layer/pooling if used;
- availability policy;
- scaler/PCA artifact identity;
- model;
- loss/balancing method;
- validation metrics;
- CARE TEST metrics;
- WSM TEST metrics;
- checkpoint;
- MLflow run ID.
