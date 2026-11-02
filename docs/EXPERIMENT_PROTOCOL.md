# CAREWSM Audio Experiment Protocol

## Selection hierarchy

TRAIN CV drives development, feature/model comparison and hyperparameter search.

DEV is used for configuration selection.

For **every CV fold and every evaluated configuration**, the fold checkpoint is also evaluated on both fixed TEST sets:

- CARE TEST;
- WSM TEST.

TEST metrics are always logged, but they are never a development/selection signal.

External D-Vlog/YouTubePD are deferred until the final external-evaluation stage and are report-only unless the owner explicitly defines another protocol.

## Required metric table

For each Configuration × Task × Fold, log three blocks: CARE, WSM and Average.

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

The Average block is not computed by pooling predictions across corpora.

Every table also includes:

- number of evaluated samples;
- number of observed labels;
- feature coverage/retention;
- corpus/domain;
- fold identifier.

The manager may define a scalar TRAIN-CV/DEV selection score after TASK-001/Stage-2 confirms label distributions. TEST metrics must never be used for that choice.

## Stage A — fast integrity audit

No model training.

Freeze:

- exact 11-family operational registry;
- schemas;
- timestamps/temporal semantics;
- validity statuses;
- per-sample availability manifest;
- coverage and filtering rules.

## Stage B — data layer and Mantis temporalization contract

Implement the Chimera-ML DataModule and leak-free preprocessing.

Then freeze:

- per-family native representation loader;
- which feature families support 1.0 s / 0.5 s windowization;
- exact Mantis checkpoint/version;
- selected layer;
- pooling rule;
- scaler/PCA rules;
- artifact schema and cache identity.

Run only small derivation/smoke checks before manager review.

## Stage C — reference baselines

Within CARE and WSM separately:

- single-feature native baselines;
- single-feature Mantis baselines where valid;
- pooled/simple reference classifier.

Purpose: detect broken representations and establish whether Mantis temporalization adds signal.

## Stage D — three-model single-feature screen

For each promoted representation:

- TSCMamba-inspired;
- MPTSNet-inspired;
- MD-Former-inspired.

Use a standardized cheap screening budget.

Promote candidates based on TRAIN CV consistency and runtime/coverage, not TEST.

## Stage E — structured feature groups

Build scientifically meaningful blocks from observed schemas, for example:

- handcrafted temporal;
- handcrafted aggregated;
- deep speech;
- emotion;
- Mantis-derived temporal;
- temporal + complementary static;
- all high-coverage families.

Exact membership is defined after TASK-001.

Avoid exhaustive `2^11 - 1` subset enumeration.

## Stage F — greedy combination search

On the strongest one/two model families:

1. start from strongest single/group representation;
2. add one candidate family/block at a time;
3. retain additions with stable CV improvement;
4. record rejected additions;
5. stop when additions cease to help or budget is exhausted.

Then re-evaluate the best few combinations on all three research models.

## Stage G — loss / multitask study

For the best representative configurations compare:

- unweighted BCE;
- weighted BCE;
- Equal 0.5/0.5;
- PCGrad;
- CAGrad;
- STCH;
- PAGB;
- Depression-only;
- Parkinson-only.

Class weights use known TRAIN labels only.

Loss-method comparison must not be confounded by changing feature sets unless the experiment explicitly studies an interaction.

## Stage H — within-corpus finalization

Finalize CARE and WSM baselines separately using TRAIN CV and DEV for decisions.

For every fold checkpoint, evaluate and persist the full required TEST table on **both CARE and WSM**.

Persist:

- best feature combination;
- native vs Mantis path;
- model;
- loss method;
- preprocessing;
- fold-wise CARE TEST metrics;
- fold-wise WSM TEST metrics;
- fold-wise Average metrics;
- seed results.

Do not use fold-level TEST results to promote/reject configurations.

## Stage I — direct transfer

Run:

- CARE -> WSM;
- WSM -> CARE.

Rules:

- train/select entirely on source TRAIN/CV/DEV;
- freeze model and preprocessing;
- do not refit scaler/PCA on target;
- do not tune thresholds on target;
- target metrics are report-only evidence.

## Stage J — joint training

Train CARE + WSM jointly.

Evaluate separately on:

- CARE;
- WSM;
- D-Vlog when available;
- YouTubePD when available.

The split/participant integrity of each source corpus must remain intact.

## Stage K — small-target adaptation

Starting from source-trained models, adapt on a pre-declared small labeled subset of target TRAIN.

Before execution freeze:

- subset fractions;
- participant-level sampling;
- seeds;
- whether preprocessing can update;
- which layers can update.

Never adapt on DEV/TEST.

## Stage L — Optuna

Only the winning model + feature combination + loss family enters the primary Optuna study.

Optuna objective uses TRAIN CV only.

Search space must be narrow and architecture-aware.

TEST/external metrics are forbidden from trial selection.

## Stage M — final reporting

Consolidate the already logged fold-level CARE/WSM TEST results for the DEV-selected configuration and run the deferred external evaluations.

Report:

- within-corpus results;
- direct transfer;
- joint training;
- adaptation;
- external tests;
- per-task/per-fold CARE metrics;
- per-task/per-fold WSM metrics;
- per-task/per-fold Average metrics;
- aggregate fold statistics;
- coverage;
- seed variability;
- model/runtime/parameter footprint where practical.

## MLflow / reproducibility

Every run must record:

- Git commit;
- resolved config;
- split/fold;
- seed;
- corpus/protocol;
- feature families;
- native/Mantis path;
- Mantis checkpoint/layer/pooling if used;
- availability policy;
- scaler/PCA artifact identity;
- model;
- loss/balancing method;
- metrics;
- checkpoint;
- MLflow run ID.
