# CAREWSM Audio Experiment Protocol

## Selection hierarchy

TRAIN CV drives development, feature/model comparison and hyperparameter search.

DEV selects the final configuration.

TEST is report-only after selection.

External D-Vlog/YouTubePD are also report-only unless the owner explicitly defines a different development protocol.

## Required metrics

Per task:

- F1;
- weighted F1;
- UAR;
- weighted average recall.

Every table also includes:

- number of evaluated samples;
- number of observed labels;
- feature coverage/retention;
- corpus/domain.

The manager may define a scalar selection score after TASK-001/Stage-2 confirms label distributions. Until then, do not silently optimize a metric not specified by the owner.

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

Finalize CARE and WSM baselines separately using full TRAIN CV and DEV.

Persist:

- best feature combination;
- native vs Mantis path;
- model;
- loss method;
- preprocessing;
- seed results.

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

Run the DEV-selected configuration on TEST and external sets.

Report:

- within-corpus results;
- direct transfer;
- joint training;
- adaptation;
- external tests;
- per-task metrics;
- per-corpus metrics;
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
