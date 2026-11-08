# DEEP-008 Batch 02 Preparation

## Scope and hard stop

This preparation defines two opt-in hypotheses for a future 15-run, five-fold Batch 02. It is CPU-only. No CUDA training, `--all` execution, real-data inference, feature extraction, audio access or TEST inspection was performed. The historical gated-fusion model, canonical folds, Dataset, DataModule, Trainer, losses and metrics remain unchanged except for the explicitly registered opt-in components described below.

## H1 — Independent task optimization

H1 contains ten configurations. Each configuration uses the existing gated EmotionMSP + TRILLsson fusion projections and gates, but exposes only one active task output and evaluates only that task.

- Depression: `carewsm_gated_single_task` with `active_task: depression`, unweighted observed-label BCE, and `val_mean_corpora/depression/macro_recall` for checkpoint selection.
- Parkinson: `carewsm_gated_single_task` with `active_task: parkinson`, TRAIN-derived fold-specific `pos_weight`, and `val_mean_corpora/parkinson/macro_recall` for checkpoint selection.
- The single-task metric emits only the active task’s six approved metrics.
- The single-task monitor averages CARE and WSM values for the active task only; it does not require or construct the four-task monitor.

The future H1 analysis will pair the Depression and Parkinson checkpoints selected independently within the same fold, combine their active-task logits, and calculate the four corpus/task UAR components. An individual incomplete single-task run is not an H1 scientific candidate.

## H2 — Corpus-specific heads

H2 contains five configurations using `carewsm_gated_corpus_heads`. The shared gated representation is the same native turn-level fusion mechanism. The model routes each sample through separate CARE or WSM Depression/Parkinson heads using the existing `sample_meta` corpus field. H2 retains the weighted Parkinson BCE and the canonical `val_mean_corpora_tasks/macro_recall` monitor. `carewsm_gated_fusion` is not modified.

## Exact configurations

All files are under `configs/audio_representation_pilot/deep008/batch02/`:

- H1 Depression: `H1_depression_fold0.yaml` through `H1_depression_fold4.yaml`.
- H1 Parkinson: `H1_parkinson_fold0.yaml` through `H1_parkinson_fold4.yaml`.
- H2 corpus heads: `H2_corpus_heads_fold0.yaml` through `H2_corpus_heads_fold4.yaml`.

All 15 configurations preserve the existing EmotionMSP and TRILLsson representations, five fixed joint folds, participant isolation, seed 5, batch size 64, hidden dimension 64, dropout 0.1, input dimension 2048, family dimensions `[1024, 1024]`, AdamW at `0.0001` with zero weight decay, 50-epoch maximum, patience 7, mixed precision, gradient clipping, native Dataset/DataModule/Trainer interfaces and validation-only selection.

H1 Parkinson weights are unchanged and fold-specific: `2.358878504672897`, `2.375234521575985`, `2.3653308480894686`, `2.4134980988593155`, and `2.3594771241830066`. H1 Depression uses unweighted BCE. H2 uses the original weighted-F loss and those same weights.

## Isolated runtime namespace and launcher

The 15 configurations use:

- Experiment name: `audio_representation_pilot_deep008_batch02`.
- Runtime/log root: `logs/audio_representation_pilot/deep008_batch02`.
- MLflow database: `sqlite:///logs/audio_representation_pilot/deep008_batch02/mlflow.db`.
- Fold-specific run names: `H1_depression_fold{0..4}`, `H1_parkinson_fold{0..4}`, and `H2_corpus_heads_fold{0..4}`.
- Launcher: `scripts/run_audio_deep008_batch02.sh`.

The launcher lists exactly 15 configs in deterministic order, performs collision checks and config validation, and launches one process at a time only under `--all`. Existing runtime and MLflow artifacts are not overwritten. `--check` is the only launcher mode used during this preparation; `--all` was not run.

## Scientific controls and future decision rule

The primary control is historical weighted-F, with five-fold VAL UAR `0.82706 ± 0.02788`. The secondary reference is DB-MTL, `0.8317 ± 0.0371`. Both hypotheses must be assessed using the same five folds and paired fold differences. Future reporting must include overall VAL UAR, the four corpus/task Macro Recalls, Patient Recall, Healthy Recall and the five paired fold differences.

A complete hypothesis is provisionally accepted only if it has mean VAL improvement of at least `+0.010` over weighted-F, no four-component mean degradation greater than `0.020`, and at least three of five paired fold wins. H1 is judged only after combining the two separately VAL-selected active-task models. TEST remains descriptive evidence and cannot govern selection, tuning, continuation or rejection.

## CPU verification record

The following checks passed:

- All 15 Chimera configurations validated successfully.
- `bash scripts/run_audio_deep008_batch02.sh --check` passed with “Preflight passed; no training was launched.”
- Focused Batch 02 synthetic tests: 4 passed.
- Historical DEEP-008 regression set plus Batch 02 tests: 34 passed.
- Python compilation passed for changed Python modules and tests.
- `bash -n scripts/run_audio_deep008_batch02.sh` passed.
- `git diff --check` passed.

The virtual environment emitted existing dependency warnings during pytest, including Cutlass deprecations, Pydantic deprecations and an existing Chimera entry-point circular-import warning; they did not fail the checks.

## Stop condition

This document records preparation only. No 15-run CUDA batch, checkpoint generation, TEST evaluation or research conclusion is authorized by this task. GPU execution requires separate owner approval after manager review.
