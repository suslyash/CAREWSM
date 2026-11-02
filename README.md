# CAREWSM

CAREWSM is an audio-first research project for joint Depression and Parkinson recognition across CARE and WSM.

The current programme is intentionally **audio-only**. Multimodal modelling is deferred.

## Corpora

CARE: `/media/maxim/Databases/CARE/`

WSM: `/media/maxim/Databases/WSM_NEW/`

Both corpora are expected to use fixed `train/dev/test` partitions and five CV folds inside TRAIN, grouped by participant/profile. Codex must verify the live metadata before the training stack is frozen.

## Current research programme

1. Audit the 11 extracted audio feature families and create deterministic validity/filtering metadata.
2. Build a clean Chimera-ML audio stack under `src/audio/`.
3. Build leak-free Mantis window embeddings for each usable feature block.
4. Train from the start on the **joint CARE + WSM training set** for every CV fold.
5. Benchmark single feature families and structured combinations under the same joint protocol.
6. Compare three recent temporal architectures.
7. Compare multitask loss/balancing strategies and single-task baselines.
8. Tune only the winning joint configuration with Optuna.
9. For every CV fold/configuration, log validation metrics for model selection and evaluate TEST on both CARE and WSM only for statistics; TEST numbers must not influence configuration selection.
10. Defer direct-transfer, external-test and target-adaptation studies until the main joint CARE+WSM audio model is established.

## Audio methodology

The intended classifier has:

`shared/common feature representation -> task-specific feature layer -> binary head`

for Depression and Parkinson independently.

Missing disease labels are masked and are never treated as negative labels.

The primary training protocol mixes CARE and WSM immediately. For fold `k`, the training set is the union of CARE fold-`k` TRAIN and WSM fold-`k` TRAIN; the validation set is the corresponding union of CARE and WSM fold-`k` VAL.

Mantis is a frozen time-series representation layer, not the final disease classifier. For every video and usable feature block, the intended derived representation is a time-ordered sequence of Mantis embeddings from 1.0-second windows with 0.5-second hop. Exact family-specific window construction is frozen only after the feature audit.

## Active documentation

Read in order:

1. `docs/README.md`
2. `docs/PROJECT_REQUIREMENTS.md`
3. `docs/METHOD.md`
4. `docs/EXPERIMENT_PROTOCOL.md`
5. `docs/MODEL_SHORTLIST.md`
6. `docs/PLAN.md`
7. `docs/PROGRESS_EN.md`
8. `docs/NEXT_TASK_EN.md`
9. `docs/MANAGER_CONTRACT.md`
10. `AGENTS.md`

Codex executes one manager-authored atomic task on `codex/<task>`, pushes normally, and stops. The manager independently reviews the branch and evidence, then owns PR creation/merge.
