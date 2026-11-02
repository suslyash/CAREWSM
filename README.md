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
4. Benchmark single feature families and structured combinations.
5. Compare three recent temporal architectures.
6. Compare multitask loss/balancing strategies and single-task baselines.
7. Run within-corpus, direct-transfer, joint-training, external-test and small-target-adaptation experiments.
8. Tune only the winning configuration with Optuna.
9. For every CV fold/configuration, evaluate TEST on both CARE and WSM and log the full required metric table; TEST numbers are evidence only and must not influence configuration selection.

## Audio methodology

The intended classifier has:

`shared/common feature representation -> task-specific feature layer -> binary head`

for Depression and Parkinson independently.

Missing disease labels are masked and are never treated as negative labels.

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
