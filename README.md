# CAREWSM

CAREWSM is an audio-first research project for joint Depression and Parkinson recognition using extracted acoustic representations from CARE and WSM.

The current phase is intentionally **audio-only**. Multimodal modelling is out of scope until the audio programme is completed and reviewed.

## Immediate goals

1. Audit all extracted audio feature families and build deterministic availability/filtering metadata.
2. Build a clean Chimera-ML audio stack under `src/audio/`.
3. Benchmark every feature family and carefully selected combinations.
4. Compare three strong recent temporal classification architectures.
5. Select the best feature/model combination using TRAIN CV and DEV.
6. Tune only the winning configuration with Optuna.
7. Evaluate TEST only as report-only evidence after model/config selection.

## Data roots

- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`

Do not delete or overwrite extracted features during auditing. Filtering is represented by deterministic manifests/masks.

## Active documentation

Read in order:

1. `docs/README.md`
2. `docs/PROJECT_REQUIREMENTS.md`
3. `docs/EXPERIMENT_PROTOCOL.md`
4. `docs/MODEL_SHORTLIST.md`
5. `docs/PLAN.md`
6. `docs/PROGRESS_EN.md`
7. `docs/NEXT_TASK_EN.md`
8. `docs/MANAGER_CONTRACT.md`
9. `AGENTS.md`

Codex implements one manager-authored task at a time on a task branch. The manager independently reviews the branch/evidence and owns PR creation and merge.
