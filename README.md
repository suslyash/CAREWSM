# CAREWSM

CAREWSM is a fresh research programme for multitask, multilabel recognition of Depression and Parkinson's disease using the CARE and WSM corpora, progressing from unimodal audio to multimodal modelling.

The repository started as a one-commit copy of the completed WSM codebase. Previous WSM experiments, claims, task history, and evidence are **not CAREWSM evidence**. Copied implementation is temporarily retained only for an explicit reuse audit.

## Data roots

- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`

Owner-provided contract to verify during the first audit:

- fixed `train/dev/test` splits in both corpora;
- five CV folds inside `train`;
- participant/profile grouping prevents overlap between partitions;
- CARE is the reference for feature names, storage layout, shapes/dtypes, and loader semantics;
- WSM must be brought to compatible extracted-feature coverage before the Chimera ML modelling programme.

## Active documentation

Read in order:

1. `docs/README.md`
2. `docs/PROJECT_REQUIREMENTS.md`
3. `docs/PLAN.md`
4. `docs/PROGRESS_EN.md`
5. `docs/NEXT_TASK_EN.md`
6. `docs/MANAGER_CONTRACT.md`
7. `AGENTS.md`

## Manager / Codex workflow

The manager defines one atomic task and provides a complete Markdown prompt. Codex implements that task on `codex/<task>`, pushes normally, and stops. The manager independently reviews the branch/evidence, opens and merges the PR after PASS, or issues a corrective task after FAIL.

No copied WSM model, DataModule, config, metric, callback, or result is considered active CAREWSM methodology until it passes the new audit.
