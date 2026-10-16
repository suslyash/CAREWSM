# CAREWSM Active Progress

## Project state

Project: **CAREWSM**

Repository bootstrap source: a one-commit copy of the completed WSM codebase.

The previous WSM evidence/task history is intentionally excluded from the active CAREWSM research record. No old WSM metric or claim is an accepted CAREWSM result.

## Current stage

Stage 1 — corpus, feature, and legacy-code audit.

Status: **TASK-001A ACCEPTED (PR #3)** and **TASK-001B ACCEPTED (PR #5)**. Owner refined TASK-001C: clean obsolete WSM source and top-level scripts, archive only actual old models under passive `src/legacy/`, delete obsolete tests without replacing them. No repository cleanup or extraction has run yet.

No CAREWSM scientific experiment has been run.

No CAREWSM baseline has been accepted.

Copied code/configs/tests remain intact. TASK-001A recommended KEEP / ADAPT / LEGACY / DELETE categories; no copied methodology is active by default.

## Data roots

- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`

TASK-001A observations: fixed train/dev/test and five TRAIN CV folds were consistent and sample/profile-disjoint in both corpora; CARE `videos.csv` and `care_features.py` were absent; WSM manifest has 34 columns and per-task missing labels; 0/19 CARE feature families currently appear in WSM. The CARE `linguistic_turnlevel` declared-vs-observed feature dimension discrepancy remains unresolved.

## Evaluation policy

CV folds are used for development/hyperparameter selection. DEV is used for final model/configuration selection. TEST is visible and stored as descriptive/reporting evidence for completed DEV-selected checkpoints, but it must not influence model, checkpoint, threshold, hyperparameter, architecture, sweep, or retry decisions.

## Completed task

TASK-001A — read-only corpus/feature/repository audit; manager **PASS**. Evidence: `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`. Merged PR: https://github.com/suslyash/CAREWSM/pull/3 ; merge SHA: `4097bd52adc3d46551f2439dd24c040f06a55e47`. No models or TEST metrics produced.

## Next task

TASK-001C (owner-finalized) — **clean old WSM source and scripts**, preserve old model code only under `src/legacy/`, remove incorrect `src/carewsm/` scaffold and obsolete copied tests, and do not add/run tests. See `docs/NEXT_TASK_EN.md`. No data writes, extraction or training.

## TASK-001B outcome

Created the inactive `src/carewsm/` namespace, future `configs/carewsm/` root,
legacy-isolation documentation, and bounded scaffold tests. Existing WSM-derived
code, configurations, registrations, scripts, and tests remain untouched.
TASK-001B accepted and merged in PR #5: https://github.com/suslyash/CAREWSM/pull/5 ; merge SHA `b23b686fe4a143d0cd8804e232110ccb8731924a`. Codex reported standalone import PASS and `git diff --check` PASS; `pytest` was unavailable (`No module named pytest`), so a pytest run is not claimed. **Owner corrected the source layout afterward; TASK-001C removes the unwanted scaffold.** Extraction, DiariZen, training, evaluation and CARE/WSM writes were not performed.
