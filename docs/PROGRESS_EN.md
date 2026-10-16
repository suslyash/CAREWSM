# CAREWSM Active Progress

## Project state

Project: **CAREWSM**

Repository bootstrap source: a one-commit copy of the completed WSM codebase.

The previous WSM evidence/task history is intentionally excluded from the active CAREWSM research record. No old WSM metric or claim is an accepted CAREWSM result.

## Current stage

Stage 0 — repository bootstrap and legacy reset.

Status: **COMPLETE pending manager review**.

No CAREWSM scientific experiment has been run.

No CAREWSM baseline has been accepted.

Copied code/configs/tests are temporarily preserved as legacy candidates pending the Stage-1 audit.

## Data roots

- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`

Owner expectations pending audit: fixed train/dev/test, five CV folds inside TRAIN, participant/profile grouping without partition overlap, CARE as the feature-format reference.

## Evaluation policy

CV folds are used for development/hyperparameter selection. DEV is used for final model/configuration selection. TEST is visible and stored as descriptive/reporting evidence for completed DEV-selected checkpoints, but it must not influence model, checkpoint, threshold, hyperparameter, architecture, sweep, or retry decisions.

## Next task

TASK-001A — repository, corpus, and CARE feature-contract audit.
