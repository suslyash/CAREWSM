# CAREWSM Active Progress

## Project state

Project: **CAREWSM**

Repository bootstrap source: a one-commit copy of the completed WSM codebase.

The previous WSM evidence/task history is intentionally excluded from the active CAREWSM research record. No old WSM metric or claim is an accepted CAREWSM result.

## Current stage

Stage 1 — corpus, feature, and legacy-code audit.

Status: **TASK-001A ACCEPTED (PR #3)**, **TASK-001B ACCEPTED (PR #5)**, and **TASK-001C ACCEPTED (PR #9 merged)**. Stage 1D (obsolete configs removal and flat source archive) is next; no extraction or training has run.

No CAREWSM scientific experiment has been run.

No CAREWSM baseline has been accepted.

TASK-001C removed obsolete WSM source, tracked old scripts and copied tests. The passive `src/legacy/` still has nested model categories; two callback sources remain under `src/common/callbacks/`. Old WSM-era configurations remain pending TASK-001D. No copied methodology is active by default.

## Data roots

- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`

TASK-001A observations: fixed train/dev/test and five TRAIN CV folds were consistent and sample/profile-disjoint in both corpora; CARE `videos.csv` and `care_features.py` were absent; WSM manifest has 34 columns and per-task missing labels; 0/19 CARE feature families currently appear in WSM. The CARE `linguistic_turnlevel` declared-vs-observed feature dimension discrepancy remains unresolved.

## Evaluation policy

CV folds are used for development/hyperparameter selection. DEV is used for final model/configuration selection. TEST is visible and stored as descriptive/reporting evidence for completed DEV-selected checkpoints, but it must not influence model, checkpoint, threshold, hyperparameter, architecture, sweep, or retry decisions.

## Completed task

TASK-001A — read-only corpus/feature/repository audit; manager **PASS**. Evidence: `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`. Merged PR: https://github.com/suslyash/CAREWSM/pull/3 ; merge SHA: `4097bd52adc3d46551f2439dd24c040f06a55e47`. No models or TEST metrics produced.

## Next task

TASK-001D — remove obsolete WSM-era `configs/`, flatten passive model archive into `src/legacy/{audio_models,fusion_models,video_models,text_models}/`, and move two historical callbacks into `src/legacy/callbacks/`; see `docs/NEXT_TASK_EN.md`. No new training configs, tests, model loading, extraction or training.

## TASK-001B outcome

Created the inactive `src/carewsm/` namespace, future `configs/carewsm/` root,
legacy-isolation documentation, and bounded scaffold tests. Existing WSM-derived
code, configurations, registrations, scripts, and tests remain untouched.
TASK-001B accepted and merged in PR #5: https://github.com/suslyash/CAREWSM/pull/5 ; merge SHA `b23b686fe4a143d0cd8804e232110ccb8731924a`. Codex reported standalone import PASS and `git diff --check` PASS; `pytest` was unavailable (`No module named pytest`), so a pytest run is not claimed. **Owner corrected the source layout afterward; TASK-001C removes the unwanted scaffold.** Extraction, DiariZen, training, evaluation and CARE/WSM writes were not performed.


## TASK-001C outcome

Removed obsolete WSM source, top-level scripts, copied tests, nested README files, and the TASK-001B scaffold. Archived only historical model implementations under passive `src/legacy/`; preserved `src/common/callbacks/wsm_summary_callback.py` and `src/common/callbacks/wsm_segment_callback.py` as non-activating future candidates. The ChimeraML plugin entry point is retained as a no-op, and the package identity is CAREWSM with archived legacy packages excluded from installation.

Future manager-approved training configurations must document `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `wsm_summary_callback`, `mlflow_logger`, and `console_file_logger`; `wsm_segment_metrics_callback` remains optional pending review. No tests were created or run. **Manager PASS**, merged PR: https://github.com/suslyash/CAREWSM/pull/9 ; merge SHA `d732b7a2e83f50fa1b06018e589cc36012797793`. Remaining nested legacy archive and old configs are the explicitly scoped TASK-001D follow-up.
