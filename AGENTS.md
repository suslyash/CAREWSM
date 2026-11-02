# CAREWSM Codex Operating Contract

These rules apply to the whole repository.

## Required reading

Before every task read:

1. `README.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/EXPERIMENT_PROTOCOL.md`
5. `docs/MODEL_SHORTLIST.md`
6. `docs/PLAN.md`
7. `docs/PROGRESS_EN.md`
8. `docs/NEXT_TASK_EN.md`
9. the active manager-authored task prompt

Inspect `git status --short` before editing.

## Authority

Order of authority:

1. latest explicit owner/manager instruction;
2. active task prompt;
3. `docs/PROJECT_REQUIREMENTS.md`;
4. `docs/EXPERIMENT_PROTOCOL.md`;
5. `docs/PLAN.md`;
6. existing implementation conventions.

Execute exactly one atomic task. Do not promote a suggested next step into authorized work.

## Git workflow

- Never implement directly on `main`.
- Start from current `origin/main`.
- Use `codex/<task-id>`.
- Normal non-force push of the active task branch is authorized.
- Never force-push.
- Never merge to `main`.
- By default the manager owns PR creation and merge.
- Final handoff must include branch, commit SHA, push status, worktree status, and diff summary against `origin/main`.

## Project scope

The current programme is **audio-only**.

Do not add video, text, multimodal fusion, ASR, diarization research, or new modalities unless the owner explicitly changes scope.

The extracted feature families are the research inputs. Do not re-extract features unless a later task explicitly authorizes it.

## Chimera-ML layout

All active Chimera-related Python code belongs under `src/audio/`, split by responsibility:

- `src/audio/data/`
- `src/audio/features/`
- `src/audio/models/`
- `src/audio/loss/`
- `src/audio/metrics/`
- `src/audio/callbacks/`
- `src/audio/utils/`

Keep code small, explicit, typed where useful, and free of old WSM experiment assumptions.

Experiment configs belong under `configs/audio/`.

Repository-level operational wrappers may live under `scripts/`, but training/data/model logic belongs under `src/audio/`.

## Data safety

CARE: `/media/maxim/Databases/CARE/`

WSM: `/media/maxim/Databases/WSM_NEW/`

Unless an active task says otherwise:

- datasets and extracted features are read-only;
- never delete a sample because one feature failed;
- never rewrite canonical train/dev/test or fold CSVs;
- represent feature availability with deterministic metadata/masks;
- derived audit reports may be written only to task-authorized output locations.

## Feature availability

A feature file is not valid merely because it exists.

Audit must distinguish at least:

- missing;
- zero-byte/unreadable;
- schema mismatch;
- empty temporal axis;
- invalid dtype/shape;
- NaN/Inf;
- valid.

Filtering must be feature-aware and reproducible.

Do not silently map unavailable features to zero without an explicit modality/feature availability mask.

## Intended targets

The intended task is two independent binary outputs:

- Depression;
- Parkinson.

Unavailable disease labels are masked, never converted to negatives.

Actual split/label columns and feature/sample identities must be verified by Codex before the new DataModule is frozen.

## Evaluation discipline

- TRAIN folds: development and hyperparameter selection.
- DEV: final feature/model/configuration selection.
- TEST: report-only after selection.
- TEST must never drive feature selection, model selection, threshold tuning, Optuna, reruns, or ablation continuation.
- Record overall and per-task metrics; also report per-corpus metrics when both CARE and WSM contribute.

## Experiment integrity

Every run must be traceable to:

- source commit;
- resolved config;
- seed;
- feature set;
- feature-availability policy;
- split/fold fingerprint;
- model;
- checkpoint;
- metrics;
- MLflow run/artifact IDs.

Negative experiments are evidence and must be recorded.

## Owner approval gates

Explicit owner approval is required before:

- changing target semantics;
- changing split/CV/DEV/TEST policy;
- re-extracting or destructively modifying features;
- adding multimodality;
- replacing the three-model research shortlist with a new major family;
- launching a large Optuna study beyond the agreed budget.

## Required Codex handoff

Use exactly these headings:

1. Outcome
2. Changed files
3. Verification
4. Data/Test access
5. Experimental evidence
6. Git handoff
7. Blockers and risks
8. Next atomic step
