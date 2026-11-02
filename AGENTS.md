# CAREWSM Codex Operating Contract

These rules apply to the whole repository.

## Required reading

Before every task read:

1. `README.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/METHOD.md`
5. `docs/EXPERIMENT_PROTOCOL.md`
6. `docs/MODEL_SHORTLIST.md`
7. `docs/PLAN.md`
8. `docs/PROGRESS_EN.md`
9. `docs/NEXT_TASK_EN.md`
10. the active manager-authored task prompt

Inspect `git status --short` before editing.

## Authority

Order of authority:

1. latest explicit owner/manager instruction;
2. active task prompt;
3. `docs/PROJECT_REQUIREMENTS.md`;
4. `docs/METHOD.md`;
5. `docs/EXPERIMENT_PROTOCOL.md`;
6. `docs/PLAN.md`;
7. existing implementation conventions.

Execute exactly one atomic task. Do not start the next task from your own recommendation.

## Git workflow

- Never implement directly on `main`.
- Start from current `origin/main`.
- Use `codex/<task-id>`.
- Normal non-force push is authorized.
- Never force-push.
- Never merge to `main`.
- By default the manager owns PR creation and merge.
- Final handoff must include branch, commit SHA, push status, worktree status, and diff summary against `origin/main`.

## Project scope

The active programme is **audio-only**.

Do not add video, text, multimodal fusion, ASR or diarization research unless the owner changes scope.

Existing audio extraction artifacts are research inputs. Do not re-extract or destructively modify them unless an active task explicitly authorizes it.

## Data roots and fixed splits

CARE:

`/media/maxim/Databases/CARE/`

WSM:

`/media/maxim/Databases/WSM_NEW/`

Expected in both corpora:

- `splits/videos.csv`;
- `splits/train.csv`, `dev.csv`, `test.csv`;
- `splits/folds/fold_{0..4}_{train,val}.csv`;
- fixed participant/profile grouping.

CARE additionally provides:

- `splits/features.json`;
- `splits/care_features.py`.

Never regenerate canonical splits unless the owner explicitly authorizes it.

## Chimera-ML layout

All active Chimera-related Python code belongs under `src/audio/`:

- `src/audio/data/`
- `src/audio/features/`
- `src/audio/models/`
- `src/audio/loss/`
- `src/audio/metrics/`
- `src/audio/callbacks/`
- `src/audio/utils/`

Experiment configs belong under `configs/audio/`.

Repository launch wrappers may live under `scripts/`, but dataset/model/loss/metric/training logic belongs under `src/audio/`.

Keep code small, explicit, typed where useful, and free of old WSM experiment assumptions.

## Feature integrity

A feature artifact is not valid merely because it exists.

Audit must distinguish at least:

- missing;
- zero-byte/unreadable;
- schema mismatch;
- empty;
- invalid shape/dtype;
- NaN/Inf;
- valid.

Never delete a sample because one feature family failed. Build deterministic per-sample/per-family availability metadata.

Experiments requiring feature set S should, by default, use samples where every feature in S is valid. Coverage must be logged.

## Mantis temporal representation

Mantis is a frozen representation layer.

Target derived representation per video and feature block:

- 1.0 s window;
- 0.5 s hop;
- time-ordered embedding sequence `[num_windows, embedding_dim]`;
- start/end time for every window;
- validity mask;
- valid-observation fraction/confidence per window;
- corpus/profile/video/block/split metadata;
- Depression/Parkinson labels and observed masks;
- Mantis checkpoint, selected layer, pooling rule;
- source-feature list and preprocessing parameters.

Do not invent a family-specific time axis. TASK-001 must first establish the actual schemas/timestamps. A later task freezes how static/turn-level families are mapped into windows.

Scaler and PCA, if used, must be fitted only on the training portion of the current experimental split/fold. Never fit them globally.

Do not store a duplicate frozen Mantis model per video.

## Targets and losses

Two independent binary tasks:

- Depression;
- Parkinson.

Unavailable disease labels are masked, never converted to negatives.

Compare:

- unweighted `BCEWithLogitsLoss`;
- weighted BCE with `pos_weight = N_negative / N_positive` computed separately per task from known TRAIN labels only.

Per-task supervised loss is averaged only over examples with that task label observed.

Base multitask combination:

`L = 0.5 * L_depression + 0.5 * L_parkinson`.

Planned task-balancing comparisons:

- Equal;
- PCGrad;
- CAGrad;
- STCH;
- PAGB;
- separate single-task models.

Do not use DEV/TEST labels to calculate `pos_weight`.

## Metrics

For every evaluated task, fold, and configuration log exactly these six classification metrics:

- Macro F1;
- Weighted F1;
- Macro Recall;
- Weighted Recall;
- Patient Recall;
- Healthy Recall.

For every fold, evaluate the same checkpoint/configuration on both CARE TEST and WSM TEST.

Also log an `Average` block computed as the simple arithmetic mean of the corresponding CARE and WSM metric values. Do not pool predictions across corpora to obtain this Average.

`Macro Recall` is the binary-class macro average of Patient Recall and Healthy Recall. It is equivalent to UAR for this setting.

Report sample counts and observed-label counts beside metrics.

## Primary joint-fold training protocol

CARE and WSM are mixed from the beginning.

For fold `k`:

- `joint_train_k = CARE fold_k_train + WSM fold_k_train`;
- `joint_val_k = CARE fold_k_val + WSM fold_k_val`;
- scaler/PCA/class weights are fitted only from `joint_train_k`;
- checkpoint selection and early stopping use validation only;
- the selected fold checkpoint is then evaluated on fixed CARE TEST and fixed WSM TEST.

Do not train separate CARE-only or WSM-only models as the main screening protocol unless a manager task explicitly requests an ablation.

## Validation and TEST discipline

For every evaluated configuration/fold, log the six required metrics on validation and on TEST.

Validation is the source of model-selection evidence.

TEST is statistics/evidence only:

- never choose features/models/losses/thresholds from TEST;
- never choose Optuna trials from TEST;
- never continue/reject an ablation because TEST improved or worsened;
- never refit preprocessing from TEST.

When useful, validation metrics should also be reported separately for CARE and WSM portions plus the same arithmetic Average used in the TEST table.

The fixed DEV split may be used only when a later manager task defines its exact role; do not substitute DEV for the fold validation set implicitly.

Direct-transfer, adaptation and external D-Vlog/YouTubePD studies are deferred until the primary joint CARE+WSM programme is established.

## Experiment integrity

Every run must be traceable to:

- source commit;
- resolved config;
- seed;
- feature set;
- raw-vs-Mantis representation;
- Mantis metadata where applicable;
- availability policy;
- split/fold fingerprint;
- preprocessing/scaler/PCA fingerprint;
- model;
- loss/balancing method;
- checkpoint;
- metrics;
- MLflow run/artifact IDs.

Negative experiments are evidence and must be recorded.

## Owner approval gates

Explicit owner approval is required before:

- changing target semantics;
- changing split/CV/DEV/TEST policy;
- re-extracting/destructively modifying audio features;
- changing the 1.0 s / 0.5 s Mantis temporalization contract;
- adding multimodality;
- replacing the three major classifier families;
- launching a large Optuna sweep beyond the agreed budget.

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
