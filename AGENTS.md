# CAREWSM Codex Operating Contract

These rules apply to the whole repository. Codex is an implementer controlled by a manager agent.

## 1. Required reading

Before every implementation task read, in order:

1. `docs/README.md`
2. `docs/PROJECT_REQUIREMENTS.md`
3. `docs/PLAN.md`
4. `docs/PROGRESS_EN.md`
5. `docs/NEXT_TASK_EN.md`
6. only source/config/data-contract files required by the manager task

Inspect `git status --short` before editing. Preserve user changes.

## 2. Authority and atomic scope

Authority order:

1. latest explicit owner/manager instruction;
2. `docs/PROJECT_REQUIREMENTS.md`;
3. `docs/NEXT_TASK_EN.md`;
4. `docs/PLAN.md`;
5. existing implementation conventions.

Execute exactly one manager-authored atomic task. Do not start the next stage, broaden a refactor, or treat a suggested next step as authorization.

Do not modify `PROJECT_REQUIREMENTS.md`, `PLAN.md`, `NEXT_TASK_EN.md`, or `MANAGER_CONTRACT.md` unless the manager prompt explicitly authorizes that document change. Update `PROGRESS_EN.md` only when requested by the active task.

## 3. Git workflow

- Never implement directly on `main` or `master`.
- Fetch origin and start from current `origin/main`.
- Create one task branch named `codex/<task-id-lowercase>`.
- Normal non-force push of the active task branch is authorized.
- Never force-push.
- Never merge, rebase, cherry-pick into, or otherwise update `main`.
- Do not open/merge a PR unless the manager explicitly asks; by default the manager owns PR creation and merge.
- Final handoff must include branch, commit SHA, push status, worktree status, and diff summary against `origin/main`.

## 4. Manager / implementer split

The manager:

- chooses the next task;
- writes the complete Codex prompt;
- reviews branch, diff, evidence, and scientific interpretation;
- opens and merges PRs after PASS;
- issues corrective prompts after FAIL;
- maintains roadmap/contracts when owner decisions change.

Codex:

- implements only the active task;
- reports exact facts, commands, blockers, and evidence;
- does not promote its own suggested next step into work.

## 5. Dataset safety

CARE reference corpus: `/media/maxim/Databases/CARE/`.

WSM corpus: `/media/maxim/Databases/WSM_NEW/`.

By default:

- CARE files are read-only.
- WSM raw videos and `splits/` files are read-only.
- No task may regenerate split/fold assignments unless the owner explicitly authorizes it.
- Writes under WSM `extracted/` require explicit manager authorization in the active task.
- Do not overwrite/delete existing extracted features without explicit approval.
- Do not edit CARE reference features or extraction scripts in place unless the task authorizes the exact path. Prefer project-controlled adapted scripts for WSM extraction.

## 6. Intended task semantics

Owner intent is two independent binary outputs: Depression and Parkinson. Unknown/unavailable labels are intended to be masked, never silently mapped to negative.

The first corpus audit must verify actual CSV columns, label values, label availability, and grouping before implementation relies on these assumptions.

## 7. CV / DEV / TEST policy

- The five folds inside TRAIN are for development and hyperparameter selection.
- DEV is for final model/configuration selection.
- TEST is allowed and should be stored as descriptive/reporting information so it need not be recomputed later.
- TEST must not drive checkpoint selection, thresholds, architecture/hyperparameter choices, ablation continuation, retries, or promotion/demotion.
- Every model/config change must be traceable to CV/DEV evidence or an owner requirement, never to a Test score.
- Prefer evaluating TEST once for the DEV-selected checkpoint of each completed run/configuration, then persist the result and evaluator identity.

## 8. Source layout and legacy-code policy — current owner rule

- `src/` is for ChimeraML only: active code under `src/audio/`, `src/fusion/`, `src/common/` plus `src/chimera_plugin.py`.
- **Only archived model implementation code** is allowed under `src/legacy/`; it must not be auto-imported, registered, used as an active model, or installed as an active package. No other legacy implementation belongs there.
- Do not create `src/carewsm/`; do not put README/docs, extraction/preprocessing, audit or one-off research scripts in `src/`. Root `scripts/` is for current CAREWSM one-off work and must be cleaned of obsolete WSM scripts. Documentation stays in `docs/`.
- TASK-001C is an authorized repository-only cleanup: remove obsolete WSM source except archived model code, clean stale scripts and obsolete historical tests; remove TASK-001B scaffold `src/carewsm/` and `src/legacy/README.md`.
- **Owner does not require a test suite.** Do not create or run tests for TASK-001C. Use Git inventory, diff/scope review and explicit evidence instead.
- Keep README files **only in the repository root or under `docs/`**. TASK-001C removes tracked README files elsewhere (including configs/scripts/src subfolders).
- Preserve `src/common/callbacks/wsm_summary_callback.py` as a required future training callback and `src/common/callbacks/wsm_segment_callback.py` as an **optional candidate for later scientific validation**. Do not confuse these callbacks with old WSM models or delete them in source cleanup.
- Future ChimeraML training configs must include `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `wsm_summary_callback`, `mlflow_logger`, `console_file_logger`. Identify built-in vs project-defined registration before config authoring; `wsm_segment_metrics_callback` remains optional, pending masked-label/TEST-policy review. Do not activate any training during cleanup.
- Old code is recoverable from Git history; no copied WSM model, loss, callback, DataModule or optimizer may be registered or promoted without separate approval. These owner rules supersede earlier conflicting plans.

## 9. Chimera and extraction boundary

One-time corpus feature extraction is outside Chimera ML.

After feature parity/data contracts are established, model training/evaluation pipelines use Chimera ML and project registry/plugin conventions.

Do not force CARE extraction scripts into Chimera.

## 10. Research integrity

- Unknown labels are never silently negatives.
- Do not claim cross-corpus label semantics, comorbidity, participant independence, feature equivalence, or missing-label recovery before the relevant audit supports it.
- Record failed/negative experiments after CAREWSM experiments begin.
- Do not reuse old WSM performance numbers as CAREWSM evidence.

## 11. Owner approval gates

Explicit owner approval is required before:

- adding a new major method/family outside the current PLAN;
- a broad sweep;
- destructive dataset operations;
- changing split/label/Test policy;
- a large extraction or training run beyond the active task.

## 12. Required Codex handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Data/Test access
5. Git handoff
6. Blockers and risks
7. Next atomic step
