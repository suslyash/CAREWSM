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

- Active ChimeraML code lives in `src/audio/`, `src/fusion/`, `src/common/` and `src/chimera_plugin.py`. Keep one-time scripts outside `src/`, in `scripts/`; documentation stays at repository root or under `docs/`. No README outside root/`docs/`.
- `src/legacy/` is a passive source archive, **not an active plugin/module source**. The owner-approved shallow archive layout is `src/legacy/audio_models/`, `fusion_models/`, `video_models/`, `text_models/` and `callbacks/`. Do not create `src/legacy/audio/models/` etc again. Keep old implementation source verbatim; do not import/register/build legacy code.
- TASK-001C (PR #9) removed old WSM source/scripts/tests and initially archived 15 models with nested domain/`models` paths. TASK-001D **flattens** this archive, moves the two preserved historic callbacks from `src/common/callbacks/` to `src/legacy/callbacks/`, and **removes the old WSM-era `configs/`**. No new configs or tests during this task.
- Training configs, when separately authorized, must include ChimeraML callbacks `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `wsm_summary_callback` and loggers `mlflow_logger`, `console_file_logger`. `wsm_summary_callback` is preserved as archival source and will require later deliberate adaptation/active registration in `src/common/`; it must **not** be silently loaded from legacy. `wsm_segment_metrics_callback` is an optional candidate only, subject to CAREWSM label-mask/metric/TEST-policy audit.
- The owner does not require a test suite; do not add or run tests for repository-cleanup tasks. Verification uses Git scope/diff and static evidence.
- The active editable install must not require old WSM-only dependencies, especially `mamba-ssm`: its isolated source build can fail with `ModuleNotFoundError: No module named 'torch'`. TASK-001D may prune `[project].dependencies` in `pyproject.toml` to ChimeraML-only; future feature-specific dependencies require evidence and their own authorized task. Do not work around by disabling build isolation or reinstalling a legacy Torch stack.
- All archived WSM source is non-authoritative; old WSM performance results do not become CAREWSM evidence. These explicit owner rules supersede earlier conflicting layout proposals.

## 9. Chimera and extraction boundary

One-time corpus feature extraction is outside Chimera ML. **All audio preprocessing/inspection, speech-turn/diarization preparation, audio feature extraction and aggregation scripts belong under root `scripts/audio/`**. They are standalone offline tools and must not import `chimera_ml`, use its plugin/registry/DataModule/trainer, or depend on training configs; training reads produced features afterward through approved ChimeraML modules under `src/audio/`. Do not put audio preparation scripts at `scripts/` root or in `src/`.

**TASK-002B owner exception:** allow a one-time download/cache of official DiariZen inference dependencies and pretrained weights to run a maximum 3-file WSM TRAIN gate. Keep that runtime in a separate compatible environment instead of modifying the CAREWSM `.venv`/`pyproject.toml`; put caches outside Git and CARE/WSM roots. No feature extraction, TEST access, training or dataset writes are authorized by this exception. See the latest `docs/NEXT_TASK_EN.md` for exact bounds and provenance evidence.

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
