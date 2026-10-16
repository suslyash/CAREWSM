# WSM Codex Operating Contract

These rules apply to the entire repository. They are intended for an implementing Codex controlled by a manager agent.

## 1. Required reading before every task

Read these files in order before editing:

1. docs/README.md
2. docs/PROJECT_REQUIREMENTS.md
3. docs/PROGRESS_EN.md
4. docs/NEXT_TASK_EN.md
5. the active-stage plan explicitly named by docs/NEXT_TASK_EN.md, if any
6. the source/config files explicitly named by docs/NEXT_TASK_EN.md

Do **not** read docs/archive/*, closed-stage plan files, or the full cross-stage PLAN.md by default. They are optional context and should be loaded only when the active task or manager explicitly requires them.

Also inspect git status before changes. Existing user changes are not disposable.

If docs/NEXT_TASK_EN.md is absent, contains more than one unrelated goal, or conflicts with docs/PROJECT_REQUIREMENTS.md, do not make implementation changes. Report the conflict to the manager.

## 2. Scope discipline

- Execute exactly one atomic task from docs/NEXT_TASK_EN.md.
- Modify only paths listed as allowed in that task, plus docs/PROGRESS_EN.md.
- Do not begin the next PLAN stage after completing the current task.
- Do not broaden a refactor because adjacent code could be improved.
- Do not change BASELINES, SOTA_REVIEW_EN, PROJECT_REQUIREMENTS, or PLAN unless the task explicitly requests a documentation revision.
- Do not delete data, start a large training run, or install a dependency unless the task explicitly authorizes it.
- Do not delegate to another agent unless the current user or NEXT_TASK_EN explicitly asks for delegation.

Repository authority order:

1. current explicit user/manager instruction;
2. docs/PROJECT_REQUIREMENTS.md;
3. docs/NEXT_TASK_EN.md;
4. docs/PROGRESS_EN.md for current stage/status and frozen recent decisions;
5. the active-stage detail file under docs/plan/ and docs/PLAN.md for planning detail;
6. existing implementation conventions.

NEXT_TASK_EN cannot waive a project invariant unless the manager explicitly identifies and approves the exact requirement being changed.

## 3. Git branch and handoff workflow

For every implementation task, Git handoff is mandatory unless the manager explicitly disables it for that task.

- Never implement directly on `main` or `master`.
- Before editing, run `git fetch origin` and inspect `git status --short`. Preserve all existing user changes.
- Start from the current `origin/main` after the manager task commit is visible.
- Create exactly one task branch named `codex/<task-id-lowercase>`, for example `codex/task-001b`.
- If that local or remote branch already exists, do not overwrite, reset, reuse, or force-push it unless the manager explicitly authorizes reuse. Stop and report the conflict.
- Keep the branch limited to the current atomic task. Do not mix manager-only task-file edits or unrelated cleanup into the implementation commit.
- After all required verification passes and `docs/PROGRESS_EN.md` is updated, commit the allowed tracked changes with a message beginning with the task ID, for example `TASK-001B: build canonical manifest`.
- Push the task branch to `origin` with upstream tracking.
- Never push implementation commits directly to `main` or `master`.
- Never force-push.
- Never merge, rebase, cherry-pick into, or otherwise update `main`/`master` as part of an implementation task.
- Do not open or merge a pull request unless the current task explicitly requests it.
- The manager or user reviews the pushed branch and decides whether and how it is integrated.
- If branch creation, commit, or push fails, do not fall back to another branch or to `main`; report the exact failure.

Final verification must include the task branch name, `git status --short`, the implementation commit SHA, and a diff summary against `origin/main`.

## 4. Non-negotiable project invariants

### Frozen audio

- Never edit a file under src/audio.
- Use the existing audio method as the frozen baseline.
- Before and after every implementation task, check git diff for src/audio.
- If src/audio is already dirty, stop and report exact paths unless the task explicitly defines preservation.

### Task semantics

- The final target is two independent binary labels: depression and Parkinson.
- Unknown labels are masked, never converted to negative.
- A healthy/depression/Parkinson softmax may exist only as a reproduced paper baseline, never as the final formulation.
- Observed ground truth always overrides a pseudo-label.

### Layout

- Put unimodal reusable code in src/audio, src/video, src/text, or src/description; src/audio remains frozen.
- Put multimodal model/data/loss code in src/fusion.
- Put modality-independent callbacks, losses, metrics, masks, and blocks in src/common.
- Put executable preprocessing/reporting code in scripts/<modality> or scripts/common, never in src.
- Put configs under configs/<experiment_name>/<modality-or-ablations>.
- Use experiment_name wsm_mm_pd_dep_v1 until PLAN explicitly starts another programme.

### Chimera ML

- Use Chimera ML abstractions and current src/audio style.
- Register every config-selectable model, loss, metric, callback, datamodule, collate, and inference step.
- Update src/chimera_plugin.py when a new registration module is introduced.
- Never hide a missing required project module as an optional-dependency warning.

### Required instrumentation

Every training config must include checkpoint_callback, snapshot_callback, early_stopping_callback, console_file_logger, mlflow_logger, wsm_summary_callback, and wsm_segment_metrics_callback or a documented compatible successor.

Checkpointing and early stopping use dev/mean_score in max mode.

### Evaluation firewall

- Select only by DEV/Mean_Score defined in PROJECT_REQUIREMENTS.
- Never use TEST_NONE/SOFT/HARD to choose an epoch, threshold, hyperparameter, architecture, modality, or ablation.
- New training datamodules must expose DEV, TEST_NONE, TEST_SOFT, and TEST_HARD as separate evaluation streams and report all four every epoch/validation cycle.
- DEV/Mean_Score is the only automatic selector/checkpoint/early-stopping signal; Test protocol metrics are monitoring outputs and must not drive automatic selection.

### DEPART-comparable video coverage

- A YOLO body-detection miss is not a video extraction failure for the DEPART-comparable V1 path.
- Use the detected body ROI when available; otherwise use the full RGB frame for that sampled frame.
- Do not exclude TRAIN/DEV/TEST samples solely because YOLO found no body.
- The training/evaluation cache must preserve full canonical protocol membership unless the source video is unreadable or another explicitly documented fatal extraction error occurs.

### Experiment limits

- Do not tune the audio architecture.
- Use at most the two planned video families and two planned text/description families.
- Do not create a broad grid unless the manager explicitly changes PLAN.
- Preserve failed and negative results in PROGRESS_EN.

## 5. Implementation routine

For every task:

1. Restate the boundary from NEXT_TASK_EN.
2. Inspect relevant code and dirty changes.
3. Make the smallest coherent change.
4. Register new selectable components.
5. Run required verification.
6. Add proportional checks for clear risks.
7. Check the full diff, especially src/audio and out-of-scope paths.
8. Update docs/PROGRESS_EN.md with facts, commands, results, and blockers.
9. Stop; do not execute a proposed next task.

Unit tests are optional, but registry/config validation and a forward/loss/backward smoke check are mandatory when relevant.

## 6. Research integrity

- Keep corpus/task identity and labels out of generation prompts unless an approved ablation studies a task token.
- Version manifests, prompts, model revisions, and feature caches.
- Log pseudo-label coverage, class balance, calibration, and uncertainty.
- Treat novelty claims as hypotheses until literature and ablations verify them.
- Without a dual-annotated audit subset, do not claim verified recovery of the truly missing disease label or comorbidity.
- Do not claim that dynamic RA-STCH inherits fixed-weight STCH theory without a new proof.

## 7. PROGRESS_EN update format

PROGRESS_EN.md is the compact active ledger, not an unlimited chronological dump.

For the current task, update it with only the evidence needed for manager review and future active-stage context:

- task identifier;
- complete, partial, or blocked outcome;
- changed files;
- exact commands;
- concise verification results;
- metrics/artifacts;
- deviations;
- blockers and recommended next atomic task.

Do not copy large historical tables or old-stage chronology back into PROGRESS_EN.md. Closed-stage history is preserved under docs/archive/ and is optional context.

Codex must not move, rewrite, or compact historical evidence on its own unless the current task explicitly authorizes a documentation refactor. The manager owns archive/compaction decisions.

Only the manager edits docs/NEXT_TASK_EN.md unless the current task explicitly transfers ownership.

## 8. Required manager handoff

Respond in English with exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Give exact commands/results and clickable paths where useful. State whether src/audio stayed unchanged and whether any Test metrics were inspected. Also state the task branch name, implementation commit SHA, whether the branch was pushed to origin, and whether main/master was left untouched by the implementing Codex.
