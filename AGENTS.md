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

Experiment configs belong under `configs/audio_feature_selection/`.

Repository launch wrappers may live under `scripts/`, but dataset/model/loss/metric/training logic belongs under `src/audio/`.

Keep code small, explicit, typed where useful, and free of old WSM experiment assumptions.

## Mandatory code style and Chimera registration

Follow `suslyash/WSM` (`src/chimera_plugin.py`, model/data/loss/metric/callback modules) as the style and registration reference while keeping CAREWSM-specific logic independent.

- Every **new** Chimera-ML model, DataModule, loss, metric, callback, optimizer or other registry component **must be registered using an explicit decorator** on a named, readable factory function, e.g. `@MODELS.register("carewsm_name")`, `@DATAMODULES.register(...)`, `@LOSSES.register(...)`, `@METRICS.register(...)`, `@CALLBACKS.register(...)`.
- Factories must expose a clear signature such as `def component_name(context: Any | None = None, **params: Any) -> Component:` and return the concrete component. **Do not** register via assignment expressions, `REGISTRY.register(name)(factory)`, dynamically generated factory closures or hidden registration helpers.
- The plugin entry point (`src/chimera_plugin.py`) imports registration modules in a clear, explicit list, as in WSM; registration lives with each component, not in scattered plugin conditionals. Avoid duplicate-registration side effects.
- Every registered component that is part of an experiment must be declared under the corresponding Chimera YAML section (`data:`, `model:`, `loss:`, `metrics:`, `callbacks:`, `logging:` as applicable). Validate config-driven registry resolution, not just direct Python calls.
- **Readable, formatted Python is a merge gate**: PEP 8/Ruff-compatible formatting; conventional imports; 4-space indentation; proper blank lines between imports, classes and functions; meaningful names and type hints; expanded multi-line calculations. No semicolon-chained statements, one-line control-flow blocks, dense single-line functions or obfuscated architecture/metric implementations.
- Preserve each research architecture's genuine distinctive mechanisms; never replace them with superficially named generic layers. State deviations from the authors' implementations explicitly.
- Before handoff, format touched Python files and check registration/config instantiation plus one representative real-data batch where feasible. A formatter and compiler passing do not replace source-code readability review.
- Applies to **all active and future implementation tasks, including corrective refactoring**. Manager review may reject noncompliant code without requesting extra experiments.

## MANDATORY: Simple Research Code

**This project prioritizes simple, readable research code over defensive programming and abstraction. These rules apply to every future task.**

1. **Write the smallest working implementation.** Prefer deleting code over adding code.
2. **Use Chimera-ML directly.** If Trainer, DataModule, Dataset, metrics, or callbacks already provide the functionality, do not implement it again.
3. **No unnecessary conditions.** Do not add checks for states already guaranteed by the configuration, input format, or established pipeline.
4. **No speculative abstractions.** No wrapper classes, adapters, compatibility layers, generic frameworks, helper factories, or additional execution modes without a demonstrated need.
5. **No silent fallbacks.** Use explicit required inputs instead of supporting many alternative parameter combinations.
6. **No duplicate state or computation.** Do not cache, transform, rename, or recompute information unnecessarily.
7. **Keep functions short and direct.** Prefer straightforward Python over several interconnected helper methods.
8. **Use consistent filenames and class names.** Callback modules end in `_callback.py`; callback classes end in `Callback`. Apply corresponding conventions to metrics, losses, datasets, and datamodules.
9. **Avoid excessive tests.** Test meaningful scientific behavior and integration, not every trivial branch, helper, or implementation detail. Do not create test infrastructure more complicated than the implementation.
10. **Preserve necessary scientific correctness.** Missing-label masking, fixed folds, TRAIN/VAL/TEST separation, finite selection metrics, and validation-only checkpoint selection are requirements—not optional defensive checks.
11. **Do not preserve obsolete code.** Remove unused functions, classes, imports, parameters, scripts, and tests rather than keeping compatibility shims.
12. **Review complexity before every handoff.** Inspect every added condition, helper, class, and parameter. If it can be removed without changing required behavior, remove it.

**The manager may reject code that works correctly but is unnecessarily complicated. Passing tests does not override these rules.**

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

Do not invent a family-specific time axis. Any future Mantis temporalization must use actual audited feature schemas and timestamp evidence. Mantis temporalization remains deferred pending manager authorization; a later authorized task must freeze how static/turn-level families are mapped into windows.

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


## REFACTOR-000 simplicity and active data contract

CAREWSM is research-oriented Python, not a general-purpose enterprise framework. Prefer direct PyTorch and Chimera-ML implementations. Do not add needless dataclasses, wrappers, adapters, inheritance hierarchies, registries beyond Chimera-ML's required registrations, or generic feature-provider frameworks. Prefer readable short classes and functions over dense abstraction. Never compress control flow, function definitions, or multiple statements onto one line. Use conventional imports and formatting. Keep checks limited to data correctness, tensor shapes, label masking, split/profile isolation, and experiment reproducibility.

All new code comments, technical documentation, and Codex handoffs must be in English. Manager review may reject code that violates these rules even when tests pass.

The active training data layer consists of exactly these two files:

- `src/audio/data/carewsm_audio_dataset.py` — one straightforward Dataset that directly loads native feature artifacts and concatenates ordered static `T=1` features.
- `src/audio/data/carewsm_audio_datamodule.py` — one straightforward DataModule that constructs canonical CARE/WSM folds and separate evaluation populations.

Do not introduce `JointDataModule`, `NativeDataset`, `Sample`, `View`, `NativeFeature`, replacement layers, or generic feature-provider frameworks into the active training pipeline. Historical feature-audit tooling and deferred Mantis utilities may remain outside the active pipeline until explicitly authorized cleanup.

The availability manifest is not guaranteed to contain every feature-path field. The DataModule must join manifest rows with authoritative source split metadata to obtain fields such as `label_id`, `subgroup`, profile, and video paths. It must preserve canonical fixed splits, profile isolation, labels, and feature availability semantics.

The required loader/evaluation names are `train`, `val_care`, `val_wsm`, `test_care`, and `test_wsm`. Joint training is allowed, but validation and TEST outputs remain corpus-specific.

Per-epoch console and `train.log` INFO lines show sample counts, observed-label counts where relevant, loss, Macro Recall, Patient Recall, and Healthy Recall. Calculate all six approved metrics—Macro F1, Weighted F1, Macro Recall, Weighted Recall, Patient Recall, and Healthy Recall—but keep the additional metrics in summaries and optionally MLflow rather than cluttering epoch INFO lines. Console and file logging both use INFO.

The summary callback is named `CAREWSMSummaryCallback` and writes one wide semicolon-delimited `summary.txt` row per epoch with explicit split/task/metric names and relevant counts. Expected fields include `epoch`, `train_loss`, `val_care_depression_macro_recall`, `val_wsm_parkinson_patient_recall`, `test_care_depression_macro_f1`, `test_wsm_parkinson_healthy_recall`, and `dev_mean_macro_recall`. The selection monitor is `dev_mean_macro_recall`: the equal-weight arithmetic mean of CARE Depression Macro Recall, CARE Parkinson Macro Recall, WSM Depression Macro Recall, and WSM Parkinson Macro Recall. Missing components are reported transparently; they are never replaced with zero or silently renormalized.

Use this run layout and keep `log_path: logs` consistent:

```text
logs/
  <experiment_name>/
    <run_name>/
      train.log
      summary.txt
      checkpoints/
```

`sqlite:///logs/mlflow.db` remains permitted. Changing `experiment_name` and `run_name` is sufficient to distinguish runs.

Audio-specific utilities belong under `scripts/audio/python/` and `scripts/audio/shell/`. Do not add one-off smoke wrappers or unnecessary script abstraction layers. The later cleanup task may remove these obsolete artifacts only after updating references; this task does not delete them:

- `configs/audio_feature_selection/audio_mptsnet_native_egemaps_lld_joint_care_wsm_fold0_smoke.yaml`
- `scripts/runs/`
- obsolete `smoke_*` scripts

Their removal is REFACTOR-005 work.


The active refactor sequence is: REFACTOR-000 documentation contract; REFACTOR-001 corrective Dataset/DataModule implementation after authorization; REFACTOR-002 metrics and epoch logging; REFACTOR-003 summary and paths; REFACTOR-004 configuration migration; REFACTOR-005 repository cleanup; then final integration review. TASK-011 remains paused until independent review and merge.
