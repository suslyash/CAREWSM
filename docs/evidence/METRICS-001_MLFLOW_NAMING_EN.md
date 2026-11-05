# METRICS-001 — Aggregate Metric Naming and MLflow Logging

## Outcome

METRICS-001 updates CAREWSM aggregate metric naming and MLflow integration without changing model behavior, data, loss functions, optimizer settings, metric formulas, or historical experiment artifacts. No CUDA training or experiment rerun was performed.

## Root cause

The installed Chimera-ML `Trainer.fit()` logs train and corpus-specific validation metrics to MLflow before invoking `on_epoch_end` callbacks. `CAREWSMValidationMonitorCallback` derives the aggregate metrics inside `on_epoch_end`, so those values were available to checkpointing, early stopping, and `CAREWSMSummaryCallback`, but were absent from MLflow.

The installed Trainer was not modified. The CAREWSM monitor now sends its newly derived canonical aggregate metrics directly to the existing `trainer.mlflow_logger` after adding them to the shared epoch-log dictionary.

## Canonical naming

Corpus-specific names are unchanged:

```text
val_care/<task>/<metric>
val_wsm/<task>/<metric>
test_care/<task>/<metric>
test_wsm/<task>/<metric>
```

The ambiguous names were replaced as follows:

| Previous name | Canonical name | Meaning |
|---|---|---|
| `val_average/<task>/<metric>` | `val_mean_corpora/<task>/<metric>` | Unweighted CARE/WSM mean for one task |
| `test_average/<task>/<metric>` | `test_mean_corpora/<task>/<metric>` | Unweighted CARE/WSM mean for one task |
| `dev_mean_<metric>` | `val_mean_corpora_tasks/<metric>` | Unweighted mean across four validation corpus/task combinations |
| `test_mean_<metric>` | `test_mean_corpora_tasks/<metric>` | Unweighted mean across four TEST corpus/task combinations |

The canonical checkpoint and early-stopping monitor is:

```text
val_mean_corpora_tasks/macro_recall
```

`dev_mean_macro_recall` remains in the shared epoch logs and summaries as a deprecated numerical alias. It is assigned exactly the value of `val_mean_corpora_tasks/macro_recall`; it is not sent as a duplicate MLflow time series. TEST aggregates remain descriptive and cannot affect selection.

No ordinary classification accuracy was introduced.

## Formula preservation

The six approved metric formulas in `CAREWSMAudioClassificationMetric` were not changed. Aggregate values remain arithmetic means of already computed corpus/task metrics; predictions, labels, masks, and sample counts are never pooled across corpora. The required four validation Macro Recall components must be present and finite. Missing or nonfinite required components raise an explicit `ValueError`. Optional aggregate values that are nonfinite are retained as unavailable in the epoch logs and excluded from MLflow.

## MLflow verification

A recording MLflow stub test verifies that the monitor:

- sends all finite canonical aggregate metrics to the existing logger;
- uses the callback epoch as the MLflow metric step;
- does not send original CARE/WSM metrics a second time;
- does not send the deprecated alias;
- excludes nonfinite optional aggregate values;
- remains valid when no MLflow logger is configured.

The callback order remains monitor, checkpoint, early stopping, summary, and snapshot. Therefore checkpointing and early stopping consume `val_mean_corpora_tasks/macro_recall` after the monitor has added it to the shared logs.

## Configuration migration

The new monitor was migrated in:

- 110 canonical YAMLs under `configs/audio_feature_selection/`;
- 40 DEEP native-turn YAMLs under `configs/audio_deep_features/native_turns/`;
- 300 monitor references total, covering checkpoint and early-stopping entries;
- the existing full-matrix validation script.

Training budgets, seeds, model parameters, folds, input representations, data roots, and logging paths were not changed.

## Verification

- Focused regression suite: **15 passed**, 3 pre-existing dependency deprecation warnings, 0 failures.
- Existing canonical YAML checker: **110/110 passed**.
- Chimera-ML validation: **40/40 native-turn YAMLs passed**.
- Python compilation of touched callback/metric modules: passed.
- `git diff --check`: passed.
- No live MLflow server was required.
- No training, CUDA benchmark, feature extraction, or experiment rerun was performed.

## Compatibility and risks

Historical reports, checkpoints, and runtime logs were not edited. Older external references to `dev_mean_macro_recall` remain compatible through the deprecated alias, but new configurations and reporting should use `val_mean_corpora_tasks/macro_recall`. Historical text in prior evidence documents intentionally retains the names used by those completed tasks.

The unrelated pre-existing local modification to `run-mlflow.sh` was preserved and was not included in the METRICS-001 changes.
