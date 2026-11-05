# REFACTOR-007 Metrics and Config Directory

## Outcome

Implemented the metric-output simplification, combined arithmetic averages, and
canonical configuration-directory rename without changing model, data, loss,
optimizer, or training protocol behavior.

## Metric contract

`CAREWSMAudioClassificationMetric` remains registered as
`carewsm_masked_multitask_classification` and returns only the six approved
metrics for each task:

- `macro_f1`
- `weighted_f1`
- `macro_recall`
- `weighted_recall`
- `patient_recall`
- `healthy_recall`

`observed_count`, `positive_support`, and `negative_support` remain internal to
masking and denominator calculations but are absent from metric results, epoch
logs, summaries, and MLflow outputs.

`CAREWSMValidationMonitorCallback` now adds equal-weight, non-size-weighted
averages for both validation and TEST:

- `val_average/<task>/<metric>` and `test_average/<task>/<metric>`;
- `dev_mean_<metric>` and `test_mean_<metric>`.

For example, synthetic validation Macro Recall values
`CARE=0.2, 0.4` and `WSM=0.6, 0.8` produce
`dev_mean_macro_recall=0.5` and
`val_average/depression/macro_recall=0.4`. TEST averages are descriptive;
checkpoint and early stopping continue to monitor only
`dev_mean_macro_recall`.

The monitor writes one native-style combined record per epoch:
`[carewsm_combined epoch N]`.

## Configuration migration

The 110 canonical YAML files now live under
`configs/audio_feature_selection/`. The two obsolete files
`joint_native_fold0.yaml` and `task004_models.yaml` were removed. Active
scripts, tests, smoke configuration, plugin imports, and project instructions
use the renamed directory. Historical task and evidence records were not
rewritten.

## Verification

- `12 passed` for focused metric, logging, and YAML tests.
- `18 passed, 2 warnings` for the complete focused regression suite.
- `bash scripts/run_audio_experiments.sh --check --all`: passed all 110 files.
- `bash scripts/run_audio_experiments.sh --list`: passed.
- Python compilation checks: passed.
- `git diff --check`: passed.
- Ruff was unavailable because `./.venv/bin/ruff` is not installed.
- Synthetic CPU `Trainer.fit()` verified five native population passes, one
  combined record, validation-only selection, separate VAL/TEST averages, and
  one wide summary row.
