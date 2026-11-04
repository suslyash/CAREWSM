# REFACTOR-002 — Corpus-Specific Metrics and Validation Monitor

## Scope and compatibility

This task adds an opt-in `evaluation_layout: corpus` to the active CAREWSM DataModule. The default remains `legacy` so the 110 canonical YAML configurations continue to use `carewsm_joint`, validation `joint`, TEST `CARE`/`WSM`, the legacy `carewsm_multitask_prf` metric, `CarewsmSummaryCallback`, and `joint/average_macro_f1` selection. No historical YAML or result evidence was changed.

Legacy mapping:

- training: the joint CARE+WSM loader;
- validation: `joint`;
- TEST: `CARE`, `WSM`.

Corpus mapping:

- training: the same joint loader, normalized by Chimera as `train`;
- validation: `val_care`, `val_wsm`;
- TEST: `test_care`, `test_wsm`.

Corpus validation and TEST datasets reuse the already filtered and limited CARE/WSM views. TEST is not included in `Trainer.fit()` validation.

## Masked metric

`carewsm_masked_multitask_classification` is one direct stateful metric. It consumes model outputs and Chimera-collated batches, uses `observed_targets`, and computes each loader population as provided. It does not regroup mixed batches by corpus.

For each task, observed rows are retained and missing labels are excluded. With TP, TN, FP, FN and `N = TP + TN + FP + FN`:

- Patient Recall = TP / (TP + FN);
- Healthy Recall = TN / (TN + FP);
- Macro Recall = (Patient Recall + Healthy Recall) / 2;
- Patient F1 = 2TP / (2TP + FP + FN);
- Healthy F1 = 2TN / (2TN + FP + FN);
- Macro F1 = (Patient F1 + Healthy F1) / 2;
- Weighted F1 = (positive support × Patient F1 + negative support × Healthy F1) / N;
- Weighted Recall = (positive support × Patient Recall + negative support × Healthy Recall) / N.

The output keys are task-prefixed, including all six metrics, `observed_count`, `positive_support`, and `negative_support`. No observed labels returns unavailable (`NaN`) metrics and zero counts. If either observed class has zero support, all six aggregate metrics are unavailable (`NaN`) and the supports remain visible; this prevents undefined class recall from being reported as zero.

The legacy registry implementation remains unchanged for historical YAMLs.

## Validation monitor

`carewsm_validation_monitor` adds `dev_mean_macro_recall` as the arithmetic mean of exactly:

- `val_care/depression/macro_recall`;
- `val_care/parkinson/macro_recall`;
- `val_wsm/depression/macro_recall`;
- `val_wsm/parkinson/macro_recall`.

All four components have equal weight, regardless of corpus size or observed-label count. The monitor reads validation keys only and cannot use TEST values. A missing, non-finite, or unsupported component raises a visible `ValueError`; it is never replaced with zero and remaining components are never renormalized. The monitor callback must be configured before checkpoint and early-stopping callbacks because Chimera invokes callbacks in list order and passes the same mutable epoch-log dictionary. The focused integration test constructs the real `CarewsmValidationMonitor`, `CheckpointCallback`, and `EarlyStoppingCallback`, invokes them in that order with one shared epoch-log dictionary, verifies both selection callbacks consume the calculated value, and verifies missing validation input raises before checkpoint selection.

## Verification

Passed:

- `.venv/bin/pytest -q tests/test_refactor_audio_data.py tests/test_training_logging.py tests/test_refactor_metrics.py` — 19 passed;
- `python3 -m py_compile` on all changed Python modules — passed;
- `bash scripts/run_audio_experiments.sh --check --all` — passed for all canonical configurations;
- `git diff --check` — passed.

The focused tests cover legacy and corpus loader names, corpus membership and fixed TEST separation, effective counts/profile isolation, observed-label masking, all six formulas, empty and single-class support, the equal-weight validation monitor, all-four-component requirement, TEST exclusion, callback ordering, and real Chimera `Batch` collation.

The environment has no `./.venv/bin/ruff`; no formatter executable was available at that path. The full repository suite was attempted but collection is blocked by missing vendored-test dependencies (`intervaltree` and `pyannote` packages). No full training, historical folds, TASK-011, Optuna, or feature matrix was run.

## Deferred cleanup

REFACTOR-003 owns concise INFO output, TEST-per-epoch display, callback renaming, and wide summary output. Later migration tasks may replace the transitional compatibility path after independent review; until then the legacy registrations and historical interpretation are preserved.
