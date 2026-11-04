# REFACTOR-003 — Epoch Logging, Wide Summary, and Run Paths

## Implemented scope

`CAREWSMSummaryCallback` is the active class name and the registry name remains `carewsm_summary_callback`. `CarewsmSummaryCallback` remains a temporary Python alias for legacy imports and will be removed during later cleanup. The callback detects the DataModule's `evaluation_layout`:

- `corpus`: wide `summary.txt`, native Trainer INFO lines, and `test_care`/`test_wsm` evaluation;
- `legacy`: existing CSV/JSON summary behavior and legacy metric/monitor key interpretation remain unchanged.

The 110 historical YAML files and historical `joint/average_macro_f1` selection behavior were not changed.

## Wide summary schema

Corpus mode writes exactly one semicolon-delimited data row per completed epoch to the logger's run directory. The stable header is generated deterministically in this order:

```text
epoch;
train_loss;train_num_samples;
train_depression_observed_count;train_depression_positive_support;train_depression_negative_support;train_depression_macro_f1;train_depression_weighted_f1;train_depression_macro_recall;train_depression_weighted_recall;train_depression_patient_recall;train_depression_healthy_recall;
train_parkinson_observed_count;train_parkinson_positive_support;train_parkinson_negative_support;train_parkinson_macro_f1;train_parkinson_weighted_f1;train_parkinson_macro_recall;train_parkinson_weighted_recall;train_parkinson_patient_recall;train_parkinson_healthy_recall;
...the same loss, sample-count, support, and six-metric columns for val_care, val_wsm, test_care, and test_wsm...;
dev_mean_macro_recall
```

The actual header expands the `...` into explicit columns for all five populations and both tasks. Undefined values are written as `-`, never as placeholder zeroes. Counts and supports are written as integer text. Repeated callback invocation for an existing epoch replaces that epoch's row, preventing duplicates.

Representative data row:

```text
1;0.5000;4;3;2;1;0.2500;0.2500;0.2500;0.2500;0.2500;0.2500;3;2;1;0.2500;0.2500;0.2500;0.2500;0.2500;0.2500;...;0.6250
```

The five populations are `train`, `val_care`, `val_wsm`, `test_care`, and `test_wsm`. Each has six metrics for Depression and Parkinson, observed-label counts, positive/negative supports, loss, and number of samples.

## INFO logging

In corpus mode the native Chimera Trainer produces one INFO record for each of the five populations, in this order:

```text
[train epoch 1] ...
[val_care epoch 1] ...
[val_wsm epoch 1] ...
[test_care epoch 1] ...
[test_wsm epoch 1] ...
```

The summary callback does not generate duplicate INFO records, install logging filters, or replace the Trainer logger. Macro F1, Weighted F1, and Weighted Recall remain in the wide summary and available metric results.

## TEST and selection isolation

Chimera invokes callbacks in configured list order. The future corpus-mode order is validation monitor, checkpoint, early stopping, then summary. The summary evaluates each `test_care` and `test_wsm` loader through the existing private `_run_epoch` primitive after selection callbacks have consumed validation logs. Returned metrics are mapped directly to `test_care/<task>/<metric>` and `test_wsm/<task>/<metric>` keys. Legacy `test/average/...` generation is skipped entirely in corpus mode.

Corpus epoch processing is idempotent: after a successful epoch, a small per-epoch result map and completed-epoch set prevent another TEST evaluation or another five-line INFO emission. A repeated callback invocation may refresh train/validation values in the one existing row, but always reapplies the previously computed TEST values. The epoch is marked complete only after TEST evaluation and summary writing succeed; failed/incomplete processing is not marked complete.

The callback uses the original Trainer logger directly and does not mutate it, does not call callback lifecycle methods recursively, and never writes TEST values into `dev_mean_macro_recall`.

## Run layout

The callback resolves the same run directory from Chimera's logger `log_path`:

```text
logs/<experiment_name>/<run_name>/
  train.log
  summary.txt
  checkpoints/
```

Changing `experiment_name` or `run_name` isolates runs. The implementation does not append a second `logs/` component or hard-code experiment-specific paths. New-style configurations must set both console and file logger levels to `INFO`; canonical legacy configurations remain unchanged until REFACTOR-004.

## Compatibility and limitations

Legacy `joint` validation, `CARE`/`WSM` TEST names, `carewsm_multitask_prf`, `CarewsmSummaryCallback` imports, and `joint/average_macro_f1` remain supported. YAML migration is intentionally deferred to REFACTOR-004. The summary still relies on Chimera's private `_run_epoch` because the installed Trainer has no per-epoch TEST hook. No real CARE/WSM data or training run was used.

## Verification

Actual checks completed:

- `.venv/bin/pytest -q tests/test_refactor_summary.py` — 4 passed after the REFACTOR-004 logging correction;
- `.venv/bin/pytest -q tests/test_refactor_audio_data.py tests/test_training_logging.py tests/test_refactor_metrics.py tests/test_refactor_summary.py` — 24 passed;
- `bash scripts/run_audio_experiments.sh --check --all` — passed for all canonical configurations;
- `.venv/bin/python -m py_compile` on touched Python files — passed;
- `git diff --check` — passed.

The summary-specific tests use controlled deterministic `_run_epoch` stubs rather than a full `Trainer.fit()` integration. They use a standard-library handler to verify that TEST failure leaves logging state unchanged; no filter or suppression path remains. The full repository suite was attempted and collection remains blocked by vendored DiariZen/pyannote tests requiring unavailable `intervaltree`, `pyannote.core`, `pyannote.database`, and `pyannote.audio` packages. No REFACTOR-003 test failed.
