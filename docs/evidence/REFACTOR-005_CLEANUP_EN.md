# REFACTOR-005 Cleanup Evidence

## Active tree before/after

The active training path is now the direct Chimera flow:

`CAREWSMAudioDataModule -> Trainer.fit(train, val_care, val_wsm, test_care, test_wsm) -> CAREWSMSummaryCallback`

The callback receives completed epoch logs only. It does not access the DataModule or execute the model.

Renamed modules:

- `callbacks/summary.py` -> `callbacks/carewsm_summary_callback.py`
- `callbacks/validation_monitor.py` -> `callbacks/carewsm_validation_monitor_callback.py`
- `metrics/classification.py` -> `metrics/carewsm_classification_metric.py`
- `loss/masked_bce.py` -> `loss/carewsm_masked_bce_loss.py`

Deleted obsolete files:

- `src/audio/data/joint_datamodule.py`
- `src/audio/features/representations.py`
- `src/audio/data/feature_audit.py`
- obsolete real-data smoke scripts and YAML
- `tests/test_feature_audit.py`

## Simplification

The production `src/` and repository `scripts/` diff adds 128 lines and deletes 867 lines (excluding YAML, tests, and documentation).

`CAREWSMSummaryCallback` now has only path setup, log receipt, deterministic wide-column persistence, and epoch idempotency. Manual `_run_epoch()` TEST evaluation, loader/cache state, legacy schema enumeration, and custom TEST logging were removed.

The DataModule now requires one explicit `feature_family`, has no sample limits or evaluation-layout modes, and exposes:

```python
val_dataset = {
    "val_care": datasets["val_care"],
    "val_wsm": datasets["val_wsm"],
    "test_care": datasets["test_care"],
    "test_wsm": datasets["test_wsm"],
}
```

The Dataset loads exactly one native family and preserves native temporal shape and values.

The remaining validation monitor is necessary because Chimera has no arithmetic four-component monitor expression. It writes only `dev_mean_macro_recall` before checkpoint and early stopping.

The old parallel data architecture and feature-audit runtime module were deleted rather than moved elsewhere. Generated audit reports and historical experiment reports remain preserved.

## Registry and configuration

Registry identifiers remain unchanged: `carewsm_joint`, `carewsm_masked_bce`, `carewsm_masked_multitask_classification`, `carewsm_validation_monitor`, and `carewsm_summary_callback`, plus active model identifiers. All 110 canonical YAMLs retain their dimensions, seeds, folds, feature families, optimizer settings, and validation-only selection policy; obsolete `evaluation_layout` and sequence/limit parameters were removed.

## Verification

- Synthetic CPU `Trainer.fit()`: 5 native passes (`train`, `val_care`, `val_wsm`, `test_care`, `test_wsm`).
- Focused CAREWSM suite: 18 passed, 2 existing Transformer warnings.
- Strict 110-YAML preflight: passed.
- `--list`: 11 canonical feature directories.
- Registry factory instantiation: passed.
- Shell syntax, Python compilation, and `git diff --check`: passed.
- No real CARE/WSM training or historical experiment was run.

Remaining blocker: the broader repository suite still requires unavailable vendored `intervaltree` and `pyannote.*` dependencies.

## Final correction

`AGENTS.md` now contains the permanent mandatory simplicity rules. `scripts/smoke_chimera_datamodule.py` now uses `train_dataloader()` plus the canonical `val_dataloader()` mapping (`val_care`, `val_wsm`, `test_care`, `test_wsm`) under a normal `main()` guard.
