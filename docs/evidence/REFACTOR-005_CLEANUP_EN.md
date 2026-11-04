# REFACTOR-005 Cleanup Evidence

## Naming

| Before | After |
|---|---|
| `CarewsmValidationMonitor` | `CAREWSMValidationMonitorCallback` |
| `CAREWSMSummaryCallback` | `CAREWSMSummaryCallback` |
| `MaskedMultitaskClassificationMetric` | `CAREWSMClassificationMetric` |
| `MaskedMultitaskBCE` | `CAREWSMMaskedBCELoss` |
| `CAREWSMAudioDataset` | `CAREWSMAudioDataset` |
| `CAREWSMAudioDataModule` | `CAREWSMAudioDataModule` |

The class renames do not change the registry identifiers. The obsolete `CarewsmSummaryCallback` alias was removed.

## Deleted legacy runtime code

`src/audio/metrics/classification.py` now contains only the active masked six-metric implementation and the `carewsm_masked_multitask_classification` registration. The old `BinaryMetric`, individual metric factories, `MultitaskPRFMetric`, corpus-average helper, and `carewsm_multitask_prf` registration were removed.

`CAREWSMSummaryCallback` now has only the corpus-specific `train`, `val_care`, `val_wsm`, `test_care`, and `test_wsm` summary contract. Legacy CSV/JSON output, legacy TEST remapping/averaging, old monitor selection, and compact duplicate logging were removed. Native Trainer logging remains authoritative.

The DataModule accepts only `evaluation_layout: corpus` and provides canonical loader names directly. The three noncanonical audio YAMLs were individually updated to the current metric/layout/monitor contract where they are executable smoke or validation configurations; they remain outside the 110-file matrix.

## Data-layer decision

The active training data layer remains exactly `carewsm_audio_dataset.py` and `carewsm_audio_datamodule.py`. `joint_datamodule.py` and `features/representations.py` remain outside the active pipeline as identifiable deferred Mantis research utilities; smoke scripts that exercise them remain explicit deferred utilities and are not imported by the Chimera training entry point. No replacement Sample/View/NativeFeature hierarchy was introduced.

## Registry and import audit

Retained registry identifiers: `carewsm_joint`, `carewsm_masked_bce`, `carewsm_masked_multitask_classification`, `carewsm_validation_monitor`, and `carewsm_summary_callback`, plus all active model identifiers. A repository search found no active imports of deleted metric, loss, monitor, or summary class names.

## Verification

- Focused CAREWSM suite: 23 passed, 2 existing Transformer warnings.
- `python3 -m py_compile` passed for all touched Python modules and `scripts/smoke_models.py`.
- Full 110-YAML preflight and `--list` were run after the cleanup.
- Historical experiment reports and the 110 canonical YAML matrix were not rewritten.

Remaining items for the separate final integration review are broader project-suite dependency availability and a manager review of the retained deferred Mantis utilities.
