# REFACTOR-001 — Audio data-layer refactor

## Previous architectural problems

The active DataModule depended on `JointDataModule`, `NativeDataset`, `Sample`, `View`, `NativeFeature`, and representation adapters. Data construction and direct artifact loading were spread across those layers, while the Chimera adapter owned dataset behavior.

## New two-file design

`CAREWSMAudioDataset` receives selected records, registry specifications, and corpus roots. It directly loads CSV/NPZ native artifacts and returns `inputs["native"]`, ordered Depression/Parkinson targets, observed-label masks, sequence masks, and corpus/sample/profile/feature metadata. `CAREWSMAudioDataModule` reads the existing manifest and canonical fold/test CSVs, filters complete selected-feature availability, checks profile isolation, and registers the compatibility name `carewsm_joint`.

The old modules remain only because deferred smoke scripts and the Mantis representation utility still import them; they are not used by the registered active DataModule and are deferred for REFACTOR-005.

## Compatibility contract

Single-family YAMLs remain valid. Temporal single-family arrays preserve their sequence dimension and use the existing contiguous full-trajectory mean when `max_sequence_length` is configured. Loader names remain a joint validation loader and `CARE`/`WSM` test loaders. Static families may be selected with `feature_families`; they must all load as real `T=1` arrays and are concatenated in configuration order. Mixed temporal/static and unavailable families are rejected or filtered, never zero-filled.

## Verification

- `python3 -m py_compile src/audio/data/carewsm_audio_dataset.py src/audio/data/carewsm_audio_datamodule.py src/chimera_plugin.py`: passed.
- `python3 -m pytest -q`: blocked because pytest is not installed in this environment.
- `ruff`: blocked because ruff is not installed in this environment.
- `bash scripts/run_audio_experiments.sh --check --all`: pending after focused tests are added; no training runs were launched.
- `git diff --check`: passed before documentation changes.

No real-data batch was available because the live audit manifest/artifact roots are not present in this execution environment.

## Remaining technical debt

Focused synthetic tests, real-data batch evidence, and the legacy-module cleanup remain to be completed in an environment with the project test dependencies and live data roots. REFACTOR-002 metric/callback loader changes, TASK-011 coverage work, Mantis, and experiment runs were intentionally deferred.
