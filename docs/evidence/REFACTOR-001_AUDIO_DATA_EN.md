# REFACTOR-001 — Correct and simplify the CAREWSM audio data layer

## Outcome

The active registered data layer now uses one direct Dataset and one direct DataModule. The obsolete `chimera_datamodule.py` adapter was removed because it had no remaining consumers. `joint_datamodule.py` remains only for deferred representation/smoke utilities and is not imported by the active Chimera plugin.

## Corrections to the rejected implementation

- Manifest rows are joined by `(corpus, sample_id)` with authoritative `train.csv`, `dev.csv`, and `test.csv` source metadata before paths are built. `label_id`, `subgroup`, profile, video, and `video_dir` are taken from source metadata. The extracted layout is preferred, with the original `video_dir` fallback preserved.
- `max_train_samples` and `max_val_samples` are applied independently within CARE and WSM after deterministic ordering, before `self.views` is stored. TEST limits are applied independently to each corpus loader. Effective view counts, Dataset lengths, and `describe_context()` counts now agree.
- Production code uses readable multi-line Python with direct functions and no replacement data abstraction hierarchy.
- Focused tests exercise realistic manifest/source split separation, fallback paths, all five folds, profile isolation, masks, fusion, filtering, registration, collated batches, and model backward passes.

## Compatibility

The temporary registry name `carewsm_joint` remains. Legacy YAMLs retain a `joint` validation loader and `CARE`/`WSM` TEST loaders; metric, callback, monitor, logging-path, and YAML migration remain deferred to REFACTOR-002 through REFACTOR-004. Static feature families concatenate in configured order only when every component loads as genuine `T=1`; temporal/mixed multi-family fusion is rejected. Single temporal families preserve sequence masks and contiguous full-trajectory pooling.

## Verification

- `.venv/bin/pytest -q tests/test_refactor_audio_data.py`: **6 passed**; includes explicit effective-count and context-count assertions; two expected Transformer nested-tensor warnings.
- `.venv/bin/pytest -q`: blocked during collection by unrelated vendored DiariZen/pyannote tests missing `intervaltree`, `pyannote.core`, `pyannote.database`, and `pyannote.audio`. This is an environment dependency issue, not a failure in the focused CAREWSM tests.
- `bash scripts/run_audio_experiments.sh --check --all`: passed for all canonical configurations; no training launched.
- `python3 -m py_compile ...`: passed for touched Python files.
- `.venv/bin/black` formatting: passed for touched Dataset, DataModule, and test files.
- `git diff --check`: passed.

## Real-data evidence

Live CARE and WSM roots and the availability manifest were accessible. With fold 0 and one retained sample per corpus:

- `emotionmsp_embeds`: CARE and WSM each loaded with tensor shape `(1, 1024)`; CARE observed mask `[True, True]`, WSM `[True, False]`.
- `emotionmsp_embeds + trillsson4_1`: CARE and WSM each loaded with tensor shape `(1, 2048)` and preserved ordered family metadata.

No full experiment or TASK-011 coverage study was run.

## Remaining risks

The full repository suite remains blocked by optional vendored dependencies. The final loader-name and logging migration is intentionally deferred.
