# FINAL-INTEGRATION-001 Independent Integration and Legacy Audit

## Overall result

**PASS with non-blocking CLEANUP findings.** The active CAREWSM audio pipeline is
ready for normal configured experiments. No code correction was authorized or
made during this audit.

## Subsystem results

| Subsystem | Result | Evidence |
|---|---|---|
| Active Chimera integration | PASS | 18 focused tests passed; synthetic CPU `Trainer.fit()` verified joint TRAIN, separate CARE/WSM VAL and TEST, five native population records, six metrics, combined averages, one wide summary row, and validation-only checkpoint selection. |
| Scientific data contract | PASS | Tests cover joint five-population construction, all five fold/corpus/profile isolation, observed-label masking, missing-feature filtering, and unchanged native temporal values. |
| Model/data registry integration | PASS | `scripts/smoke_models.py` instantiated registry DataModule, MPTSNet, MD-Former, masked loss, and metric on a real batch. `scripts/smoke_chimera_datamodule.py` verified native shapes and sequence masks. |
| Canonical configurations | PASS | 110 YAML files passed `scripts/run_audio_experiments.sh --check --all`; the canonical YAML passed native Chimera `validate-config`. |
| Launcher | PASS | `--list`, no-argument help, shell syntax, and read-only full preflight passed. |
| Historical preservation | PASS | Historical report, fold-results CSV, analysis script, and historical `joint/average_macro_f1` references remain preserved and explicitly historical. |
| Optional TSCMamba dependency | BLOCKED/EXPECTED | `mamba_ssm` is unavailable; the TSCMamba smoke reports the concrete dependency error and uses no Transformer fallback. TSCMamba is not part of the 110 canonical matrix. |

## Verification performed

- `.venv/bin/pytest -q tests/test_refactor_audio_data.py tests/test_training_logging.py tests/test_refactor_metrics.py tests/test_refactor_summary.py tests/test_refactor_yaml_migration.py`: **18 passed, 2 warnings**.
- `bash scripts/run_audio_experiments.sh --check --all`: passed all 110 canonical YAMLs.
- `bash scripts/run_audio_experiments.sh --list`: passed.
- `bash scripts/run_audio_experiments.sh` with no arguments: printed help and did not launch training.
- `.venv/bin/chimera-ml validate-config -c configs/audio_feature_selection/emotionmsp_embeds/mlp_fold0.yaml`: valid.
- `.venv/bin/python scripts/smoke_chimera_datamodule.py`: real batch smoke passed with joint TRAIN 6998, CARE VAL 352, WSM VAL 1429, CARE TEST 318, and WSM TEST 1364 samples.
- `.venv/bin/python scripts/smoke_models.py`: MPTSNet and MD-Former produced finite `(2, 2)` outputs; masked metric registry resolved; TSCMamba reported the expected missing dependency.
- Python compilation, `bash -n scripts/run_audio_experiments.sh`, and `git diff --check`: passed.
- No real training, 110-run matrix, historical experiment, new scientific experiment, or TASK-011 was run.

## Legacy and dead-code findings

### CLEANUP — TSCMamba constructor and unreachable fallback

`src/audio/models/carewsm_audio_models.py:85-93,109-135` accepts
`mamba_required` but never reads it. `encoder_type` is assigned the constant
`"mamba"` at line 122, so the `else` branch at lines 133-135 is unreachable.
`safe_valid_mask()` at lines 169-173 is used only by that unreachable branch.
These are unnecessary legacy-derived paths and can be removed in a separately
authorized cleanup; they did not break the active matrix.

### CLEANUP — optional input-mask fallback

`src/audio/models/carewsm_audio_models.py:23-28` accepts `native_mask` and then
an all-ones mask when `sequence_mask` is absent. The active Dataset always
provides `sequence_mask`; this fallback is not needed by the current native
pipeline and should be considered for separately authorized simplification.

### HISTORICAL — preserved analysis and extraction utilities

`scripts/analyze_audio_experiments.py` and the other non-Chimera audit/extraction
utilities are not part of the active training path. Historical reports and the
original experiment analysis remain preserved and are not classified as dead
code by this audit.

## Scientific and integration defects

No active blocker or scientific correctness defect was found. The tests and
smokes verify fixed fold/profile isolation, joint CARE+WSM TRAIN, separate
corpus evaluation, missing-label masking, unchanged native temporal values and
sequence masks, equal-weight unpooled averages, finite validation selection,
and TEST isolation from checkpoint/early-stopping decisions.

## Required corrections

No immediate correction is required for normal configured experiments. The two
model cleanup findings above require a future authorized cleanup only if the
manager elects to remove the remaining TSCMamba compatibility-derived paths.
