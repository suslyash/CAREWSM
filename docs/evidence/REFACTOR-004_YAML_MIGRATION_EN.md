# REFACTOR-004 — Complete Canonical YAML Migration and Trainer Integration

## Migration inventory

The canonical matrix contains 110 fold configurations under the 11 feature-family directories in `configs/audio/`. Three top-level YAML files are preserved outside this canonical matrix: the smoke configuration and two task-level configuration files. All 110 canonical files were migrated; no historical result files or YAML interpretations were changed.

Each canonical configuration now has:

- `data.params.evaluation_layout: corpus`;
- metric `carewsm_masked_multitask_classification`;
- callbacks in monitor, checkpoint, early-stopping, summary, snapshot order;
- checkpoint and early-stopping monitor `dev_mean_macro_recall`, mode `max`;
- checkpoint `save_top_k: 1` and summary filename `summary.txt`;
- `logs` as the logger/checkpoint/snapshot root, `train.log`, and INFO console/file levels;
- deterministic identity `audio_<feature_family>/<feature_family>_<model>_fold<fold>`, with `include_time: false`.

For example, the former `joint/average_macro_f1` checkpoint and `summary.csv` settings in `emotionmsp_embeds/mlp_fold0.yaml` are now `dev_mean_macro_recall` and `summary.txt`; the model, feature family, dimension, fold, seed, optimizer, and training settings are unchanged.

## Preflight and integration evidence

`scripts/run_audio_experiments.sh --check --all` validates exactly 110 files without launching training or requiring a GPU. It fails if any expected feature directory, allowed feature/model pair, fold, or canonical filename is missing, duplicated, or misnamed. Partial feature/model checks remain scoped to their selection. It checks native dimensions, fixed roots, corpus layout, exact metric/callback order, validation-only selection, logging paths/levels, deterministic unique identities, and absence of the historical selection key. The negative fixture test removes `emotionmsp_embeds/mlp_fold0.yaml` and confirms full-matrix preflight fails with the expected 110-file error. `--list` continues to enumerate all 11 feature families.

`tests/test_refactor_yaml_migration.py` runs the installed Chimera `Trainer.fit()` for one synthetic CPU epoch using a joint four-sample TRAIN loader (2 CARE + 2 WSM), separate CARE/WSM validation loaders, and separate CARE/WSM TEST loaders, the pooled model, masked BCE loss, masked six-metric classification metric, validation monitor, checkpoint, early stopping, summary, snapshot, and console/file logger. The test verifies:

- joint TRAIN contains both corpus identities, both contribute to the actual model batches, and `train/num_samples` is 4;
- four validation Macro Recall values and their arithmetic `dev_mean_macro_recall`;
- checkpoint and early stopping consumption of the validation monitor;
- TEST evaluation after selection, with deliberately non-identical synthetic TEST features and flipped targets; actual VAL Macro Recall was 0.5000 for both tasks in both corpora, while TEST Macro Recall was 0.5000 for Depression and 0.0000 for Parkinson in both corpora;
- exactly five native named epoch records and no `[epoch summary]` duplicates;
- one wide summary row with all six metrics for both tasks and all five populations;
- expected `train.log`, `summary.txt`, and checkpoint artifacts;
- TEST values do not appear in checkpoint selection;
- missing validation components fail before checkpoint selection.

## Verification

- `bash scripts/run_audio_experiments.sh --check --all` — exactly 110 configurations passed.
- Negative temporary-fixture preflight — removing one canonical YAML failed as required.
- `bash scripts/run_audio_experiments.sh --list` — 11 feature families listed.
- Focused regression and migration suite — 27 passed, 2 existing Transformer warnings.
- `.venv/bin/pytest -q tests/test_refactor_yaml_migration.py` — 3 passed.
- `bash -n scripts/run_audio_experiments.sh` — passed.
- Python compilation checks — passed.
- `git diff --check` — passed.

No historical CARE/WSM experiment was run and no GPU or real dataset was required.
