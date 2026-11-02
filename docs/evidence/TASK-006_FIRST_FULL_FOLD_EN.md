# TASK-006 experiment-system preparation

Status: PREPARED; no real training was run in this revision by explicit owner instruction.

## Series and configs

- `configs/audio/series/audio_mptsnet_native_egemaps_lld.yaml`
- `configs/audio/series/audio_mdformer_native_egemaps_lld.yaml`
- fixed seed `5`; fold is substituted only by the per-series launcher
- native `egemaps_lld`; no Mantis and no TSCMamba fallback

## Launchers

- `scripts/run_audio_experiments.py` with explicit `--list`, `--series`, and `--all` modes

Each launcher uses `set -euo pipefail`, runs folds 0–4 sequentially through `.venv/bin/chimera-ml train`, and stops on the first failed fold. It does not sweep seeds, architectures, features or hyperparameters.

## Data and sequence policy

The existing joint CARE+WSM DataModule reads canonical fold TRAIN/VAL rows for each requested fold. `max_train_samples` and `max_val_samples` are absent. Native trajectories longer than 256 frames use contiguous mean bins over the complete trajectory, preserving video-wide evidence and avoiding first-N truncation; the operation has no learned statistics.

## Selection and diagnostics

`carewsm_multitask_prf` now emits `joint/average_macro_f1`, the arithmetic mean of CARE/WSM macro F1 for Depression and Parkinson only when both diseases have valid corpus metrics. Checkpoint and early stopping monitor this VAL-only key with `mode: max`. TEST is not loaded or evaluated by the training configuration; it remains a separate fixed-checkpoint reporting task.

## Artifacts and logs

The intended final run directory is `logs/audio_<architecture>_<representation>_<feature>_fold<k>/`, created directly by the project Chimera plugin path hook, containing `train.log`, `summary.txt`, resolved YAML, checkpoint and snapshot. `scripts/collect_summary.py --root logs --output all_summary.txt` collects completed runs. `scripts/clean_audio_logs.sh` is a separate dry-run-by-default cleaner; it was not run.

## Static verification

Performed: YAML validation for both series, Python compilation of touched modules/scripts, and shell syntax checks. No Chimera training, MLflow run, log cleanup, or TEST evaluation was performed for this revision.
