# TASK-006 experiment-system preparation

Status: PREPARED; no real training was run in this revision by explicit owner instruction.

## Series and configs

Only the confirmed native eGeMAPS LLD family is prepared: `configs/audio/egemaps/` (the former `configs/audio/series/` layout is removed). It contains exactly ten explicit YAML files: `mptsnet_fold0.yaml` through `mptsnet_fold4.yaml` and `mdformer_fold0.yaml` through `mdformer_fold4.yaml`. The confirmed input dimension is 25. No other feature family is configured because its native input dimensions are not frozen in the current project contract.

The sole launcher is `scripts/run_audio_experiments.sh`. It selects existing YAML files and invokes the official Chimera-ML CLI without dynamic fold substitution or generated configurations.

## Data and sequence policy

The existing joint CARE+WSM DataModule reads canonical fold TRAIN/VAL rows for each requested fold. `max_train_samples` and `max_val_samples` are absent. Native trajectories longer than 256 frames use contiguous mean bins over the complete trajectory, preserving video-wide evidence and avoiding first-N truncation; the operation has no learned statistics.

## Selection and diagnostics

`carewsm_multitask_prf` now emits `joint/average_macro_f1`, the arithmetic mean of CARE/WSM macro F1 for Depression and Parkinson only when both diseases have valid corpus metrics. Checkpoint and early stopping monitor this VAL-only key with `mode: max`. TEST is not loaded or evaluated by the training configuration; it remains a separate fixed-checkpoint reporting task.

## Artifacts and logs

The intended final run directory is `logs/audio/egemaps/<model>_fold<k>/<generated-run>/`, created by the standard Chimera-ML run layout, containing `train.log`, `summary.txt`, resolved YAML, checkpoint and snapshot. `scripts/collect_summary.py --root logs --output all_summary.txt` collects completed runs. `scripts/clean_audio_logs.sh` is a separate dry-run-by-default cleaner; it was not run.

## Static verification

Performed: YAML validation for both series, Python compilation of touched modules/scripts, and shell syntax checks. No Chimera training, MLflow run, log cleanup, or TEST evaluation was performed for this revision.
