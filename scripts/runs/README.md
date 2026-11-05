# CAREWSM audio experiment launcher

The only launcher is `scripts/run_audio_experiments.sh`.

```bash
scripts/run_audio_experiments.sh --list
scripts/run_audio_experiments.sh --check --feature egemaps
scripts/run_audio_experiments.sh --check --feature disvoice_prosody_static
scripts/run_audio_experiments.sh --check --feature egemaps_func --model mlp
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --feature egemaps
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --feature egemaps --model mptsnet
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --all
```

With no arguments it prints help and launches nothing. `--feature FEATURE` runs the existing YAMLs in `configs/audio_feature_selection/FEATURE/`; `--model MODEL` restricts the selection to that model's five literal fold files. `--all` discovers every prepared feature folder. The script never generates or edits YAML, sweeps seeds, or implements training. Failed Chimera commands are recorded while the launcher continues through the selected configurations, and the launcher exits nonzero if any configuration fails.

Each run uses the standard Chimera layout `logs/<experiment_name>/<run_name>/`. It contains `train.log`, `summary.txt`, the YAML snapshot, `code.zip`, and `checkpoints/`. The project checkpoint callback resolves its directory from the active console logger and selects by the validation-only `dev_mean_macro_recall` monitor. No post-hoc movement or cleanup is performed.

`--check` is read-only: it parses the selected literal YAMLs under `configs/audio_feature_selection/` and checks folds, model registry names, input dimensions, fixed data roots, loggers, callbacks, and VAL-only `dev_mean_macro_recall` monitoring. It creates no training or MLflow run. Static native families are printed as `STATIC_INPUT / TEMPORAL_INTERPRETATION_LIMITED`; they are exploratory classifier inputs, not evidence that MPTSNet or MD-Former temporal mechanisms work.

TASK-010 matrix rules: temporal families (`egemaps`, `disvoice_prosody_dynamic`, `disvoice_phonation_dynamic`) allow only MPTSNet and MD-Former. Static `T=1` families allow MLP and MD-Former; MPTSNet is rejected. `--check --all` validates the resulting 110-file matrix without training or creating MLflow runs.
