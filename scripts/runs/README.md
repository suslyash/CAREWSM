# CAREWSM audio experiment launcher

The only launcher is `scripts/run_audio_experiments.sh`.

```bash
scripts/run_audio_experiments.sh --list
scripts/run_audio_experiments.sh --check --feature egemaps
scripts/run_audio_experiments.sh --check --feature disvoice_prosody_static
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --feature egemaps
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --feature egemaps --model mptsnet
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --all
```

With no arguments it prints help and launches nothing. `--feature FEATURE` runs the existing YAMLs in `configs/audio/FEATURE/`; `--model MODEL` restricts the selection to that model's five literal fold files. `--all` discovers every prepared feature folder. The script never generates or edits YAML, sweeps seeds, or implements training. A failed Chimera command stops the script.

Each run uses the standard Chimera layout under `logs/audio/<feature>/`, with one experiment/run directory containing train log, summary, YAML snapshot, checkpoint, code snapshot and matching MLflow run. No post-hoc movement or cleanup is performed.

`--check` is read-only: it parses the selected literal YAMLs and checks folds, model registry names, input dimensions, fixed data roots, CUDA device, loggers, callbacks, and VAL-only `joint/average_macro_f1` monitoring. It creates no training or MLflow run. Static native families are printed as `STATIC_INPUT / TEMPORAL_INTERPRETATION_LIMITED`; they are exploratory classifier inputs, not evidence that MPTSNet or MD-Former temporal mechanisms work.
