# CAREWSM audio experiment launcher

The only launcher is `scripts/run_audio_experiments.sh`.

```bash
scripts/run_audio_experiments.sh --list
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --feature egemaps
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --feature egemaps --model mptsnet
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --all
```

With no arguments it prints help and launches nothing. `--feature FEATURE` runs the existing YAMLs in `configs/audio/FEATURE/`; `--model MODEL` restricts the selection to that model's five literal fold files. `--all` discovers every prepared feature folder. The script never generates or edits YAML, sweeps seeds, or implements training. A failed Chimera command stops the script.

Each run uses the standard Chimera layout under `logs/audio/<feature>/`, with one experiment/run directory containing train log, summary, YAML snapshot, checkpoint, code snapshot and matching MLflow run. No post-hoc movement or cleanup is performed.
