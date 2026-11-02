# CAREWSM audio experiment launcher

The only launcher is `scripts/run_audio_experiments.sh`; it runs existing YAML files through the official Chimera-ML CLI.

```bash
scripts/run_audio_experiments.sh --list
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --feature egemaps
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --feature egemaps --model mptsnet
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --all
```

With no arguments the script prints help and does nothing. `--feature egemaps` runs all prepared eGeMAPS models and folds; `--model mptsnet` restricts it to the five MPTSNet YAMLs. The script never edits YAML, generates configs, sweeps seeds, or implements a training loop. `set -euo pipefail` stops at the first failed fold.

The prepared layout is `configs/audio/egemaps/{mptsnet,mdformer}_fold{0..4}.yaml`. Chimera-ML uses `logs/audio/egemaps/<model>_fold<k>/<generated-run>/` with standard train log, summary, YAML snapshot, checkpoints, snapshots and MLflow identity. No log cleanup or post-hoc movement is performed.
