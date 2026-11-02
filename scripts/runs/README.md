# CAREWSM audio experiment launcher

The only launcher is `scripts/run_audio_experiments.sh`; it is Bash and invokes the official Chimera-ML CLI.

```bash
scripts/run_audio_experiments.sh --list
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --series mptsnet_native_egemaps_lld
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --series mdformer_native_egemaps_lld
CUDA_VISIBLE_DEVICES=0 scripts/run_audio_experiments.sh --all
```

With no arguments the script prints help and does nothing. The only automatic loop inside a selected series is folds 0–4. The seed is read from the selected YAML; there is no seed sweep, Optuna, or hidden model/feature loop. `set -euo pipefail` stops at the first failed fold.

Chimera-ML owns the standard run layout. Each fold gets one experiment directory plus one generated run directory containing `train.log`, `summary.txt`, resolved YAML, checkpoints, snapshots and the matching MLflow run. No `find`, `mv`, rename, or post-hoc log manipulation is used. Existing results are preserved by Chimera's unique timestamp/hash run names.

Collect summaries with `python3 scripts/collect_summary.py --root logs --output all_summary.txt`. TEST evaluation is not part of training or checkpoint selection.
