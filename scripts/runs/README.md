# CAREWSM audio series launchers

These launchers are prepared for owner-run experiments. They do not run automatically during repository checks.

## Commands

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/runs/run_audio_mptsnet_native_egemaps_lld.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/runs/run_audio_mdformer_native_egemaps_lld.sh
```

Each launcher runs folds 0–4 sequentially from one fixed-seed YAML series. A non-zero Chimera exit stops the launcher because `set -euo pipefail` is enabled. Stop a running fold with `Ctrl-C`; rerun the launcher only after deciding whether to preserve or manually archive its existing run directory. No automatic cleanup is performed.

Each completed fold is flattened to:

`logs/audio_<architecture>_<representation>_<feature>_fold<k>/`

The directory contains the console/file `train.log`, `summary.txt`, resolved YAML snapshot, VAL-selected checkpoint, and code snapshot. MLflow uses the configured local tracking database and receives a separate run for each fold.

`python3 scripts/collect_summary.py --root logs --output all_summary.txt` collects completed summaries without modifying source logs.

`bash scripts/clean_audio_logs.sh` is a dry-run by default; pass `--apply` only after reviewing its printed directories. It was not run as part of static preparation.

TEST evaluation is intentionally not part of epoch training or checkpoint selection. It must be implemented/run separately against a selected fixed checkpoint.
