# CAREWSM audio series launcher

Use the single explicit launcher:

```bash
.venv/bin/python scripts/run_audio_experiments.py --list
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_audio_experiments.py --series mptsnet_native_egemaps_lld
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_audio_experiments.py --series mdformer_native_egemaps_lld
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_audio_experiments.py --all
```

With no arguments the script prints help and launches nothing. It has one automatic loop per selected series, over folds 0–4 only. The seed remains the single fixed seed in the series YAML; there is no seed sweep. Each fold invokes the official Chimera-ML CLI and a non-zero exit stops the script.

Before launching, the script refuses to overwrite `logs/audio_<architecture>_<representation>_<feature>_fold<k>/`. The project Chimera plugin configures the standard console/file logger, checkpoint and snapshot callbacks to use that directory directly; no `find`, `mv`, rename, or post-hoc flattening is used. MLflow run identity remains the experiment/run identity for the same fold.

Each run contains `train.log`, `summary.txt`, resolved YAML, checkpoint(s), and snapshot artifacts. Collect completed summaries with `python3 scripts/collect_summary.py --root logs --output all_summary.txt`. TEST is not part of training or checkpoint selection.
