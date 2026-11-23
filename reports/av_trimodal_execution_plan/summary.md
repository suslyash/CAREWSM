# AV-004 fusion execution plan

AV-004 prepares 15 provisional exploratory AUDIO + VIDEO + TEXT runs: three
AV-003 models across folds 0–4, using `audio_id1` (64), `video_4` (128), and
`text_id9` (64) under the `shared_features` key. Every run uses seed 5, batch
size 32, eight workers, shuffled training, AdamW at `1e-4`, 50 maximum epochs,
mixed precision, gradient clipping at 1.0, and validation early stopping with
patience 7 on `val_mean_corpora_tasks/macro_recall`.

The upstream FIT/SELECTION status is `UPSTREAM_PROVENANCE=UNKNOWN —
EXPLORATORY RESULTS ONLY`. These runs make no leakage-free, independent,
clinical, or SOTA claim.

## Owner tmux workflow

```bash
cd <CAREWSM_REPOSITORY>

bash scripts/fusion/run_all.sh --check

tmux new -s carewsm_fusion

bash scripts/fusion/run_all.sh --run
```

Detach with `Ctrl+B`, then `D`. Reconnect with:

```bash
tmux attach -t carewsm_fusion
```

For interrupted runs, use:

```bash
bash scripts/fusion/run_all.sh --resume
```

`--run` is the owner-triggered execution command. The launcher ledger and
per-run launcher output are under `logs/fusion/launcher/`. Chimera writes each
run's `train.log`, `summary.txt`, `checkpoints/`, and snapshot artifacts under
`logs/fusion/fusion/<model>_fold<fold>/`; MLflow uses
`sqlite:///logs/fusion/mlflow.db`.
