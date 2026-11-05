# LAUNCH-001 Real CARE+WSM Fold Smoke

## Outcome

SUCCESS. One complete fold-0 epoch ran through the native Chimera-ML CLI on
CUDA using the real CARE and WSM data. The canonical YAML was not modified.

## Configuration and command

- Canonical configuration: `configs/audio/emotionmsp_embeds/mlp_fold0.yaml`
- Temporary configuration: `/tmp/carewsm-launch001.LoluLc/config.yaml`
- Temporary-only changes: `train.params.epochs: 1` and
  `experiment_info.params.run_name: launch001_real_fold0_20261104`
- Device: `cuda` (`NVIDIA GeForce RTX 4080`, one CUDA device)
- Command:

```bash
.venv/bin/chimera-ml train -c /tmp/carewsm-launch001.LoluLc/config.yaml
```

The resolved run name was
`launch001_real_fold0_20261104_2026-11-04_18-03_carewsm_pooled_mlp_25366b7c`.

## Native epoch results

| Population | Samples | Loss | Depression Macro Recall | Parkinson Macro Recall |
|---|---:|---:|---:|---:|
| train | 6998 | 0.6266 | 0.5821 | 0.5009 |
| val_care | 352 | 0.5718 | 0.5921 | 0.5000 |
| val_wsm | 1429 | 0.6144 | 0.7312 | 0.5087 |
| test_care | 318 | 0.5816 | 0.4600 | 0.5000 |
| test_wsm | 1364 | 0.5964 | 0.6417 | 0.5000 |

The run produced all five native population records in `train.log`. The
validation monitor reported `dev_mean_macro_recall=0.583008` (summary value
`0.5830`), which is the arithmetic mean of the four validation Macro Recall
values:

```text
(0.5921 + 0.5000 + 0.7312 + 0.5087) / 4 = 0.5830
```

TEST values were logged separately for CARE and WSM and were descriptive only;
they were not used for selection or parameter changes.

## Generated artifacts

The run directory was:

`logs/audio_emotionmsp_embeds/launch001_real_fold0_20261104_2026-11-04_18-03_carewsm_pooled_mlp_25366b7c/`

It contains `train.log`, a one-row wide `summary.txt`, and checkpoints
`checkpoints/epoch=1_dev_mean_macro_recall=0.5830.pt` and `checkpoints/last.pt`.
These generated artifacts are not committed.

## Verification

- Native TRAIN, CARE VAL, WSM VAL, CARE TEST, and WSM TEST all completed.
- `summary.txt` contains one data row.
- The selection metric is calculated from VAL only.
- The canonical YAML and source files were not changed.
