# DEEP-003 — Native-Turn Deep Audio Baseline Run Evidence

## Outcome

All 40 owner-approved native-turn baselines completed successfully on CUDA device 0. No training, model, dataset, configuration, or feature-artifact changes were made.

The experiment matrix was:

- four feature families: `emotionmsp_embeds`, `emotionmsp_trajectories`, `trillsson4_1`, and `wav2vec_xlsr300m`;
- two models: `carewsm_pooled_mlp` and `carewsm_md_former`;
- five fixed folds per family/model pair.

The project protocol does not approve ordinary classification accuracy as a selection metric. Accordingly, “accuracy” is not invented or reconstructed here. Results are ranked by the validation-only `dev_mean_macro_recall`, the arithmetic mean of CARE Depression, CARE Parkinson, WSM Depression, and WSM Parkinson validation Macro Recall. TEST values are reported as evidence only.

## Execution and integrity

Source commit at launch: `9b0397ad9af9c4a3eb8cf12cc160dcee576d3eaf`.

Each run used the native CLI form:

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/chimera-ml train -c configs/audio_deep_features/native_turns/<family>/<model>_fold<fold>.yaml
```

Runs were executed sequentially in the prescribed family/model/fold order. The 40 run directories contain `train.log`, `summary.txt`, and a `checkpoints/` directory. All 40 logs end with `Training complete`; all 40 have finite summary values and no missing required population fields.

Each run logged `val_care`, `val_wsm`, `test_care`, and `test_wsm`, with the six approved metrics for both Depression and Parkinson: Macro F1, Weighted F1, Macro Recall, Weighted Recall, Patient Recall, and Healthy Recall. The logs also contain `dev_mean_macro_recall` and validation-only early-stopping decisions.

## Aggregate results

Values are means across the five folds at each run’s best validation epoch. `dev` is `dev_mean_macro_recall`; `TEST` is `test_mean_macro_recall`. Standard deviation is population SD across folds.

| Feature family | Model | n | DEV mean | DEV SD | TEST mean |
|---|---:|---:|---:|---:|---:|
| `emotionmsp_embeds` | pooled MLP | 5 | 0.7919 | 0.0354 | 0.7510 |
| `emotionmsp_embeds` | MD-Former | 5 | **0.7944** | 0.0383 | 0.7281 |
| `emotionmsp_trajectories` | pooled MLP | 5 | 0.5020 | 0.0031 | 0.4978 |
| `emotionmsp_trajectories` | MD-Former | 5 | 0.5000 | 0.0000 | 0.5000 |
| `trillsson4_1` | pooled MLP | 5 | 0.7802 | 0.0389 | 0.7269 |
| `trillsson4_1` | MD-Former | 5 | 0.7601 | 0.0369 | 0.7017 |
| `wav2vec_xlsr300m` | pooled MLP | 5 | 0.6832 | 0.0204 | 0.6313 |
| `wav2vec_xlsr300m` | MD-Former | 5 | 0.6791 | 0.0252 | 0.6058 |

These are observed baseline results, not claims of statistical superiority. The trajectory family is at or near chance-level Macro Recall in this native-turn configuration, while the strongest validation aggregate is `emotionmsp_embeds` with MD-Former. No feature or model selection was made from TEST.

## Per-fold evidence

The table reports the best validation epoch selected by `dev_mean_macro_recall`, that selected validation value, the corresponding TEST value, and native sample counts. CARE/WSM counts vary by fold because the fixed fold validation populations differ; fixed TEST counts are 318 CARE and 1,364 WSM in every run.

| Family | Model | Fold | Best epoch | DEV Macro Recall | TEST Macro Recall | Val CARE | Val WSM | Test CARE | Test WSM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| emotionmsp_embeds | MD-Former | 0 | 29 | 0.7650 | 0.7086 | 352 | 1429 | 318 | 1364 |
| emotionmsp_embeds | MD-Former | 1 | 24 | 0.8504 | 0.7595 | 345 | 1407 | 318 | 1364 |
| emotionmsp_embeds | MD-Former | 2 | 19 | 0.8108 | 0.7451 | 335 | 1399 | 318 | 1364 |
| emotionmsp_embeds | MD-Former | 3 | 11 | 0.7403 | 0.6982 | 331 | 1428 | 318 | 1364 |
| emotionmsp_embeds | MD-Former | 4 | 11 | 0.8055 | 0.7292 | 350 | 1403 | 318 | 1364 |
| emotionmsp_embeds | pooled MLP | 0 | 23 | 0.7461 | 0.7226 | 352 | 1429 | 318 | 1364 |
| emotionmsp_embeds | pooled MLP | 1 | 24 | 0.8319 | 0.7375 | 345 | 1407 | 318 | 1364 |
| emotionmsp_embeds | pooled MLP | 2 | 27 | 0.7914 | 0.7380 | 335 | 1399 | 318 | 1364 |
| emotionmsp_embeds | pooled MLP | 3 | 29 | 0.7594 | 0.7768 | 331 | 1428 | 318 | 1364 |
| emotionmsp_embeds | pooled MLP | 4 | 22 | 0.8306 | 0.7803 | 350 | 1403 | 318 | 1364 |
| emotionmsp_trajectories | MD-Former | 0 | 1 | 0.5000 | 0.5000 | 352 | 1429 | 318 | 1364 |
| emotionmsp_trajectories | MD-Former | 1 | 1 | 0.5000 | 0.5000 | 345 | 1407 | 318 | 1364 |
| emotionmsp_trajectories | MD-Former | 2 | 1 | 0.5000 | 0.5000 | 335 | 1399 | 318 | 1364 |
| emotionmsp_trajectories | MD-Former | 3 | 1 | 0.5000 | 0.5000 | 331 | 1428 | 318 | 1364 |
| emotionmsp_trajectories | MD-Former | 4 | 1 | 0.5000 | 0.5000 | 350 | 1403 | 318 | 1364 |
| emotionmsp_trajectories | pooled MLP | 0 | 9 | 0.5000 | 0.5000 | 352 | 1429 | 318 | 1364 |
| emotionmsp_trajectories | pooled MLP | 1 | 7 | 0.5012 | 0.4985 | 345 | 1407 | 318 | 1364 |
| emotionmsp_trajectories | pooled MLP | 2 | 2 | 0.5082 | 0.4909 | 335 | 1399 | 318 | 1364 |
| emotionmsp_trajectories | pooled MLP | 3 | 7 | 0.5001 | 0.4997 | 331 | 1428 | 318 | 1364 |
| emotionmsp_trajectories | pooled MLP | 4 | 7 | 0.5005 | 0.4997 | 350 | 1403 | 318 | 1364 |
| trillsson4_1 | MD-Former | 0 | 22 | 0.7776 | 0.7246 | 352 | 1429 | 318 | 1364 |
| trillsson4_1 | MD-Former | 1 | 3 | 0.7410 | 0.6648 | 345 | 1407 | 318 | 1364 |
| trillsson4_1 | MD-Former | 2 | 11 | 0.6986 | 0.6566 | 335 | 1399 | 318 | 1364 |
| trillsson4_1 | MD-Former | 3 | 49 | 0.7782 | 0.7167 | 331 | 1428 | 318 | 1364 |
| trillsson4_1 | MD-Former | 4 | 29 | 0.8050 | 0.7459 | 350 | 1403 | 318 | 1364 |
| trillsson4_1 | pooled MLP | 0 | 29 | 0.8194 | 0.7382 | 352 | 1429 | 318 | 1364 |
| trillsson4_1 | pooled MLP | 1 | 26 | 0.7817 | 0.7311 | 345 | 1407 | 318 | 1364 |
| trillsson4_1 | pooled MLP | 2 | 7 | 0.7074 | 0.7027 | 335 | 1399 | 318 | 1364 |
| trillsson4_1 | pooled MLP | 3 | 39 | 0.7864 | 0.7336 | 331 | 1428 | 318 | 1364 |
| trillsson4_1 | pooled MLP | 4 | 26 | 0.8061 | 0.7288 | 350 | 1403 | 318 | 1364 |
| wav2vec_xlsr300m | MD-Former | 0 | 25 | 0.6585 | 0.6323 | 352 | 1429 | 318 | 1364 |
| wav2vec_xlsr300m | MD-Former | 1 | 39 | 0.7170 | 0.5943 | 345 | 1407 | 318 | 1364 |
| wav2vec_xlsr300m | MD-Former | 2 | 19 | 0.6607 | 0.6035 | 335 | 1399 | 318 | 1364 |
| wav2vec_xlsr300m | MD-Former | 3 | 17 | 0.6576 | 0.5868 | 331 | 1428 | 318 | 1364 |
| wav2vec_xlsr300m | MD-Former | 4 | 28 | 0.7018 | 0.6122 | 350 | 1403 | 318 | 1364 |
| wav2vec_xlsr300m | pooled MLP | 0 | 13 | 0.6431 | 0.6216 | 352 | 1429 | 318 | 1364 |
| wav2vec_xlsr300m | pooled MLP | 1 | 14 | 0.6882 | 0.6068 | 345 | 1407 | 318 | 1364 |
| wav2vec_xlsr300m | pooled MLP | 2 | 22 | 0.6913 | 0.6363 | 335 | 1399 | 318 | 1364 |
| wav2vec_xlsr300m | pooled MLP | 3 | 28 | 0.7005 | 0.6733 | 331 | 1428 | 318 | 1364 |
| wav2vec_xlsr300m | pooled MLP | 4 | 25 | 0.6927 | 0.6183 | 350 | 1403 | 318 | 1364 |

## Scientific interpretation and limits

- The strongest observed validation aggregate is `emotionmsp_embeds` + MD-Former (`0.7944` mean `dev_mean_macro_recall`), closely followed by `emotionmsp_embeds` + pooled MLP (`0.7919`).
- `trillsson4_1` is the second strongest family in this screening, with pooled MLP above MD-Former on the validation aggregate.
- `wav2vec_xlsr300m` is lower than the other non-trajectory families in this native-turn comparison.
- `emotionmsp_trajectories` produces near-chance Macro Recall in both model families. This is a negative baseline result, not evidence that the underlying representation is scientifically useless.
- These observations do not establish statistical significance, generalization beyond the fixed folds, or superiority of one architecture. No TEST-driven selection, threshold tuning, feature selection, or new performance claim was performed.

## Runtime artifacts and remaining risks

Runtime outputs remain under `logs/audio_deep_features/` and are not part of the Git handoff. The report records aggregate evidence from those outputs; reproducibility additionally depends on the resolved configs, source commit, fixed data splits, seed, and installed environment recorded by the run artifacts.

The worktree contained a pre-existing modification to `run-mlflow.sh`; it was not changed or included in this task’s commit. No PR or merge was created.
