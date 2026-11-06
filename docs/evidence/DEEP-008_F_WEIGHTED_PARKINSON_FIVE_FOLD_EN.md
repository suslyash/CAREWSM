# DEEP-008 TRAIN-only Parkinson-weighted F experiment

## 1. Outcome

Five new weighted-F CUDA runs completed sequentially for folds 0--4. The loss
used complete-case joint CARE+WSM TRAIN labels only to set a fold-specific
Parkinson positive weight. No existing A/F checkpoint was retrained.

The weighted model improved CARE Parkinson Patient Recall on four of five folds
and raised its five-fold mean from 0.5484 to 0.6988. Healthy Recall fell on
four folds, from a mean of 0.9658 to 0.9466. Overall validation Macro Recall
rose from 0.8084 to 0.8271. This is an exploratory five-fold result, not a
clinical or statistically established improvement.

## 2. Reproducibility and acceptance gate

- Required base: `02d3401c3e587de668e4f0ae871de1c9b929697c`
- Branch: `codex/deep-008-f-parkinson-posweight`
- Training source SHA: `ee06330629c77e1be0af7d2fd377fdba779c7e44`
- Seed: 5; maximum epochs: 50; patience: 7
- Device: NVIDIA GeForce RTX 4080; PyTorch `2.11.0+cu128`; CUDA 12.8
- Chimera-ML: `0.3.0`; MLflow: `2.22.5`
- Experiment: `audio_representation_pilot_deep008_weighted_f`
- MLflow database: `logs/audio_representation_pilot/deep008_weighted_f/mlflow.db`
- Launcher: `scripts/run_audio_deep008_weighted_f.sh`
- Session: `carewsm-deep008-weighted-f`

The mandatory acceptance gate passed before training: all five configs validated;
all weights matched TRAIN-only counts; `pos_weight=1` matched the original loss
and gradients; Parkinson positive gradients scaled as intended; Depression and
missing-label behavior were unchanged; gradients were finite on CPU and CUDA;
GPU availability, process state, and run collisions were clear.

The historical masked loss implementation was not modified. The new registered
loss keeps Depression weight 1.0, applies the configured Parkinson weight, and
preserves observed-label masking and equal task-loss averaging.

## 3. Frozen TRAIN-only weights

| Fold | Observed TRAIN Parkinson positive | Observed TRAIN Parkinson negative | `negative / positive` |
|---:|---:|---:|---:|
| 0 | 1,070 | 2,524 | 2.358878504672897 |
| 1 | 1,066 | 2,532 | 2.375234521575985 |
| 2 | 1,073 | 2,538 | 2.365330848089469 |
| 3 | 1,052 | 2,539 | 2.413498098859316 |
| 4 | 1,071 | 2,527 | 2.359477124183007 |

Counts use only observed Parkinson labels from complete-case joint CARE+WSM
TRAIN populations for EmotionMSP+TRILLsson. VAL and TEST labels were not used.

## 4. Run identities and selected checkpoints

All five MLflow runs had status `FINISHED`; each runtime had complete logs,
summary, selected checkpoint, and `last.pt`.

| Fold | Runtime directory | MLflow run ID | Selected checkpoint |
|---:|---|---|---|
| 0 | `F_weighted_parkinson_fold0_2026-11-06_14-56_carewsm_gated_fusion_18e922c2` | `542b03733f9b45f39e40ab0ec0051c0d` | `epoch=14_val_mean_corpora_tasks_macro_recall=0.8321.pt` |
| 1 | `F_weighted_parkinson_fold1_2026-11-06_14-58_carewsm_gated_fusion_309e95a2` | `bf9395ed33d640d28653102fc6620619` | `epoch=3_val_mean_corpora_tasks_macro_recall=0.8400.pt` |
| 2 | `F_weighted_parkinson_fold2_2026-11-06_14-59_carewsm_gated_fusion_151e5750` | `4afa8dfef8144666a14e4a44017c77bb` | `epoch=11_val_mean_corpora_tasks_macro_recall=0.7979.pt` |
| 3 | `F_weighted_parkinson_fold3_2026-11-06_15-01_carewsm_gated_fusion_f7e04dbe` | `93c47d48889b4445bc293cab24b40735` | `epoch=14_val_mean_corpora_tasks_macro_recall=0.8010.pt` |
| 4 | `F_weighted_parkinson_fold4_2026-11-06_15-04_carewsm_gated_fusion_410a2700` | `9e15409719554d2daa5819df7b7b8546` | `epoch=7_val_mean_corpora_tasks_macro_recall=0.8643.pt` |

## 5. Validation metrics and paired comparison

Component order is CARE Depression / CARE Parkinson / WSM Depression / WSM
Parkinson. Differences are weighted minus unweighted F, matched by fold.

| Fold | Unweighted components | Weighted components | Unweighted overall | Weighted overall | Overall difference |
|---:|---|---|---:|---:|---:|
| 0 | 0.8225 / 0.6739 / 0.7823 / 0.9304 | 0.8203 / 0.7921 / 0.7888 / 0.9271 | 0.8023 | 0.8321 | +0.0298 |
| 1 | 0.8633 / 0.7275 / 0.7348 / 0.8571 | 0.8825 / 0.8363 / 0.7521 / 0.8890 | 0.7957 | 0.8400 | +0.0443 |
| 2 | 0.7542 / 0.7867 / 0.7024 / 0.9038 | 0.7304 / 0.8558 / 0.7073 / 0.8979 | 0.7868 | 0.7979 | +0.0111 |
| 3 | 0.8726 / 0.6552 / 0.7182 / 0.9141 | 0.8506 / 0.7177 / 0.7164 / 0.9193 | 0.7900 | 0.8010 | +0.0110 |
| 4 | 0.8780 / 0.9421 / 0.7852 / 0.8627 | 0.8483 / 0.9116 / 0.8166 / 0.8805 | 0.8670 | 0.8643 | -0.0027 |

Five-fold means and sample SDs:

| Metric | Unweighted F mean ± SD | Weighted F mean ± SD | Paired mean difference ± SD |
|---|---:|---:|---:|
| CARE Depression Macro Recall | 0.8381 ± 0.0517 | 0.8264 ± 0.0580 | -0.0117 ± 0.0201 |
| CARE Parkinson Macro Recall | 0.7571 ± 0.1154 | 0.8227 ± 0.0727 | +0.0656 ± 0.0589 |
| WSM Depression Macro Recall | 0.7446 ± 0.0376 | 0.7562 ± 0.0466 | +0.0117 ± 0.0130 |
| WSM Parkinson Macro Recall | 0.8936 ± 0.0323 | 0.9028 ± 0.0198 | +0.0091 ± 0.0157 |
| Overall validation Macro Recall | 0.8084 ± 0.0333 | 0.8271 ± 0.0279 | +0.0187 ± 0.0184 |

The CARE Parkinson improvement is accompanied by a Healthy Recall reduction:

| Metric | Unweighted F mean | Weighted F mean | Paired mean difference ± SD |
|---|---:|---:|---:|
| CARE Parkinson Patient Recall | 0.5484 | 0.6988 | +0.1505 ± 0.1147 |
| CARE Parkinson Healthy Recall | 0.9658 | 0.9466 | -0.0192 ± 0.0128 |

## 6. CARE Parkinson confusion matrices

These counts were reproduced by validation-only inference with the original
`logit > 0` threshold and observed Parkinson mask.

| Fold | Model | Patients | Healthy | TP | FN | TN | FP | Patient Recall | Healthy Recall | Macro Recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Unweighted F | 34 | 318 | 13 | 21 | 307 | 11 | 0.3824 | 0.9654 | 0.6739 |
| 0 | Weighted F | 34 | 318 | 22 | 12 | 298 | 20 | 0.6471 | 0.9371 | 0.7921 |
| 1 | Unweighted F | 39 | 306 | 18 | 21 | 304 | 2 | 0.4615 | 0.9935 | 0.7275 |
| 1 | Weighted F | 39 | 306 | 27 | 12 | 300 | 6 | 0.6923 | 0.9804 | 0.8363 |
| 2 | Unweighted F | 31 | 304 | 19 | 12 | 292 | 12 | 0.6129 | 0.9605 | 0.7867 |
| 2 | Weighted F | 31 | 304 | 24 | 7 | 285 | 19 | 0.7742 | 0.9375 | 0.8558 |
| 3 | Unweighted F | 32 | 299 | 11 | 21 | 289 | 10 | 0.3438 | 0.9666 | 0.6552 |
| 3 | Weighted F | 32 | 299 | 15 | 17 | 289 | 10 | 0.4688 | 0.9666 | 0.7177 |
| 4 | Unweighted F | 34 | 316 | 32 | 2 | 298 | 18 | 0.9412 | 0.9430 | 0.9421 |
| 4 | Weighted F | 34 | 316 | 31 | 3 | 288 | 28 | 0.9118 | 0.9114 | 0.9116 |

Across all folds, weighted F changed the CARE Parkinson totals from 93 TP / 77
FN / 1,490 TN / 53 FP to 119 TP / 51 FN / 1,460 TN / 83 FP. The weighting
therefore traded 26 additional correctly identified patients for 30 additional
healthy false positives.

## 7. Interpretation and limitations

The result supports the pre-registered class-imbalance hypothesis for CARE
Parkinson: increasing the TRAIN-derived positive weight moved the fixed
zero-logit decision rule toward more patient detections. The cost was reduced
Healthy Recall on four folds and a lower fold-4 CARE Parkinson Macro Recall.

Other-task effects were mixed. CARE Depression fell by 0.0117 on average,
while WSM Depression rose by 0.0117 and WSM Parkinson rose by 0.0091. WSM
metrics therefore did not deteriorate on average, but individual folds changed
in both directions. The weighted experiment did not alter the model, threshold,
features, seed, or validation monitor, so it is a controlled loss-weighting
comparison. It remains observational evidence from five folds, not a clinical
or statistically established claim.

TEST was not used for selection, checkpoint changes, reruns, or interpretation.
