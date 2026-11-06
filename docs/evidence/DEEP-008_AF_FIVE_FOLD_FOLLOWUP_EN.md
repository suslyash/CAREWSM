# DEEP-008 Controlled A/F Five-Fold Follow-up

## 1. Outcome

The eight authorized additional CUDA runs completed sequentially for A and F on
folds 1--4. Fold 0 was not rerun. Together with the existing fold-0 results,
the complete five-fold validation Macro Recall means are:

| Model | Fold values | Mean | Sample SD (`ddof=1`) |
|---|---|---:|---:|
| A: native EmotionMSP Pooled MLP-64 | 0.7713, 0.8225, 0.8129, 0.7697, 0.8330 | 0.8019 | 0.0295 |
| F: native EmotionMSP+TRILLsson Gated Fusion-64 | 0.8023, 0.7957, 0.7868, 0.7900, 0.8670 | 0.8084 | 0.0333 |

The fold-0 pilot winner F remains higher on the five-fold mean by 0.0065, but
the result is a small exploratory comparison and not a claim of statistical or
clinical significance.

## 2. Reproducibility

- Required base: `d2b3ed80ff578bfab6f21be5cce4f6e81a70a4cb`
- Branch: `codex/deep-008-af-folds1to4`
- Training source SHA: `82f314bd6fd9540a291a43847e08ee1e451ae697`
- Preparation commits: `09d824b`, `4bec6ab`, `82f314b`
- Seed: 5; folds: 0--4; maximum epochs: 50; patience: 7
- Device: NVIDIA GeForce RTX 4080; PyTorch `2.11.0+cu128`; CUDA 12.8
- Chimera-ML `0.3.0`; MLflow `2.22.5`
- Experiment: `audio_representation_pilot_deep008`
- MLflow database: `logs/audio_representation_pilot/mlflow.db`
- Launcher: `scripts/run_audio_deep008_af_followup.sh`
- Launcher session: `carewsm-deep008-af-followup`
- Progress ledger: `logs/audio_representation_pilot/deep008_af_followup_launcher/progress.tsv`

The fixed native artifacts, folds, labels, masks, loss, monitor, and validation
protocol were unchanged. Each selected aggregate below is the arithmetic mean
of CARE Depression, CARE Parkinson, WSM Depression, and WSM Parkinson validation
Macro Recall. Within each slash-separated field, the order is CARE Depression /
CARE Parkinson / WSM Depression / WSM Parkinson.

## 3. All run identities and selected validation results

| Arm/fold | Runtime directory | MLflow run ID | Status | Epoch/checkpoint | Components | Aggregate | Patient Recall components | Healthy Recall components |
|---|---|---|---|---|---|---:|---|---|
| A/0 | `A_native_mlp64_fold0_2026-11-06_13-11_carewsm_pooled_mlp_5ce940d8` | `08a203a1aec044779c60d8a3f06d6d7b` | FINISHED | 15 / `epoch=15_val_mean_corpora_tasks_macro_recall=0.7713.pt` | 0.8034 / 0.7070 / 0.7849 / 0.7899 | 0.7713 | 0.6744 / 0.4706 / 0.7588 / 0.6943 | 0.9323 / 0.9434 / 0.8110 / 0.8854 |
| A/1 | `A_native_mlp64_fold1_2026-11-06_13-41_carewsm_pooled_mlp_e23d2bee` | `e6ae4c3006d6463e8066710c2d0f1a79` | FINISHED | 6 / `epoch=6_val_mean_corpora_tasks_macro_recall=0.8225.pt` | 0.8900 / 0.8137 / 0.7438 / 0.8425 | 0.8225 | 0.8352 / 0.6667 / 0.6435 / 0.7675 | 0.9449 / 0.9608 / 0.8442 / 0.9174 |
| A/2 | `A_native_mlp64_fold2_2026-11-06_13-52_carewsm_pooled_mlp_ca88f9a4` | `25a16e6d94bf4046898857b38d1b6d2e` | FINISHED | 29 / `epoch=29_val_mean_corpora_tasks_macro_recall=0.8129.pt` | 0.8416 / 0.7821 / 0.7465 / 0.8813 | 0.8129 | 0.7356 / 0.5806 / 0.6977 / 0.8865 | 0.9476 / 0.9836 / 0.7952 / 0.8762 |
| A/3 | `A_native_mlp64_fold3_2026-11-06_13-58_carewsm_pooled_mlp_5f2c92fc` | `946d1558f8dc42c7bc8e3b77091da440` | FINISHED | 23 / `epoch=23_val_mean_corpora_tasks_macro_recall=0.7697.pt` | 0.8399 / 0.6462 / 0.7429 / 0.8499 | 0.7697 | 0.7561 / 0.3125 / 0.7122 / 0.8313 | 0.9237 / 0.9799 / 0.7736 / 0.8685 |
| A/4 | `A_native_mlp64_fold4_2026-11-06_14-03_carewsm_pooled_mlp_ae4b3154` | `8b7f86cf5c864b6f9730c7df47adc8e3` | FINISHED | 16 / `epoch=16_val_mean_corpora_tasks_macro_recall=0.8330.pt` | 0.8393 / 0.8417 / 0.7791 / 0.8718 | 0.8330 | 0.7556 / 0.7941 / 0.6919 / 0.8772 | 0.9231 / 0.8892 / 0.8664 / 0.8665 |
| F/0 | `F_gated_emotion_trillsson_native64_fold0_2026-11-06_13-20_carewsm_gated_fusion_9ab3fc26` | `ddf7c9e3e72641abafca3e1a426cd58f` | FINISHED | 14 / `epoch=14_val_mean_corpora_tasks_macro_recall=0.8023.pt` | 0.8225 / 0.6739 / 0.7823 / 0.9304 | 0.8023 | 0.6977 / 0.3824 / 0.7615 / 0.9258 | 0.9474 / 0.9654 / 0.8031 / 0.9350 |
| F/1 | `F_gated_emotion_trillsson_native64_fold1_2026-11-06_13-51_carewsm_gated_fusion_06be910a` | `bf9d5deb4c7b42a6a89ee978a0c88182` | FINISHED | 2 / `epoch=2_val_mean_corpora_tasks_macro_recall=0.7957.pt` | 0.8633 / 0.7275 / 0.7348 / 0.8571 | 0.7957 | 0.8132 / 0.4615 / 0.6551 / 0.7632 | 0.9134 / 0.9935 / 0.8146 / 0.9511 |
| F/2 | `F_gated_emotion_trillsson_native64_fold2_2026-11-06_13-55_carewsm_gated_fusion_e7c710bf` | `108ee477891c46749bd1a0847b980bc3` | FINISHED | 23 / `epoch=23_val_mean_corpora_tasks_macro_recall=0.7868.pt` | 0.7542 / 0.7867 / 0.7024 / 0.9038 | 0.7868 | 0.6092 / 0.6129 / 0.6076 / 0.9345 | 0.8992 / 0.9605 / 0.7972 / 0.8731 |
| F/3 | `F_gated_emotion_trillsson_native64_fold3_2026-11-06_14-01_carewsm_gated_fusion_63058ebe` | `302d74b7ab764bcc8980d9ecd4a8cacf` | FINISHED | 17 / `epoch=17_val_mean_corpora_tasks_macro_recall=0.7900.pt` | 0.8726 / 0.6552 / 0.7182 / 0.9141 | 0.7900 | 0.8537 / 0.3438 / 0.7297 / 0.9076 | 0.8916 / 0.9666 / 0.7067 / 0.9205 |
| F/4 | `F_gated_emotion_trillsson_native64_fold4_2026-11-06_14-06_carewsm_gated_fusion_32175da7` | `c15776880971492e863401560de384de` | FINISHED | 23 / `epoch=23_val_mean_corpora_tasks_macro_recall=0.8670.pt` | 0.8780 / 0.9421 / 0.7852 / 0.8627 | 0.8670 | 0.8444 / 0.9412 / 0.7355 / 0.8465 | 0.9115 / 0.9430 / 0.8350 / 0.8789 |

Every runtime had `train.log`, `summary.txt`, a selected checkpoint and
`last.pt`, and a `Training complete` record. Each had exactly one matching
MLflow run with `FINISHED` status.

## 4. Breakdown and recall balance

Five-fold component means (sample SDs in parentheses) were:

| Model | CARE Depression | CARE Parkinson | WSM Depression | WSM Parkinson |
|---|---:|---:|---:|---:|
| A | 0.8428 (0.0308) | 0.7581 (0.0803) | 0.7594 (0.0207) | 0.8471 (0.0356) |
| F | 0.8381 (0.0517) | 0.7571 (0.1154) | 0.7446 (0.0376) | 0.8936 (0.0323) |

Mean Patient / Healthy Recall was A: CARE Depression 0.7514 / 0.9343, CARE
Parkinson 0.5649 / 0.9514, WSM Depression 0.7008 / 0.8181, WSM Parkinson
0.8114 / 0.8828. For F it was CARE Depression 0.7636 / 0.9126, CARE
Parkinson 0.5484 / 0.9658, WSM Depression 0.6979 / 0.7913, and WSM Parkinson
0.8755 / 0.9117.

The worst CARE Parkinson Patient Recall was A 0.3125 on fold 3 and F 0.3438
on fold 3. The consistently higher Healthy Recall in CARE Parkinson shows a
patient/healthy imbalance that Macro Recall alone does not fully convey.

## 5. Matched-fold historical comparison

Historical matched-fold references were EmotionMSP MLP-16
`[0.7461, 0.8319, 0.7914, 0.7594, 0.8306]` and EmotionMSP+TRILLsson MLP-16
`[0.8293, 0.8046, 0.7009, 0.8050, 0.8361]`.

| Model vs reference | Paired fold differences | Mean difference | Sample SD | Folds improved |
|---|---|---:|---:|---:|
| A vs historical EmotionMSP MLP-16 | +0.0252, -0.0094, +0.0215, +0.0103, +0.0024 | +0.0100 | 0.0141 | 4/5 |
| F vs historical EmotionMSP+TRILLsson MLP-16 | -0.0270, -0.0089, +0.0859, -0.0150, +0.0309 | +0.0132 | 0.0461 | 2/5 |

These are descriptive five-fold comparisons only. F changes the representation,
capacity, and architecture relative to the historical fusion reference, so the
results do not isolate a causal effect of gating. Five folds are insufficient
for statistical significance or clinical diagnostic claims. Validation-only
selection was retained; TEST was not used for selection, checkpoint changes,
threshold tuning, or interpretation of the pilot winner.
