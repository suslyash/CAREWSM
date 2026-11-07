# DEEP008 Stage G — corrected WSM-inspired CAREWSM multitask suite

## Decision status

The manager-review corrections are implemented and the valid five-fold ledger is complete: 8 methods × 5 folds = 40 scientifically valid cells. Equal, Static-STCH, PCGrad, CAGrad and DB-MTL retain their original successful runs after parity/invariant verification. Progress, RA-STCH and GradNorm use the corrected implementation and distinct corrected artifact namespaces.

All training remains through Chimera-ML using the existing CAREWSM gated-fusion model, EmotionMSP + TRILLsson 4.1, joint CARE+WSM folds, seed 5, AdamW, TRAIN-only Parkinson weighting, mixed precision, validation-only monitoring, and unchanged feature/data contracts.

## Corrected algorithms

- Progress uses explicitly configured chance-level references `(0.5, 0.5)` and applies each normalized target weight directly to the next epoch.
- RA-STCH uses EMA shared-gradient norms and cosine over the actual `projections.*` and `gates.*` parameters. The CAREWSM pseudo-reliability factor is explicitly neutral (`1.0, 1.0`), because no pseudo-label reliability signal exists.
- GradNorm uses the weights before its separate weight-optimizer update for the current model update, applied to shared and corresponding private-head gradients.
- Equal is numerically parity-checked against `carewsm_weighted_parkinson_bce` on identical model output and batch objects.

## Complete 40-cell ledger

| Method | Fold | Selected epoch | VAL CARE D/P | VAL WSM D/P | Selection VAL | Run directory | MLflow run ID | Checkpoint |
|---|---:|---:|---:|---:|---:|---|---|---|
| Equal | 0 | 14 | 0.8203 / 0.7921 | 0.7888 / 0.9271 | 0.8321 | `F_weighted_parkinson_fold0_2026-11-06_14-56_carewsm_gated_fusion_18e922c2` | `542b03733f9b45f39e40ab0ec0051c0d` | `epoch=14_val_mean_corpora_tasks_macro_recall=0.8321.pt` |
| Equal | 1 | 3 | 0.8825 / 0.8363 | 0.7521 / 0.8890 | 0.8400 | `F_weighted_parkinson_fold1_2026-11-06_14-58_carewsm_gated_fusion_309e95a2` | `bf9395ed33d640d28653102fc6620619` | `epoch=3_val_mean_corpora_tasks_macro_recall=0.8400.pt` |
| Equal | 2 | 11 | 0.7304 / 0.8558 | 0.7073 / 0.8979 | 0.7979 | `F_weighted_parkinson_fold2_2026-11-06_14-59_carewsm_gated_fusion_151e5750` | `4afa8dfef8144666a14e4a44017c77bb` | `epoch=11_val_mean_corpora_tasks_macro_recall=0.7979.pt` |
| Equal | 3 | 14 | 0.8506 / 0.7177 | 0.7164 / 0.9193 | 0.8010 | `F_weighted_parkinson_fold3_2026-11-06_15-01_carewsm_gated_fusion_f7e04dbe` | `93c47d48889b4445bc293cab24b40735` | `epoch=14_val_mean_corpora_tasks_macro_recall=0.8010.pt` |
| Equal | 4 | 7 | 0.8483 / 0.9116 | 0.8166 / 0.8805 | 0.8643 | `F_weighted_parkinson_fold4_2026-11-06_15-04_carewsm_gated_fusion_410a2700` | `9e15409719554d2daa5819df7b7b8546` | `epoch=7_val_mean_corpora_tasks_macro_recall=0.8643.pt` |
| Static-STCH | 0 | 8 | 0.8554 / 0.7926 | 0.7954 / 0.9020 | 0.8363 | `stch_fold0_2026-11-07_12-06_carewsm_gated_fusion_e7902c8d` | `276bb158801e4cd1b85aa99949d07071` | `epoch=8_val_mean_corpora_tasks_macro_recall=0.8363.pt` |
| Static-STCH | 1 | 3 | 0.8691 / 0.8426 | 0.7570 / 0.8847 | 0.8383 | `stch_fold1_2026-11-07_13-14_carewsm_gated_fusion_ed43ba64` | `091ae97c22424743ad3c2f1038ff3d4a` | `epoch=3_val_mean_corpora_tasks_macro_recall=0.8383.pt` |
| Static-STCH | 2 | 11 | 0.7284 / 0.8687 | 0.6992 / 0.9041 | 0.8001 | `stch_fold2_2026-11-07_13-15_carewsm_gated_fusion_c2bd2d9e` | `dd1d68a4be7c46068d75502fc784c4c0` | `epoch=11_val_mean_corpora_tasks_macro_recall=0.8001.pt` |
| Static-STCH | 3 | 18 | 0.8567 / 0.7160 | 0.7053 / 0.9147 | 0.7982 | `stch_fold3_2026-11-07_13-17_carewsm_gated_fusion_f353fa84` | `28a7ab859bb24ae084420b95be66890f` | `epoch=18_val_mean_corpora_tasks_macro_recall=0.7982.pt` |
| Static-STCH | 4 | 7 | 0.8564 / 0.9021 | 0.8122 / 0.8687 | 0.8599 | `stch_fold4_2026-11-07_13-20_carewsm_gated_fusion_9eb4affb` | `74399d3faae447adbc66425529052fe4` | `epoch=7_val_mean_corpora_tasks_macro_recall=0.8599.pt` |
| Progress | 0 | 9 | 0.8613 / 0.7595 | 0.7939 / 0.9127 | 0.8319 | `progress_corrected_fold0_2026-11-07_14-30_carewsm_gated_fusion_74b6398b` | `ae158a4ca28e48be90f5180bd2ae01df` | `epoch=9_val_mean_corpora_tasks_macro_recall=0.8319.pt` |
| Progress | 1 | 3 | 0.8825 / 0.8475 | 0.7453 / 0.8932 | 0.8421 | `progress_corrected_fold1_2026-11-07_14-32_carewsm_gated_fusion_66f14afe` | `f67b079b360e40f5a24b41a05bf2e1ad` | `epoch=3_val_mean_corpora_tasks_macro_recall=0.8421.pt` |
| Progress | 2 | 15 | 0.7258 / 0.8351 | 0.6939 / 0.8846 | 0.7848 | `progress_corrected_fold2_2026-11-07_14-33_carewsm_gated_fusion_67d0dc55` | `876017b8e3614c0490678463df90bd68` | `epoch=15_val_mean_corpora_tasks_macro_recall=0.7848.pt` |
| Progress | 3 | 14 | 0.8568 / 0.7177 | 0.7173 / 0.9209 | 0.8031 | `progress_corrected_fold3_2026-11-07_14-36_carewsm_gated_fusion_6a6310d3` | `63ec2b2dab46483594f25d423291d18c` | `epoch=14_val_mean_corpora_tasks_macro_recall=0.8031.pt` |
| Progress | 4 | 7 | 0.8412 / 0.9052 | 0.8185 / 0.8771 | 0.8605 | `progress_corrected_fold4_2026-11-07_14-38_carewsm_gated_fusion_bbc9bfb2` | `92e0f5f91bc74a4099ad91373f5d2502` | `epoch=7_val_mean_corpora_tasks_macro_recall=0.8605.pt` |
| RA-STCH | 0 | 8 | 0.8478 / 0.7911 | 0.7963 / 0.8948 | 0.8325 | `ra_stch_corrected_fold0_2026-11-07_14-40_carewsm_gated_fusion_b519cdd5` | `fc5c9a7f48c54148b8db141469175b37` | `epoch=8_val_mean_corpora_tasks_macro_recall=0.8325.pt` |
| RA-STCH | 1 | 3 | 0.8785 / 0.8410 | 0.7432 / 0.8862 | 0.8372 | `ra_stch_corrected_fold1_2026-11-07_14-41_carewsm_gated_fusion_73a81d4a` | `833ec87ccad34881af8b5e6403825587` | `epoch=3_val_mean_corpora_tasks_macro_recall=0.8372.pt` |
| RA-STCH | 2 | 11 | 0.7304 / 0.8124 | 0.7038 / 0.8880 | 0.7837 | `ra_stch_corrected_fold2_2026-11-07_14-43_carewsm_gated_fusion_8e581ac1` | `4850fa04ffa6458a802d11e9d89f6fc8` | `epoch=11_val_mean_corpora_tasks_macro_recall=0.7837.pt` |
| RA-STCH | 3 | 14 | 0.8506 / 0.7177 | 0.7111 / 0.9184 | 0.7994 | `ra_stch_corrected_fold3_2026-11-07_14-45_carewsm_gated_fusion_96caa0c4` | `cedb5b0f4f564274b346d270f1ffdea5` | `epoch=14_val_mean_corpora_tasks_macro_recall=0.7994.pt` |
| RA-STCH | 4 | 7 | 0.8468 / 0.8905 | 0.8146 / 0.8718 | 0.8559 | `ra_stch_corrected_fold4_2026-11-07_14-47_carewsm_gated_fusion_b360c316` | `96df82073f9d43fb81aac99c0eea3b8d` | `epoch=7_val_mean_corpora_tasks_macro_recall=0.8559.pt` |
| GradNorm | 0 | 8 | 0.8398 / 0.7732 | 0.8005 / 0.8920 | 0.8264 | `gradnorm_corrected_fold0_2026-11-07_14-50_carewsm_gated_fusion_9dc93c44` | `be8535dd860f4e6da92d04d78a2f65d1` | `epoch=8_val_mean_corpora_tasks_macro_recall=0.8264.pt` |
| GradNorm | 1 | 3 | 0.8825 / 0.8475 | 0.7442 / 0.8886 | 0.8407 | `gradnorm_corrected_fold1_2026-11-07_14-52_carewsm_gated_fusion_d242fef0` | `201bb24f68794809991dbdd2e4eafac3` | `epoch=3_val_mean_corpora_tasks_macro_recall=0.8407.pt` |
| GradNorm | 2 | 11 | 0.7284 / 0.8591 | 0.6988 / 0.8707 | 0.7893 | `gradnorm_corrected_fold2_2026-11-07_14-53_carewsm_gated_fusion_97b381c6` | `491c5bd634c5409aa7878ea900bd0316` | `epoch=11_val_mean_corpora_tasks_macro_recall=0.7893.pt` |
| GradNorm | 3 | 17 | 0.8586 / 0.7160 | 0.7206 / 0.9210 | 0.8040 | `gradnorm_corrected_fold3_2026-11-07_14-55_carewsm_gated_fusion_29f8aeae` | `f3e3dca3fa864a6cb3abfb87fd6b5e23` | `epoch=17_val_mean_corpora_tasks_macro_recall=0.8040.pt` |
| GradNorm | 4 | 7 | 0.8521 / 0.8774 | 0.8266 / 0.8681 | 0.8561 | `gradnorm_corrected_fold4_2026-11-07_14-58_carewsm_gated_fusion_cb0fb471` | `688a28433b3c447d995df011e836814d` | `epoch=7_val_mean_corpora_tasks_macro_recall=0.8561.pt` |
| PCGrad | 0 | 9 | 0.8535 / 0.7611 | 0.7932 / 0.9112 | 0.8297 | `pcgrad_fold0_2026-11-07_12-44_carewsm_gated_fusion_19409f5f` | `e241d06c55ee4b509979f62b7e87f6ee` | `epoch=9_val_mean_corpora_tasks_macro_recall=0.8297.pt` |
| PCGrad | 1 | 3 | 0.8805 / 0.8123 | 0.7482 / 0.8870 | 0.8320 | `pcgrad_fold1_2026-11-07_12-46_carewsm_gated_fusion_4c28fe99` | `b086cf2380f5440a8723bcdf7a7b92df` | `epoch=3_val_mean_corpora_tasks_macro_recall=0.8320.pt` |
| PCGrad | 2 | 11 | 0.7304 / 0.8703 | 0.7073 / 0.9026 | 0.8027 | `pcgrad_fold2_2026-11-07_12-47_carewsm_gated_fusion_eef4e882` | `2de59fa8a2504ca0a2f8200c776000c2` | `epoch=11_val_mean_corpora_tasks_macro_recall=0.8027.pt` |
| PCGrad | 3 | 18 | 0.8546 / 0.7316 | 0.7101 / 0.9158 | 0.8030 | `pcgrad_fold3_2026-11-07_12-49_carewsm_gated_fusion_944574ad` | `6738b37f02e34f44a12e87f193125d03` | `epoch=18_val_mean_corpora_tasks_macro_recall=0.8030.pt` |
| PCGrad | 4 | 7 | 0.8521 / 0.9000 | 0.8142 / 0.8783 | 0.8612 | `pcgrad_fold4_2026-11-07_12-52_carewsm_gated_fusion_c7b74aaa` | `ac9e55aed1af4c6c96c3682ba9d3834e` | `epoch=7_val_mean_corpora_tasks_macro_recall=0.8612.pt` |
| CAGrad | 0 | 9 | 0.8656 / 0.7438 | 0.8013 / 0.9043 | 0.8288 | `cagrad_fold0_2026-11-07_12-11_carewsm_gated_fusion_6b38a5a4` | `f0eade557acd4d5a82b2093953026315` | `epoch=9_val_mean_corpora_tasks_macro_recall=0.8288.pt` |
| CAGrad | 1 | 3 | 0.8844 / 0.8252 | 0.7452 / 0.8868 | 0.8354 | `cagrad_fold1_2026-11-07_12-13_carewsm_gated_fusion_cea93f19` | `e3f259d4448c42cdb80b2cecfcffac72` | `epoch=3_val_mean_corpora_tasks_macro_recall=0.8354.pt` |
| CAGrad | 2 | 11 | 0.7267 / 0.8430 | 0.7035 / 0.8930 | 0.7916 | `cagrad_fold2_2026-11-07_12-14_carewsm_gated_fusion_875bb5d8` | `f785cf770c23467fbce9455680825824` | `epoch=11_val_mean_corpora_tasks_macro_recall=0.7916.pt` |
| CAGrad | 3 | 12 | 0.8460 / 0.6535 | 0.7016 / 0.9281 | 0.7823 | `cagrad_fold3_2026-11-07_12-17_carewsm_gated_fusion_7ffe96a9` | `38a65526d8684df89d2c24dd0c7f0c02` | `epoch=12_val_mean_corpora_tasks_macro_recall=0.7823.pt` |
| CAGrad | 4 | 7 | 0.8571 / 0.8869 | 0.8142 / 0.8692 | 0.8568 | `cagrad_fold4_2026-11-07_12-19_carewsm_gated_fusion_f46f0efb` | `c1951c3bd4ff433aa7973a9129c1ad9f` | `epoch=7_val_mean_corpora_tasks_macro_recall=0.8568.pt` |
| DB-MTL | 0 | 11 | 0.8502 / 0.7849 | 0.8106 / 0.8202 | 0.8165 | `dbmtl_fold0_2026-11-07_12-21_carewsm_gated_fusion_cf9487cf` | `9577904d62e84d54bc4564f91ec69b7e` | `epoch=11_val_mean_corpora_tasks_macro_recall=0.8165.pt` |
| DB-MTL | 1 | 12 | 0.9028 / 0.9177 | 0.7682 / 0.8738 | 0.8656 | `dbmtl_fold1_2026-11-07_12-23_carewsm_gated_fusion_e5db9469` | `7add8f3965c347ba88f20facac5a66df` | `epoch=12_val_mean_corpora_tasks_macro_recall=0.8656.pt` |
| DB-MTL | 2 | 14 | 0.8429 / 0.8111 | 0.7276 / 0.8685 | 0.8125 | `dbmtl_fold2_2026-11-07_12-25_carewsm_gated_fusion_0faaee34` | `80a098e23c144683a7185c46d61ccb1c` | `epoch=14_val_mean_corpora_tasks_macro_recall=0.8125.pt` |
| DB-MTL | 3 | 9 | 0.8686 / 0.6993 | 0.7458 / 0.8410 | 0.7886 | `dbmtl_fold3_2026-11-07_12-27_carewsm_gated_fusion_bba77f53` | `ccd497e18e854c4caf8ea98075103df6` | `epoch=9_val_mean_corpora_tasks_macro_recall=0.7886.pt` |
| DB-MTL | 4 | 28 | 0.8714 / 0.9184 | 0.8084 / 0.9029 | 0.8752 | `dbmtl_fold4_2026-11-07_12-29_carewsm_gated_fusion_eb9e0ae2` | `047b19dc4cbd459eba57adf4b3fd574d` | `epoch=28_val_mean_corpora_tasks_macro_recall=0.8752.pt` |

The run directory, genuine MLflow run ID, and selected checkpoint are separate fields above. The prior Stage G Progress/RA-STCH/GradNorm runs in `logs/audio_representation_pilot/deep008_stageg_wsm/` are historical **SUPERSEDED** artifacts; they are preserved and are not used in this ledger.

## Five-fold validation evidence

Values are mean ± sample SD over folds. The selection monitor is the arithmetic mean of the four VAL Macro Recall components shown in the ledger. TEST is descriptive only.

| Method | VAL selection | CARE D Macro Recall | CARE P Macro Recall | WSM D Macro Recall | WSM P Macro Recall |
|---|---:|---:|---:|---:|---:|
| Equal | 0.8271 ± 0.0279 | 0.8264 ± 0.0580 | 0.8227 ± 0.0727 | 0.7562 ± 0.0466 | 0.9028 ± 0.0198 |
| Static-STCH | 0.8266 ± 0.0267 | 0.8332 ± 0.0589 | 0.8244 ± 0.0726 | 0.7538 ± 0.0512 | 0.8948 ± 0.0181 |
| Progress | 0.8245 ± 0.0304 | 0.8335 ± 0.0620 | 0.8130 ± 0.0744 | 0.7538 ± 0.0519 | 0.8977 ± 0.0186 |
| RA-STCH | 0.8217 ± 0.0294 | 0.8308 ± 0.0576 | 0.8105 ± 0.0639 | 0.7538 ± 0.0498 | 0.8918 ± 0.0170 |
| GradNorm | 0.8233 ± 0.0270 | 0.8323 ± 0.0601 | 0.8146 ± 0.0679 | 0.7581 ± 0.0539 | 0.8881 ± 0.0212 |
| PCGrad | 0.8257 ± 0.0243 | 0.8342 ± 0.0592 | 0.8151 ± 0.0710 | 0.7546 ± 0.0482 | 0.8990 ± 0.0159 |
| CAGrad | 0.8190 ± 0.0312 | 0.8360 ± 0.0627 | 0.7905 ± 0.0925 | 0.7532 ± 0.0530 | 0.8963 ± 0.0219 |
| DB-MTL | 0.8317 ± 0.0371 | 0.8672 ± 0.0233 | 0.8263 ± 0.0934 | 0.7721 ± 0.0370 | 0.8613 ± 0.0318 |

Paired five-fold VAL differences versus Equal, computed fold by fold from the validation selection monitor:

| Method | Mean difference ± SD | Wins | Losses |
|---|---:|---:|---:|
| Static-STCH | -0.0005 ± 0.0036 | 2 | 3 |
| Progress | -0.0026 ± 0.0064 | 2 | 3 |
| RA-STCH | -0.0053 ± 0.0059 | 1 | 4 |
| GradNorm | -0.0038 ± 0.0053 | 2 | 3 |
| PCGrad | -0.0013 ± 0.0049 | 2 | 3 |
| CAGrad | -0.0081 ± 0.0061 | 0 | 5 |
| DB-MTL | +0.0046 ± 0.0179 | 3 | 2 |

## Canonical six-metric and Parkinson tradeoff evidence

Every selected run retains the canonical six metrics for CARE VAL, WSM VAL, CARE TEST and WSM TEST in its wide `summary.txt`: Macro F1, Weighted F1, Macro Recall, Weighted Recall, Patient Recall and Healthy Recall, with native corpus sample counts. The following aggregate table records the requested Parkinson tradeoff blocks; values are mean ± SD over folds.

| Method | CARE TEST Parkinson Macro F1 | Weighted F1 | Macro Recall | Weighted Recall | Patient Recall | Healthy Recall | WSM TEST Parkinson Macro Recall | Patient Recall | Healthy Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equal | 0.7725 ± 0.0358 | 0.9140 ± 0.0134 | 0.7687 ± 0.0498 | 0.9151 ± 0.0148 | 0.5824 ± 0.1089 | 0.9549 ± 0.0195 | 0.7906 ± 0.0273 | 0.7248 ± 0.0492 | 0.8564 ± 0.0424 |
| Static-STCH | 0.7829 ± 0.0510 | 0.9164 ± 0.0190 | 0.7897 ± 0.0588 | 0.9157 ± 0.0188 | 0.6294 ± 0.1132 | 0.9500 ± 0.0125 | 0.8007 ± 0.0236 | 0.7564 ± 0.0274 | 0.8451 ± 0.0401 |
| Progress | 0.7789 ± 0.0413 | 0.9169 ± 0.0157 | 0.7704 ± 0.0429 | 0.9182 ± 0.0162 | 0.5824 ± 0.0842 | 0.9584 ± 0.0148 | 0.7942 ± 0.0346 | 0.7369 ± 0.0466 | 0.8515 ± 0.0445 |
| RA-STCH | 0.7817 ± 0.0421 | 0.9165 ± 0.0165 | 0.7823 ± 0.0419 | 0.9163 ± 0.0170 | 0.6118 ± 0.0762 | 0.9528 ± 0.0124 | 0.8067 ± 0.0244 | 0.7579 ± 0.0268 | 0.8554 ± 0.0391 |
| GradNorm | 0.7607 ± 0.0240 | 0.9089 ± 0.0091 | 0.7603 ± 0.0408 | 0.9094 ± 0.0114 | 0.5706 ± 0.0944 | 0.9500 ± 0.0187 | 0.8047 ± 0.0268 | 0.7504 ± 0.0381 | 0.8589 ± 0.0722 |
| PCGrad | 0.7728 ± 0.0378 | 0.9153 ± 0.0134 | 0.7623 ± 0.0523 | 0.9176 ± 0.0140 | 0.5647 ± 0.1128 | 0.9598 ± 0.0177 | 0.7933 ± 0.0187 | 0.7248 ± 0.0556 | 0.8619 ± 0.0442 |
| CAGrad | 0.7740 ± 0.0368 | 0.9152 ± 0.0152 | 0.7645 ± 0.0470 | 0.9170 ± 0.0179 | 0.5706 ± 0.1031 | 0.9585 ± 0.0230 | 0.8073 ± 0.0130 | 0.7353 ± 0.0312 | 0.8792 ± 0.0489 |
| DB-MTL | 0.7441 ± 0.0602 | 0.8902 ± 0.0361 | 0.7930 ± 0.0512 | 0.8799 ± 0.0478 | 0.6823 ± 0.1165 | 0.9035 ± 0.0588 | 0.7977 ± 0.0420 | 0.7474 ± 0.1057 | 0.8480 ± 0.0564 |

TEST metrics above are statistics only and did not affect method correction, checkpoint selection, or rerun decisions.

## Verified diagnostics

The existing callback now persists loss diagnostics through the standard MLflow logger. A one-epoch corrected Progress Chimera smoke run completed successfully with genuine run ID `f6312bf888a84007856fad0783dbf5be` and persisted eight `mtl/*` metrics, including task losses, scalar weights, controller weights, and VAL feedback. Corrected production runs persist the same standard diagnostics; representative method-specific keys include `mtl/gradient_norm_ema_depression`, `mtl/gradient_norm_ema_parkinson`, `mtl/gradient_cosine_ema`, `mtl/weight_depression`, `mtl/weight_parkinson`, and `mtl/dbmtl_cosine` where applicable.

Focused numerical tests demonstrate that changing RA shared-gradient norm/cosine changes subsequent controller weights. Tests also cover STCH scalarization, PCGrad/CAGrad/DB-MTL invariants, GradNorm pre-update weighting and private-head application, exact Equal parity, masking, strict two-corpus VAL feedback, and TEST-feedback independence.

## Historical and integrity notes

- The original flawed Progress/RA-STCH/GradNorm runs are preserved as **SUPERSEDED**, not overwritten.
- Failed pre-correction attempts and the GradNorm helper-collision attempt are not counted as valid cells.
- No feature extraction, split regeneration, new trainer, logger, metric framework, callback framework, or checkpoint subsystem was introduced.
- Full six-metric fold-level evidence remains in the selected run summaries and MLflow tracking databases under the standard log roots.
