# AV-008 Final Fusion Research Report

`UPSTREAM_PROVENANCE=UNKNOWN — EXPLORATORY RESULTS ONLY`

## 1. Executive summary and key findings

The verified sweep contains **285/285 runs**, **57/57 five-fold variants**, and **48 metrics per run** (24 VAL + 24 TEST). All TEST results are descriptive and were evaluated from the original selected VAL checkpoints; no TEST value was used for selection.

The best prespecified configuration is `bimodal_shared_mtl__A2_T10` (`audio_id2+text_id10`), with VAL Overall UAR **0.8218 ± 0.0327**. Its descriptive TEST Overall UAR is **0.7819 ± 0.0259**.

## 2. Experiment design and data provenance

The matrix contains 165 bimodal jobs (3 architectures × 11 compatible feature selections × 5 folds) and 120 trimodal jobs (4 architectures × 6 selections × 5 folds). Selection is the five-fold mean VAL Overall UAR. Fold aggregation uses arithmetic mean and sample SD (`ddof=1`).

## 3. Completion and verification

All runtime identities, selected checkpoints, VAL monitors, config hashes, recovery provenance records and 48 metric fields passed verification. No participant-level records or source paths are included in this report. The local collector artifacts remain under `logs/` and are not committed.

## 4. Full VAL results

The complete six-metric VAL tables follow in the collector-derived section below.

## 5. Full descriptive TEST results

The complete six-metric TEST tables follow below. They are not a selection signal.

## 6. Bimodal and trimodal architecture comparisons

| Architecture | Variants | Mean VAL Overall UAR | Best variant |
|---|---:|---:|---|
| `audio_guided` | 6 | 0.6863 | `audio_guided__A1_V4_T10` |
| `bimodal_gated_late` | 11 | 0.7389 | `bimodal_gated_late__A2_T10` |
| `bimodal_shared_mtl` | 11 | 0.7469 | `bimodal_shared_mtl__A2_T10` |
| `bimodal_task_aware` | 11 | 0.7492 | `bimodal_task_aware__A2_T10` |
| `cross_attention` | 6 | 0.6977 | `cross_attention__A2_V4_T9` |
| `shared_private` | 6 | 0.6957 | `shared_private__A4_V4_T10` |
| `text_guided` | 6 | 0.6817 | `text_guided__A1_V4_T10` |

These architecture means average compatible feature variants and are descriptive; feature coverage differs by modality family. The complete split-by-task tables are retained below.

## 7. Audio-Guided vs Text-Guided

The matched table contains 30 paired folds (six feature combinations × five folds). Differences are Audio-Guided minus Text-Guided; no significance claim is made.

| Metric | Mean difference ± sample SD | Audio wins | Text wins | Ties |
|---|---:|---:|---:|---:|
| `val_overall_uar` | +0.0047 ± 0.0073 | 26 | 4 | 0 |
| `test_overall_uar` | +0.0024 ± 0.0083 | 18 | 12 | 0 |
| `val_care_depression_macro_recall` | +0.0124 ± 0.0126 | 25 | 4 | 1 |
| `val_care_parkinson_macro_recall` | -0.0005 ± 0.0327 | 16 | 10 | 4 |
| `val_wsm_depression_macro_recall` | +0.0031 ± 0.0146 | 21 | 9 | 0 |
| `val_wsm_parkinson_macro_recall` | +0.0036 ± 0.0139 | 22 | 7 | 1 |
| `test_care_depression_macro_recall` | +0.0063 ± 0.0155 | 20 | 8 | 2 |
| `test_care_parkinson_macro_recall` | +0.0033 ± 0.0195 | 14 | 11 | 5 |
| `test_wsm_depression_macro_recall` | +0.0058 ± 0.0112 | 23 | 7 | 0 |
| `test_wsm_parkinson_macro_recall` | -0.0058 ± 0.0209 | 13 | 17 | 0 |

## 8. Audio/text feature-version and modality ablations

`bimodal_comparison.csv`, `trimodal_comparison.csv`, `feature_ablation.csv`, and `guided_comparison_summary.csv` are matched-design summaries. Audio-version comparisons are interpreted only within the same architecture, modality set, text version and fold; text-version comparisons are interpreted only within the same architecture, audio version and fold. Overall modality-family means must not be read as causal feature effects because architecture and input set also change.

## 9. Best models by overall and individual tasks

Task-specific rows use VAL Macro Recall as the exploratory task metric; they are not replacements for the prespecified overall ranking.

| Category | VAL selection metric | Experiment | Features | VAL mean ± SD | TEST Overall UAR mean ± SD |
|---|---|---|---|---:|---:|
| overall | val_overall_uar_mean | `bimodal_shared_mtl__A2_T10` | `audio_id2+text_id10` | 0.8218 ± 0.0327 | 0.7819 ± 0.0259 |
| bimodal | val_overall_uar_mean | `bimodal_shared_mtl__A2_T10` | `audio_id2+text_id10` | 0.8218 ± 0.0327 | 0.7819 ± 0.0259 |
| trimodal | val_overall_uar_mean | `cross_attention__A2_V4_T9` | `audio_id2+video_4+text_id9` | 0.6995 ± 0.0465 | 0.6695 ± 0.0275 |
| audio_video | val_overall_uar_mean | `bimodal_shared_mtl__A1_V4` | `audio_id1+video_4` | 0.7048 ± 0.0402 | 0.6729 ± 0.0214 |
| audio_text | val_overall_uar_mean | `bimodal_shared_mtl__A2_T10` | `audio_id2+text_id10` | 0.8218 ± 0.0327 | 0.7819 ± 0.0259 |
| video_text | val_overall_uar_mean | `bimodal_gated_late__V4_T10` | `video_4+text_id10` | 0.6224 ± 0.0418 | 0.5928 ± 0.0227 |
| care_depression | val_care_depression_macro_recall_mean | `bimodal_gated_late__A1_T9` | `audio_id1+text_id9` | 0.8481 ± 0.0639 | 0.7721 ± 0.0186 |
| care_parkinson | val_care_parkinson_macro_recall_mean | `bimodal_shared_mtl__A2_T10` | `audio_id2+text_id10` | 0.8097 ± 0.1068 | 0.7819 ± 0.0259 |
| wsm_depression | val_wsm_depression_macro_recall_mean | `bimodal_gated_late__A4_T9` | `audio_id4+text_id9` | 0.7574 ± 0.0283 | 0.7691 ± 0.0255 |
| wsm_parkinson | val_wsm_parkinson_macro_recall_mean | `bimodal_shared_mtl__A1_T10` | `audio_id1+text_id10` | 0.9052 ± 0.0123 | 0.7656 ± 0.0245 |

## 10. Interpretation, limitations and next research questions

These are exploratory results. Upstream FIT/SELECTION identities remain unknown. TEST measurements were not used to choose features, architectures, checkpoints or winners. No leakage-free generalization, clinical validity or statistical significance claim is made. Follow-up work should pre-register any confirmatory comparison and preserve the validation-only decision rule.

## 11. Reproducibility

The report is generated from `logs/fusion/sweep/reports/*.csv` by `scripts/fusion/final_report.py`; the collector is `scripts/fusion/collect_results.py`. Figures are deterministic SVG files under `reports/av_fusion_sweep/figures/`. Training logs, checkpoints, raw features and MLflow databases are excluded from Git.

## 12. Figures

- `figures/val_overall_uar_top15.svg`
- `figures/architecture_val_means.svg`
- `figures/audio_minus_text_overall.svg`

## Detailed VAL metrics — five-fold mean ± sample SD

### Care Depression — VAL

| Model | Feature versions | Macro F1 | Weighted F1 | Macro Recall / UAR | Weighted Recall | Patient Recall | Healthy Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id10` | 0.8368 ± 0.0537 | 0.8803 ± 0.0378 | 0.8330 ± 0.0649 | 0.8822 ± 0.0348 | 0.7353 ± 0.1334 | 0.9307 ± 0.0275 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id10` | 0.8410 ± 0.0445 | 0.8833 ± 0.0296 | 0.8404 ± 0.0667 | 0.8853 ± 0.0250 | 0.7508 ± 0.1553 | 0.9300 ± 0.0287 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id9` | 0.8394 ± 0.0408 | 0.8822 ± 0.0277 | 0.8362 ± 0.0570 | 0.8841 ± 0.0240 | 0.7408 ± 0.1286 | 0.9315 ± 0.0235 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id10` | 0.8375 ± 0.0352 | 0.8806 ± 0.0230 | 0.8368 ± 0.0566 | 0.8822 ± 0.0188 | 0.7462 ± 0.1386 | 0.9274 ± 0.0306 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id9` | 0.8356 ± 0.0382 | 0.8794 ± 0.0255 | 0.8329 ± 0.0581 | 0.8816 ± 0.0216 | 0.7361 ± 0.1394 | 0.9298 ± 0.0310 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id10` | 0.8436 ± 0.0535 | 0.8854 ± 0.0374 | 0.8421 ± 0.0739 | 0.8874 ± 0.0346 | 0.7519 ± 0.1567 | 0.9323 ± 0.0228 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id9` | 0.8408 ± 0.0543 | 0.8832 ± 0.0389 | 0.8380 ± 0.0685 | 0.8849 ± 0.0371 | 0.7447 ± 0.1379 | 0.9313 ± 0.0263 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id9` | 0.8389 ± 0.0379 | 0.8814 ± 0.0252 | 0.8388 ± 0.0572 | 0.8829 ± 0.0215 | 0.7510 ± 0.1371 | 0.9266 ± 0.0312 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id9` | 0.8531 ± 0.0523 | 0.8924 ± 0.0374 | 0.8481 ± 0.0639 | 0.8941 ± 0.0357 | 0.7568 ± 0.1262 | 0.9395 ± 0.0246 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id10` | 0.8512 ± 0.0473 | 0.8908 ± 0.0334 | 0.8478 ± 0.0618 | 0.8924 ± 0.0313 | 0.7592 ± 0.1289 | 0.9364 ± 0.0244 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id10` | 0.8436 ± 0.0500 | 0.8849 ± 0.0355 | 0.8418 ± 0.0625 | 0.8863 ± 0.0329 | 0.7537 ± 0.1223 | 0.9298 ± 0.0080 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id10` | 0.8298 ± 0.0599 | 0.8757 ± 0.0410 | 0.8262 ± 0.0739 | 0.8783 ± 0.0365 | 0.7227 ± 0.1539 | 0.9297 ± 0.0247 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id9` | 0.8293 ± 0.0579 | 0.8755 ± 0.0395 | 0.8255 ± 0.0752 | 0.8783 ± 0.0348 | 0.7205 ± 0.1596 | 0.9306 ± 0.0207 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id9` | 0.8307 ± 0.0598 | 0.8777 ± 0.0396 | 0.8220 ± 0.0758 | 0.8820 ± 0.0333 | 0.7029 ± 0.1667 | 0.9412 ± 0.0279 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id10` | 0.8269 ± 0.0504 | 0.8731 ± 0.0335 | 0.8260 ± 0.0691 | 0.8754 ± 0.0284 | 0.7279 ± 0.1540 | 0.9241 ± 0.0227 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id9` | 0.8342 ± 0.0488 | 0.8781 ± 0.0347 | 0.8317 ± 0.0609 | 0.8796 ± 0.0322 | 0.7368 ± 0.1192 | 0.9266 ± 0.0079 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id10` | 0.8300 ± 0.0560 | 0.8761 ± 0.0380 | 0.8249 ± 0.0698 | 0.8789 ± 0.0335 | 0.7177 ± 0.1495 | 0.9321 ± 0.0270 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id9` | 0.8314 ± 0.0444 | 0.8761 ± 0.0308 | 0.8294 ± 0.0595 | 0.8778 ± 0.0278 | 0.7332 ± 0.1276 | 0.9256 ± 0.0205 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+video_4` | 0.7234 ± 0.0640 | 0.8048 ± 0.0432 | 0.7065 ± 0.0651 | 0.8166 ± 0.0375 | 0.4897 ± 0.1264 | 0.9233 ± 0.0229 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+video_4` | 0.7261 ± 0.0467 | 0.8055 ± 0.0318 | 0.7100 ± 0.0483 | 0.8154 ± 0.0285 | 0.5026 ± 0.0971 | 0.9174 ± 0.0244 |
| `carewsm_bimodal_task_aware` | `audio_id1+video_4` | 0.7037 ± 0.0508 | 0.7860 ± 0.0378 | 0.6934 ± 0.0471 | 0.7925 ± 0.0370 | 0.4980 ± 0.0723 | 0.8888 ± 0.0302 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id9` | 0.7144 ± 0.0588 | 0.7986 ± 0.0378 | 0.6992 ± 0.0597 | 0.8111 ± 0.0290 | 0.4784 ± 0.1240 | 0.9201 ± 0.0142 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id10` | 0.7057 ± 0.0625 | 0.7935 ± 0.0418 | 0.6889 ± 0.0577 | 0.8078 ± 0.0362 | 0.4543 ± 0.1175 | 0.9236 ± 0.0376 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id10` | 0.7147 ± 0.0531 | 0.7972 ± 0.0344 | 0.7024 ± 0.0546 | 0.8079 ± 0.0285 | 0.4940 ± 0.1232 | 0.9108 ± 0.0356 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id10` | 0.7140 ± 0.0596 | 0.7958 ± 0.0402 | 0.7022 ± 0.0602 | 0.8053 ± 0.0336 | 0.4984 ± 0.1194 | 0.9059 ± 0.0203 |
| `carewsm_bimodal_task_aware` | `audio_id4+video_4` | 0.7081 ± 0.0351 | 0.7917 ± 0.0245 | 0.6942 ± 0.0345 | 0.8012 ± 0.0222 | 0.4831 ± 0.0676 | 0.9053 ± 0.0213 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id10` | 0.7238 ± 0.0661 | 0.8029 ± 0.0446 | 0.7112 ± 0.0678 | 0.8120 ± 0.0388 | 0.5128 ± 0.1295 | 0.9096 ± 0.0185 |
| `carewsm_bimodal_task_aware` | `audio_id2+video_4` | 0.7134 ± 0.0501 | 0.7933 ± 0.0358 | 0.7032 ± 0.0484 | 0.8004 ± 0.0333 | 0.5111 ± 0.0928 | 0.8953 ± 0.0320 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id9` | 0.7081 ± 0.0609 | 0.7943 ± 0.0427 | 0.6901 ± 0.0564 | 0.8070 ± 0.0389 | 0.4602 ± 0.0951 | 0.9201 ± 0.0239 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id9` | 0.7041 ± 0.0505 | 0.7907 ± 0.0338 | 0.6895 ± 0.0512 | 0.8031 ± 0.0269 | 0.4654 ± 0.1075 | 0.9137 ± 0.0184 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id10` | 0.7050 ± 0.0460 | 0.7886 ± 0.0286 | 0.6962 ± 0.0526 | 0.7976 ± 0.0192 | 0.4960 ± 0.1239 | 0.8965 ± 0.0210 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id9` | 0.7051 ± 0.0392 | 0.7884 ± 0.0266 | 0.6941 ± 0.0402 | 0.7968 ± 0.0222 | 0.4911 ± 0.0855 | 0.8970 ± 0.0210 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id10` | 0.6988 ± 0.0666 | 0.7854 ± 0.0453 | 0.6876 ± 0.0657 | 0.7958 ± 0.0376 | 0.4739 ± 0.1247 | 0.9012 ± 0.0106 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id9` | 0.7213 ± 0.0655 | 0.8016 ± 0.0448 | 0.7073 ± 0.0653 | 0.8112 ± 0.0397 | 0.5028 ± 0.1204 | 0.9118 ± 0.0206 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id9` | 0.6917 ± 0.0593 | 0.7794 ± 0.0388 | 0.6831 ± 0.0610 | 0.7890 ± 0.0304 | 0.4737 ± 0.1267 | 0.8923 ± 0.0159 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+video_4` | 0.6885 ± 0.0454 | 0.7812 ± 0.0313 | 0.6717 ± 0.0422 | 0.7957 ± 0.0277 | 0.4278 ± 0.0780 | 0.9158 ± 0.0209 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6965 ± 0.0665 | 0.7832 ± 0.0465 | 0.6843 ± 0.0625 | 0.7929 ± 0.0433 | 0.4703 ± 0.1081 | 0.8982 ± 0.0324 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6974 ± 0.0634 | 0.7862 ± 0.0431 | 0.6828 ± 0.0633 | 0.7988 ± 0.0379 | 0.4543 ± 0.1179 | 0.9113 ± 0.0199 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6835 ± 0.0700 | 0.7757 ± 0.0482 | 0.6710 ± 0.0662 | 0.7880 ± 0.0433 | 0.4404 ± 0.1185 | 0.9016 ± 0.0297 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.7100 ± 0.0490 | 0.7960 ± 0.0320 | 0.6933 ± 0.0509 | 0.8092 ± 0.0248 | 0.4648 ± 0.1065 | 0.9217 ± 0.0105 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6919 ± 0.0694 | 0.7833 ± 0.0473 | 0.6768 ± 0.0670 | 0.7973 ± 0.0408 | 0.4395 ± 0.1227 | 0.9141 ± 0.0176 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6938 ± 0.0539 | 0.7837 ± 0.0375 | 0.6786 ± 0.0526 | 0.7964 ± 0.0340 | 0.4468 ± 0.0965 | 0.9104 ± 0.0227 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.7066 ± 0.0485 | 0.7896 ± 0.0349 | 0.6945 ± 0.0461 | 0.7979 ± 0.0333 | 0.4906 ± 0.0824 | 0.8983 ± 0.0301 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6823 ± 0.0716 | 0.7764 ± 0.0491 | 0.6681 ± 0.0683 | 0.7907 ± 0.0434 | 0.4269 ± 0.1221 | 0.9093 ± 0.0221 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6931 ± 0.0600 | 0.7837 ± 0.0417 | 0.6773 ± 0.0576 | 0.7970 ± 0.0380 | 0.4418 ± 0.1031 | 0.9129 ± 0.0254 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6869 ± 0.0581 | 0.7776 ± 0.0399 | 0.6748 ± 0.0550 | 0.7894 ± 0.0347 | 0.4486 ± 0.1072 | 0.9010 ± 0.0299 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6881 ± 0.0574 | 0.7806 ± 0.0397 | 0.6724 ± 0.0545 | 0.7946 ± 0.0361 | 0.4318 ± 0.0990 | 0.9129 ± 0.0262 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6894 ± 0.0638 | 0.7831 ± 0.0432 | 0.6723 ± 0.0600 | 0.7992 ± 0.0361 | 0.4223 ± 0.1092 | 0.9224 ± 0.0108 |
| `carewsm_bimodal_gated_late` | `audio_id1+video_4` | 0.6414 ± 0.0520 | 0.7475 ± 0.0363 | 0.6302 ± 0.0458 | 0.7632 ± 0.0344 | 0.3681 ± 0.0750 | 0.8924 ± 0.0317 |
| `carewsm_bimodal_gated_late` | `audio_id4+video_4` | 0.6320 ± 0.0613 | 0.7379 ± 0.0429 | 0.6239 ± 0.0570 | 0.7501 ± 0.0398 | 0.3754 ± 0.0947 | 0.8726 ± 0.0282 |
| `carewsm_bimodal_gated_late` | `audio_id2+video_4` | 0.6465 ± 0.0682 | 0.7502 ± 0.0486 | 0.6354 ± 0.0610 | 0.7647 ± 0.0469 | 0.3804 ± 0.0968 | 0.8903 ± 0.0407 |
| `carewsm_bimodal_gated_late` | `video_4+text_id10` | 0.6170 ± 0.0680 | 0.7214 ± 0.0466 | 0.6157 ± 0.0720 | 0.7276 ± 0.0417 | 0.3958 ± 0.1372 | 0.8357 ± 0.0201 |
| `carewsm_bimodal_task_aware` | `video_4+text_id9` | 0.6185 ± 0.0726 | 0.7233 ± 0.0503 | 0.6158 ± 0.0727 | 0.7301 ± 0.0467 | 0.3907 ± 0.1277 | 0.8407 ± 0.0263 |
| `carewsm_bimodal_task_aware` | `video_4+text_id10` | 0.6253 ± 0.0700 | 0.7271 ± 0.0494 | 0.6226 ± 0.0701 | 0.7326 ± 0.0464 | 0.4062 ± 0.1208 | 0.8391 ± 0.0285 |
| `carewsm_bimodal_gated_late` | `video_4+text_id9` | 0.6144 ± 0.0670 | 0.7201 ± 0.0475 | 0.6119 ± 0.0681 | 0.7270 ± 0.0451 | 0.3856 ± 0.1233 | 0.8381 ± 0.0325 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id9` | 0.6124 ± 0.0687 | 0.7193 ± 0.0495 | 0.6093 ± 0.0694 | 0.7270 ± 0.0483 | 0.3781 ± 0.1232 | 0.8405 ± 0.0385 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id10` | 0.6129 ± 0.0665 | 0.7201 ± 0.0476 | 0.6090 ± 0.0656 | 0.7281 ± 0.0462 | 0.3749 ± 0.1124 | 0.8431 ± 0.0352 |

### Care Parkinson — VAL

| Model | Feature versions | Macro F1 | Weighted F1 | Macro Recall / UAR | Weighted Recall | Patient Recall | Healthy Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id10` | 0.8147 ± 0.0958 | 0.9338 ± 0.0303 | 0.8097 ± 0.1068 | 0.9358 ± 0.0269 | 0.6509 ± 0.2134 | 0.9687 ± 0.0209 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id10` | 0.7997 ± 0.0951 | 0.9276 ± 0.0292 | 0.8031 ± 0.1065 | 0.9290 ± 0.0256 | 0.6444 ± 0.2161 | 0.9618 ± 0.0228 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id9` | 0.7964 ± 0.0967 | 0.9276 ± 0.0286 | 0.7931 ± 0.1078 | 0.9302 ± 0.0235 | 0.6203 ± 0.2202 | 0.9659 ± 0.0209 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id10` | 0.7943 ± 0.1117 | 0.9257 ± 0.0356 | 0.8007 ± 0.1223 | 0.9270 ± 0.0322 | 0.6418 ± 0.2423 | 0.9597 ± 0.0234 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id9` | 0.8010 ± 0.0999 | 0.9285 ± 0.0312 | 0.8030 ± 0.1129 | 0.9301 ± 0.0276 | 0.6429 ± 0.2259 | 0.9631 ± 0.0205 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id10` | 0.7868 ± 0.1004 | 0.9233 ± 0.0311 | 0.7861 ± 0.1154 | 0.9261 ± 0.0283 | 0.6089 ± 0.2423 | 0.9634 ± 0.0326 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id9` | 0.7867 ± 0.0966 | 0.9235 ± 0.0285 | 0.7894 ± 0.1251 | 0.9268 ± 0.0240 | 0.6154 ± 0.2673 | 0.9635 ± 0.0312 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id9` | 0.7928 ± 0.1040 | 0.9269 ± 0.0318 | 0.7849 ± 0.1167 | 0.9302 ± 0.0271 | 0.6017 ± 0.2341 | 0.9680 ± 0.0197 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id9` | 0.7776 ± 0.0848 | 0.9222 ± 0.0230 | 0.7648 ± 0.1034 | 0.9272 ± 0.0167 | 0.5596 ± 0.2194 | 0.9701 ± 0.0216 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id10` | 0.7765 ± 0.0830 | 0.9217 ± 0.0225 | 0.7645 ± 0.1025 | 0.9266 ± 0.0172 | 0.5596 ± 0.2194 | 0.9695 ± 0.0242 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id10` | 0.7736 ± 0.1101 | 0.9204 ± 0.0312 | 0.7701 ± 0.1283 | 0.9254 ± 0.0238 | 0.5734 ± 0.2706 | 0.9668 ± 0.0277 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id10` | 0.7992 ± 0.0805 | 0.9279 ± 0.0250 | 0.7970 ± 0.1031 | 0.9297 ± 0.0222 | 0.6302 ± 0.2074 | 0.9638 ± 0.0133 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id9` | 0.8074 ± 0.0801 | 0.9319 ± 0.0254 | 0.7908 ± 0.0884 | 0.9346 ± 0.0232 | 0.6095 ± 0.1762 | 0.9721 ± 0.0194 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id9` | 0.7966 ± 0.0979 | 0.9302 ± 0.0304 | 0.7746 ± 0.1161 | 0.9351 ± 0.0268 | 0.5731 ± 0.2333 | 0.9762 ± 0.0194 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id10` | 0.7985 ± 0.0963 | 0.9289 ± 0.0299 | 0.7898 ± 0.1156 | 0.9320 ± 0.0258 | 0.6110 ± 0.2310 | 0.9686 ± 0.0141 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id9` | 0.7840 ± 0.1068 | 0.9249 ± 0.0303 | 0.7751 ± 0.1260 | 0.9303 ± 0.0223 | 0.5788 ± 0.2601 | 0.9714 ± 0.0181 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id10` | 0.7953 ± 0.0904 | 0.9290 ± 0.0280 | 0.7748 ± 0.1053 | 0.9332 ± 0.0249 | 0.5755 ± 0.2094 | 0.9741 ± 0.0166 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id9` | 0.7833 ± 0.0883 | 0.9246 ± 0.0275 | 0.7618 ± 0.0979 | 0.9290 ± 0.0249 | 0.5514 ± 0.1962 | 0.9721 ± 0.0200 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+video_4` | 0.6302 ± 0.1171 | 0.8803 ± 0.0398 | 0.6057 ± 0.0921 | 0.9013 ± 0.0302 | 0.2318 ± 0.1777 | 0.9796 ± 0.0131 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+video_4` | 0.6555 ± 0.1044 | 0.8865 ± 0.0356 | 0.6311 ± 0.0928 | 0.9041 ± 0.0240 | 0.2855 ± 0.1861 | 0.9767 ± 0.0047 |
| `carewsm_bimodal_task_aware` | `audio_id1+video_4` | 0.6642 ± 0.0914 | 0.8914 ± 0.0300 | 0.6307 ± 0.0791 | 0.9103 ± 0.0207 | 0.2771 ± 0.1579 | 0.9843 ± 0.0053 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id9` | 0.6499 ± 0.1000 | 0.8868 ± 0.0299 | 0.6253 ± 0.0887 | 0.9064 ± 0.0179 | 0.2698 ± 0.1831 | 0.9808 ± 0.0071 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id10` | 0.6465 ± 0.0610 | 0.8852 ± 0.0221 | 0.6158 ± 0.0532 | 0.9042 ± 0.0153 | 0.2515 ± 0.1093 | 0.9802 ± 0.0069 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id10` | 0.6549 ± 0.0876 | 0.8878 ± 0.0277 | 0.6286 ± 0.0823 | 0.9065 ± 0.0162 | 0.2771 ± 0.1711 | 0.9801 ± 0.0068 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id10` | 0.6490 ± 0.0605 | 0.8860 ± 0.0234 | 0.6163 ± 0.0502 | 0.9047 ± 0.0173 | 0.2517 ± 0.0992 | 0.9808 ± 0.0052 |
| `carewsm_bimodal_task_aware` | `audio_id4+video_4` | 0.6306 ± 0.0616 | 0.8814 ± 0.0220 | 0.6014 ± 0.0488 | 0.9024 ± 0.0165 | 0.2212 ± 0.0989 | 0.9816 ± 0.0099 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id10` | 0.6467 ± 0.1077 | 0.8860 ± 0.0361 | 0.6197 ± 0.0923 | 0.9067 ± 0.0247 | 0.2564 ± 0.1842 | 0.9829 ± 0.0067 |
| `carewsm_bimodal_task_aware` | `audio_id2+video_4` | 0.6602 ± 0.0822 | 0.8896 ± 0.0249 | 0.6315 ± 0.0783 | 0.9078 ± 0.0150 | 0.2821 ± 0.1624 | 0.9809 ± 0.0079 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id9` | 0.6489 ± 0.0839 | 0.8865 ± 0.0267 | 0.6187 ± 0.0688 | 0.9058 ± 0.0183 | 0.2559 ± 0.1377 | 0.9815 ± 0.0056 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id9` | 0.6417 ± 0.0689 | 0.8842 ± 0.0257 | 0.6109 ± 0.0562 | 0.9041 ± 0.0185 | 0.2403 ± 0.1119 | 0.9815 ± 0.0056 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id10` | 0.6600 ± 0.0908 | 0.8899 ± 0.0271 | 0.6319 ± 0.0885 | 0.9084 ± 0.0170 | 0.2823 ± 0.1818 | 0.9814 ± 0.0063 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id9` | 0.6639 ± 0.0889 | 0.8912 ± 0.0277 | 0.6323 ± 0.0820 | 0.9096 ± 0.0185 | 0.2819 ± 0.1664 | 0.9828 ± 0.0057 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id10` | 0.6587 ± 0.0904 | 0.8886 ± 0.0304 | 0.6290 ± 0.0822 | 0.9067 ± 0.0220 | 0.2779 ± 0.1641 | 0.9802 ± 0.0077 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id9` | 0.6437 ± 0.1011 | 0.8845 ± 0.0339 | 0.6183 ± 0.0867 | 0.9048 ± 0.0226 | 0.2557 ± 0.1748 | 0.9808 ± 0.0078 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id9` | 0.6523 ± 0.0892 | 0.8871 ± 0.0298 | 0.6233 ± 0.0827 | 0.9060 ± 0.0214 | 0.2659 ± 0.1658 | 0.9807 ± 0.0071 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+video_4` | 0.6392 ± 0.0740 | 0.8817 ± 0.0284 | 0.6122 ± 0.0611 | 0.9004 ± 0.0193 | 0.2477 ± 0.1215 | 0.9767 ± 0.0034 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6404 ± 0.1072 | 0.8839 ± 0.0360 | 0.6131 ± 0.0872 | 0.9049 ± 0.0251 | 0.2439 ± 0.1710 | 0.9823 ± 0.0077 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6188 ± 0.0566 | 0.8776 ± 0.0227 | 0.5928 ± 0.0438 | 0.8998 ± 0.0143 | 0.2048 ± 0.0897 | 0.9808 ± 0.0035 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6520 ± 0.1085 | 0.8867 ± 0.0358 | 0.6268 ± 0.0929 | 0.9060 ± 0.0236 | 0.2733 ± 0.1874 | 0.9803 ± 0.0076 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6410 ± 0.0969 | 0.8839 ± 0.0323 | 0.6149 ± 0.0842 | 0.9042 ± 0.0218 | 0.2489 ± 0.1684 | 0.9809 ± 0.0048 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6434 ± 0.1081 | 0.8845 ± 0.0364 | 0.6158 ± 0.0877 | 0.9049 ± 0.0255 | 0.2500 ± 0.1715 | 0.9816 ± 0.0075 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6214 ± 0.0577 | 0.8790 ± 0.0230 | 0.5939 ± 0.0439 | 0.9017 ± 0.0147 | 0.2048 ± 0.0897 | 0.9829 ± 0.0054 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6457 ± 0.0916 | 0.8851 ± 0.0313 | 0.6180 ± 0.0808 | 0.9048 ± 0.0214 | 0.2551 ± 0.1616 | 0.9809 ± 0.0048 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6481 ± 0.1090 | 0.8851 ± 0.0361 | 0.6232 ± 0.0914 | 0.9042 ± 0.0244 | 0.2675 ± 0.1823 | 0.9789 ± 0.0089 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6233 ± 0.0639 | 0.8782 ± 0.0248 | 0.5980 ± 0.0515 | 0.8992 ± 0.0161 | 0.2172 ± 0.1050 | 0.9788 ± 0.0054 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6360 ± 0.0710 | 0.8815 ± 0.0262 | 0.6093 ± 0.0602 | 0.9010 ± 0.0177 | 0.2405 ± 0.1224 | 0.9781 ± 0.0060 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6152 ± 0.0657 | 0.8760 ± 0.0242 | 0.5920 ± 0.0512 | 0.8980 ± 0.0152 | 0.2050 ± 0.1057 | 0.9789 ± 0.0071 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6265 ± 0.0743 | 0.8785 ± 0.0261 | 0.6023 ± 0.0614 | 0.8984 ± 0.0169 | 0.2278 ± 0.1235 | 0.9767 ± 0.0037 |
| `carewsm_bimodal_gated_late` | `audio_id1+video_4` | 0.6401 ± 0.0872 | 0.8841 ± 0.0291 | 0.6110 ± 0.0675 | 0.9047 ± 0.0202 | 0.2397 ± 0.1354 | 0.9823 ± 0.0087 |
| `carewsm_bimodal_gated_late` | `audio_id4+video_4` | 0.5972 ± 0.0394 | 0.8721 ± 0.0187 | 0.5743 ± 0.0277 | 0.8968 ± 0.0147 | 0.1671 ± 0.0549 | 0.9815 ± 0.0069 |
| `carewsm_bimodal_gated_late` | `audio_id2+video_4` | 0.5874 ± 0.0502 | 0.8666 ± 0.0189 | 0.5705 ± 0.0401 | 0.8888 ± 0.0140 | 0.1684 ± 0.0814 | 0.9726 ± 0.0112 |
| `carewsm_bimodal_gated_late` | `video_4+text_id10` | 0.5856 ± 0.0674 | 0.8538 ± 0.0303 | 0.5785 ± 0.0623 | 0.8631 ± 0.0269 | 0.2187 ± 0.1159 | 0.9383 ± 0.0147 |
| `carewsm_bimodal_task_aware` | `video_4+text_id9` | 0.6077 ± 0.0964 | 0.8664 ± 0.0391 | 0.5953 ± 0.0884 | 0.8813 ± 0.0328 | 0.2333 ± 0.1717 | 0.9573 ± 0.0179 |
| `carewsm_bimodal_task_aware` | `video_4+text_id10` | 0.5962 ± 0.0921 | 0.8611 ± 0.0381 | 0.5861 ± 0.0835 | 0.8747 ± 0.0322 | 0.2209 ± 0.1589 | 0.9512 ± 0.0160 |
| `carewsm_bimodal_gated_late` | `video_4+text_id9` | 0.5882 ± 0.0751 | 0.8556 ± 0.0330 | 0.5808 ± 0.0700 | 0.8661 ± 0.0287 | 0.2198 ± 0.1323 | 0.9417 ± 0.0150 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id9` | 0.5848 ± 0.0831 | 0.8560 ± 0.0341 | 0.5799 ± 0.0840 | 0.8684 ± 0.0299 | 0.2149 ± 0.1688 | 0.9449 ± 0.0209 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id10` | 0.5852 ± 0.0804 | 0.8557 ± 0.0351 | 0.5794 ± 0.0821 | 0.8678 ± 0.0330 | 0.2147 ± 0.1645 | 0.9442 ± 0.0257 |

### Wsm Depression — VAL

| Model | Feature versions | Macro F1 | Weighted F1 | Macro Recall / UAR | Weighted Recall | Patient Recall | Healthy Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id10` | 0.7404 ± 0.0324 | 0.7496 ± 0.0317 | 0.7406 ± 0.0306 | 0.7502 ± 0.0325 | 0.6907 ± 0.0592 | 0.7905 ± 0.0615 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id10` | 0.7419 ± 0.0389 | 0.7507 ± 0.0374 | 0.7436 ± 0.0383 | 0.7511 ± 0.0371 | 0.7043 ± 0.0829 | 0.7830 ± 0.0659 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id9` | 0.7446 ± 0.0374 | 0.7536 ± 0.0366 | 0.7453 ± 0.0358 | 0.7545 ± 0.0369 | 0.6973 ± 0.0786 | 0.7934 ± 0.0725 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id10` | 0.7425 ± 0.0435 | 0.7517 ± 0.0416 | 0.7432 ± 0.0429 | 0.7525 ± 0.0417 | 0.6956 ± 0.0848 | 0.7909 ± 0.0668 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id9` | 0.7403 ± 0.0400 | 0.7498 ± 0.0386 | 0.7403 ± 0.0389 | 0.7506 ± 0.0389 | 0.6865 ± 0.0729 | 0.7941 ± 0.0635 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id10` | 0.7419 ± 0.0420 | 0.7506 ± 0.0400 | 0.7428 ± 0.0425 | 0.7504 ± 0.0399 | 0.7036 ± 0.0611 | 0.7820 ± 0.0342 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id9` | 0.7396 ± 0.0378 | 0.7485 ± 0.0370 | 0.7406 ± 0.0363 | 0.7487 ± 0.0378 | 0.6982 ± 0.0645 | 0.7828 ± 0.0642 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id9` | 0.7411 ± 0.0292 | 0.7511 ± 0.0282 | 0.7401 ± 0.0280 | 0.7523 ± 0.0289 | 0.6760 ± 0.0578 | 0.8042 ± 0.0546 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id9` | 0.7467 ± 0.0460 | 0.7557 ± 0.0439 | 0.7472 ± 0.0461 | 0.7563 ± 0.0439 | 0.7007 ± 0.0823 | 0.7937 ± 0.0582 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id10` | 0.7447 ± 0.0461 | 0.7537 ± 0.0440 | 0.7455 ± 0.0464 | 0.7541 ± 0.0440 | 0.7013 ± 0.0826 | 0.7897 ± 0.0576 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id10` | 0.7402 ± 0.0385 | 0.7498 ± 0.0368 | 0.7402 ± 0.0389 | 0.7510 ± 0.0365 | 0.6863 ± 0.0877 | 0.7942 ± 0.0643 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id10` | 0.7378 ± 0.0309 | 0.7477 ± 0.0277 | 0.7384 ± 0.0340 | 0.7491 ± 0.0252 | 0.6816 ± 0.0963 | 0.7952 ± 0.0459 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id9` | 0.7449 ± 0.0320 | 0.7543 ± 0.0294 | 0.7453 ± 0.0335 | 0.7555 ± 0.0273 | 0.6911 ± 0.0869 | 0.7995 ± 0.0466 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id9` | 0.7579 ± 0.0269 | 0.7672 ± 0.0248 | 0.7574 ± 0.0283 | 0.7682 ± 0.0238 | 0.7008 ± 0.0713 | 0.8140 ± 0.0407 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id10` | 0.7356 ± 0.0401 | 0.7450 ± 0.0376 | 0.7368 ± 0.0416 | 0.7457 ± 0.0363 | 0.6894 ± 0.0912 | 0.7842 ± 0.0510 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id9` | 0.7380 ± 0.0368 | 0.7474 ± 0.0351 | 0.7386 ± 0.0375 | 0.7483 ± 0.0350 | 0.6895 ± 0.0859 | 0.7877 ± 0.0618 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id10` | 0.7459 ± 0.0317 | 0.7553 ± 0.0299 | 0.7459 ± 0.0329 | 0.7563 ± 0.0292 | 0.6916 ± 0.0761 | 0.8002 ± 0.0473 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id9` | 0.7401 ± 0.0318 | 0.7491 ± 0.0304 | 0.7412 ± 0.0326 | 0.7496 ± 0.0299 | 0.6971 ± 0.0764 | 0.7854 ± 0.0523 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+video_4` | 0.7015 ± 0.0566 | 0.7134 ± 0.0524 | 0.7009 ± 0.0569 | 0.7153 ± 0.0507 | 0.6245 ± 0.0959 | 0.7774 ± 0.0455 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+video_4` | 0.7011 ± 0.0699 | 0.7137 ± 0.0658 | 0.6992 ± 0.0693 | 0.7159 ± 0.0646 | 0.6111 ± 0.0920 | 0.7872 ± 0.0480 |
| `carewsm_bimodal_task_aware` | `audio_id1+video_4` | 0.6884 ± 0.0620 | 0.7009 ± 0.0587 | 0.6876 ± 0.0616 | 0.7028 ± 0.0583 | 0.6069 ± 0.0900 | 0.7683 ± 0.0617 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id9` | 0.7022 ± 0.0675 | 0.7145 ± 0.0637 | 0.7011 ± 0.0677 | 0.7169 ± 0.0630 | 0.6180 ± 0.1050 | 0.7842 ± 0.0632 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id10` | 0.7030 ± 0.0663 | 0.7152 ± 0.0636 | 0.7010 ± 0.0655 | 0.7174 ± 0.0641 | 0.6145 ± 0.0800 | 0.7874 ± 0.0675 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id10` | 0.6965 ± 0.0721 | 0.7082 ± 0.0694 | 0.6960 ± 0.0713 | 0.7099 ± 0.0702 | 0.6222 ± 0.0974 | 0.7697 ± 0.0853 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id10` | 0.6949 ± 0.0753 | 0.7065 ± 0.0718 | 0.6943 ± 0.0750 | 0.7077 ± 0.0718 | 0.6238 ± 0.0990 | 0.7649 ± 0.0687 |
| `carewsm_bimodal_task_aware` | `audio_id4+video_4` | 0.7033 ± 0.0632 | 0.7162 ± 0.0599 | 0.7010 ± 0.0626 | 0.7191 ± 0.0597 | 0.6062 ± 0.0847 | 0.7959 ± 0.0591 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id10` | 0.6813 ± 0.0524 | 0.6933 ± 0.0483 | 0.6820 ± 0.0538 | 0.6951 ± 0.0470 | 0.6135 ± 0.1089 | 0.7505 ± 0.0557 |
| `carewsm_bimodal_task_aware` | `audio_id2+video_4` | 0.6889 ± 0.0747 | 0.7019 ± 0.0713 | 0.6874 ± 0.0738 | 0.7047 ± 0.0712 | 0.5957 ± 0.0981 | 0.7790 ± 0.0745 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id9` | 0.6942 ± 0.0525 | 0.7067 ± 0.0492 | 0.6931 ± 0.0531 | 0.7087 ± 0.0486 | 0.6115 ± 0.0861 | 0.7746 ± 0.0460 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id9` | 0.7025 ± 0.0704 | 0.7145 ± 0.0661 | 0.7016 ± 0.0705 | 0.7163 ± 0.0648 | 0.6238 ± 0.1038 | 0.7794 ± 0.0529 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id10` | 0.6897 ± 0.0781 | 0.7018 ± 0.0739 | 0.6905 ± 0.0786 | 0.7041 ± 0.0730 | 0.6183 ± 0.1309 | 0.7626 ± 0.0808 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id9` | 0.6953 ± 0.0805 | 0.7080 ± 0.0764 | 0.6943 ± 0.0800 | 0.7108 ± 0.0756 | 0.6076 ± 0.1172 | 0.7811 ± 0.0762 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id10` | 0.6844 ± 0.0660 | 0.6965 ± 0.0618 | 0.6848 ± 0.0671 | 0.6980 ± 0.0606 | 0.6156 ± 0.1133 | 0.7541 ± 0.0563 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id9` | 0.6844 ± 0.0505 | 0.6967 ± 0.0468 | 0.6845 ± 0.0516 | 0.6986 ± 0.0456 | 0.6104 ± 0.1013 | 0.7585 ± 0.0528 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id9` | 0.6876 ± 0.0703 | 0.6994 ± 0.0658 | 0.6887 ± 0.0715 | 0.7013 ± 0.0645 | 0.6221 ± 0.1275 | 0.7553 ± 0.0686 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+video_4` | 0.6855 ± 0.0699 | 0.6994 ± 0.0652 | 0.6835 ± 0.0690 | 0.7028 ± 0.0633 | 0.5812 ± 0.0977 | 0.7858 ± 0.0445 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6848 ± 0.0680 | 0.6975 ± 0.0641 | 0.6848 ± 0.0682 | 0.7000 ± 0.0632 | 0.6046 ± 0.1147 | 0.7650 ± 0.0704 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.7007 ± 0.0577 | 0.7120 ± 0.0548 | 0.7009 ± 0.0582 | 0.7134 ± 0.0549 | 0.6352 ± 0.0995 | 0.7666 ± 0.0687 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6823 ± 0.0734 | 0.6954 ± 0.0694 | 0.6820 ± 0.0732 | 0.6983 ± 0.0687 | 0.5953 ± 0.1154 | 0.7686 ± 0.0746 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6947 ± 0.0713 | 0.7078 ± 0.0669 | 0.6934 ± 0.0707 | 0.7111 ± 0.0654 | 0.5998 ± 0.1102 | 0.7870 ± 0.0643 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6883 ± 0.0650 | 0.7015 ± 0.0611 | 0.6872 ± 0.0644 | 0.7047 ± 0.0603 | 0.5946 ± 0.1028 | 0.7799 ± 0.0675 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6990 ± 0.0562 | 0.7109 ± 0.0533 | 0.6982 ± 0.0564 | 0.7127 ± 0.0533 | 0.6218 ± 0.0897 | 0.7746 ± 0.0617 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6856 ± 0.0763 | 0.6988 ± 0.0724 | 0.6848 ± 0.0758 | 0.7020 ± 0.0719 | 0.5938 ± 0.1157 | 0.7758 ± 0.0793 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6861 ± 0.0648 | 0.6994 ± 0.0602 | 0.6855 ± 0.0651 | 0.7028 ± 0.0584 | 0.5949 ± 0.1162 | 0.7763 ± 0.0597 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6874 ± 0.0594 | 0.6996 ± 0.0556 | 0.6877 ± 0.0605 | 0.7014 ± 0.0550 | 0.6156 ± 0.1100 | 0.7598 ± 0.0629 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6925 ± 0.0749 | 0.7049 ± 0.0718 | 0.6914 ± 0.0746 | 0.7072 ± 0.0723 | 0.6079 ± 0.1024 | 0.7750 ± 0.0810 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6887 ± 0.0551 | 0.7010 ± 0.0521 | 0.6884 ± 0.0555 | 0.7029 ± 0.0523 | 0.6123 ± 0.0959 | 0.7646 ± 0.0668 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6970 ± 0.0679 | 0.7097 ± 0.0642 | 0.6955 ± 0.0676 | 0.7124 ± 0.0640 | 0.6055 ± 0.0998 | 0.7855 ± 0.0685 |
| `carewsm_bimodal_gated_late` | `audio_id1+video_4` | 0.6671 ± 0.0740 | 0.6825 ± 0.0709 | 0.6651 ± 0.0713 | 0.6880 ± 0.0718 | 0.5430 ± 0.0903 | 0.7872 ± 0.0953 |
| `carewsm_bimodal_gated_late` | `audio_id4+video_4` | 0.6615 ± 0.0744 | 0.6763 ± 0.0715 | 0.6597 ± 0.0721 | 0.6803 ± 0.0727 | 0.5497 ± 0.0856 | 0.7698 ± 0.0918 |
| `carewsm_bimodal_gated_late` | `audio_id2+video_4` | 0.6766 ± 0.0742 | 0.6920 ± 0.0706 | 0.6734 ± 0.0725 | 0.6973 ± 0.0707 | 0.5469 ± 0.0811 | 0.7998 ± 0.0704 |
| `carewsm_bimodal_gated_late` | `video_4+text_id10` | 0.6471 ± 0.0776 | 0.6625 ± 0.0729 | 0.6459 ± 0.0768 | 0.6661 ± 0.0714 | 0.5407 ± 0.1100 | 0.7512 ± 0.0576 |
| `carewsm_bimodal_task_aware` | `video_4+text_id9` | 0.6386 ± 0.0714 | 0.6536 ± 0.0672 | 0.6381 ± 0.0706 | 0.6565 ± 0.0664 | 0.5401 ± 0.1017 | 0.7360 ± 0.0646 |
| `carewsm_bimodal_task_aware` | `video_4+text_id10` | 0.6377 ± 0.0731 | 0.6521 ± 0.0702 | 0.6369 ± 0.0725 | 0.6542 ± 0.0708 | 0.5455 ± 0.0884 | 0.7283 ± 0.0726 |
| `carewsm_bimodal_gated_late` | `video_4+text_id9` | 0.6431 ± 0.0788 | 0.6592 ± 0.0738 | 0.6421 ± 0.0780 | 0.6639 ± 0.0721 | 0.5285 ± 0.1186 | 0.7556 ± 0.0620 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id9` | 0.6414 ± 0.0786 | 0.6564 ± 0.0748 | 0.6406 ± 0.0778 | 0.6594 ± 0.0746 | 0.5416 ± 0.1049 | 0.7396 ± 0.0732 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id10` | 0.6393 ± 0.0718 | 0.6544 ± 0.0678 | 0.6388 ± 0.0710 | 0.6578 ± 0.0671 | 0.5396 ± 0.1068 | 0.7381 ± 0.0686 |

### Wsm Parkinson — VAL

| Model | Feature versions | Macro F1 | Weighted F1 | Macro Recall / UAR | Weighted Recall | Patient Recall | Healthy Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id10` | 0.9033 ± 0.0086 | 0.9062 ± 0.0084 | 0.9040 ± 0.0102 | 0.9062 ± 0.0085 | 0.8926 ± 0.0418 | 0.9154 ± 0.0298 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id10` | 0.8982 ± 0.0166 | 0.9013 ± 0.0157 | 0.8992 ± 0.0165 | 0.9012 ± 0.0159 | 0.8880 ± 0.0443 | 0.9103 ± 0.0359 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id9` | 0.9028 ± 0.0150 | 0.9059 ± 0.0141 | 0.9025 ± 0.0161 | 0.9061 ± 0.0141 | 0.8826 ± 0.0445 | 0.9223 ± 0.0301 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id10` | 0.8928 ± 0.0217 | 0.8962 ± 0.0206 | 0.8927 ± 0.0220 | 0.8964 ± 0.0206 | 0.8725 ± 0.0540 | 0.9128 ± 0.0410 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id9` | 0.8952 ± 0.0196 | 0.8984 ± 0.0187 | 0.8954 ± 0.0194 | 0.8986 ± 0.0188 | 0.8780 ± 0.0483 | 0.9128 ± 0.0397 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id10` | 0.8993 ± 0.0150 | 0.9024 ± 0.0137 | 0.8997 ± 0.0176 | 0.9024 ± 0.0134 | 0.8846 ± 0.0424 | 0.9148 ± 0.0139 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id9` | 0.9002 ± 0.0160 | 0.9034 ± 0.0148 | 0.8996 ± 0.0176 | 0.9036 ± 0.0146 | 0.8776 ± 0.0383 | 0.9217 ± 0.0176 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id9` | 0.8965 ± 0.0113 | 0.8996 ± 0.0114 | 0.8968 ± 0.0084 | 0.8996 ± 0.0118 | 0.8821 ± 0.0363 | 0.9116 ± 0.0396 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id9` | 0.9011 ± 0.0169 | 0.9043 ± 0.0156 | 0.9004 ± 0.0186 | 0.9045 ± 0.0155 | 0.8772 ± 0.0493 | 0.9235 ± 0.0292 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id10` | 0.9015 ± 0.0183 | 0.9047 ± 0.0171 | 0.9008 ± 0.0196 | 0.9049 ± 0.0171 | 0.8782 ± 0.0484 | 0.9235 ± 0.0310 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id10` | 0.9038 ± 0.0118 | 0.9066 ± 0.0111 | 0.9052 ± 0.0123 | 0.9065 ± 0.0111 | 0.8981 ± 0.0304 | 0.9123 ± 0.0206 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id10` | 0.8904 ± 0.0223 | 0.8934 ± 0.0211 | 0.8937 ± 0.0223 | 0.8932 ± 0.0213 | 0.8978 ± 0.0496 | 0.8897 ± 0.0382 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id9` | 0.8869 ± 0.0202 | 0.8900 ± 0.0194 | 0.8898 ± 0.0195 | 0.8898 ± 0.0195 | 0.8906 ± 0.0553 | 0.8890 ± 0.0477 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id9` | 0.8932 ± 0.0195 | 0.8964 ± 0.0184 | 0.8941 ± 0.0195 | 0.8964 ± 0.0185 | 0.8817 ± 0.0442 | 0.9066 ± 0.0338 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id10` | 0.8920 ± 0.0222 | 0.8949 ± 0.0211 | 0.8954 ± 0.0224 | 0.8947 ± 0.0212 | 0.9004 ± 0.0497 | 0.8904 ± 0.0372 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id9` | 0.8988 ± 0.0119 | 0.9018 ± 0.0113 | 0.9004 ± 0.0113 | 0.9017 ± 0.0114 | 0.8929 ± 0.0357 | 0.9079 ± 0.0308 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id10` | 0.8979 ± 0.0121 | 0.9009 ± 0.0113 | 0.8993 ± 0.0130 | 0.9009 ± 0.0113 | 0.8907 ± 0.0468 | 0.9078 ± 0.0346 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id9` | 0.8950 ± 0.0218 | 0.8979 ± 0.0205 | 0.8982 ± 0.0230 | 0.8976 ± 0.0204 | 0.9022 ± 0.0449 | 0.8941 ± 0.0234 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+video_4` | 0.8115 ± 0.0470 | 0.8200 ± 0.0427 | 0.8060 ± 0.0479 | 0.8248 ± 0.0367 | 0.6985 ± 0.1141 | 0.9135 ± 0.0270 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+video_4` | 0.7746 ± 0.0420 | 0.7848 ± 0.0388 | 0.7699 ± 0.0418 | 0.7908 ± 0.0338 | 0.6492 ± 0.1076 | 0.8907 ± 0.0427 |
| `carewsm_bimodal_task_aware` | `audio_id1+video_4` | 0.7912 ± 0.0716 | 0.8014 ± 0.0660 | 0.7869 ± 0.0711 | 0.8091 ± 0.0576 | 0.6566 ± 0.1563 | 0.9172 ± 0.0275 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id9` | 0.7767 ± 0.0513 | 0.7873 ± 0.0476 | 0.7724 ± 0.0507 | 0.7954 ± 0.0401 | 0.6361 ± 0.1368 | 0.9088 ± 0.0504 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id10` | 0.7963 ± 0.0638 | 0.8058 ± 0.0599 | 0.7914 ± 0.0646 | 0.8121 ± 0.0542 | 0.6694 ± 0.1395 | 0.9134 ± 0.0360 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id10` | 0.7740 ± 0.0555 | 0.7847 ± 0.0517 | 0.7696 ± 0.0548 | 0.7921 ± 0.0451 | 0.6366 ± 0.1302 | 0.9026 ± 0.0367 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id10` | 0.7884 ± 0.0509 | 0.7980 ± 0.0482 | 0.7828 ± 0.0502 | 0.8035 ± 0.0439 | 0.6602 ± 0.1015 | 0.9054 ± 0.0278 |
| `carewsm_bimodal_task_aware` | `audio_id4+video_4` | 0.8000 ± 0.0694 | 0.8093 ± 0.0646 | 0.7956 ± 0.0698 | 0.8157 ± 0.0583 | 0.6758 ± 0.1483 | 0.9153 ± 0.0365 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id10` | 0.7795 ± 0.0695 | 0.7899 ± 0.0647 | 0.7754 ± 0.0686 | 0.7970 ± 0.0579 | 0.6487 ± 0.1440 | 0.9021 ± 0.0362 |
| `carewsm_bimodal_task_aware` | `audio_id2+video_4` | 0.7698 ± 0.0608 | 0.7808 ± 0.0566 | 0.7655 ± 0.0586 | 0.7888 ± 0.0494 | 0.6278 ± 0.1325 | 0.9033 ± 0.0425 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id9` | 0.7891 ± 0.0547 | 0.7989 ± 0.0507 | 0.7842 ± 0.0553 | 0.8053 ± 0.0447 | 0.6595 ± 0.1270 | 0.9090 ± 0.0279 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id9` | 0.7887 ± 0.0518 | 0.7984 ± 0.0484 | 0.7833 ± 0.0514 | 0.8041 ± 0.0435 | 0.6599 ± 0.1091 | 0.9066 ± 0.0304 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id10` | 0.7680 ± 0.0535 | 0.7783 ± 0.0491 | 0.7651 ± 0.0535 | 0.7847 ± 0.0417 | 0.6495 ± 0.1353 | 0.8807 ± 0.0401 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id9` | 0.7655 ± 0.0528 | 0.7761 ± 0.0486 | 0.7621 ± 0.0521 | 0.7828 ± 0.0414 | 0.6404 ± 0.1288 | 0.8838 ± 0.0391 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id10` | 0.7831 ± 0.0515 | 0.7933 ± 0.0478 | 0.7780 ± 0.0509 | 0.8001 ± 0.0417 | 0.6476 ± 0.1177 | 0.9084 ± 0.0345 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id9` | 0.7718 ± 0.0663 | 0.7828 ± 0.0612 | 0.7681 ± 0.0657 | 0.7910 ± 0.0531 | 0.6330 ± 0.1508 | 0.9033 ± 0.0352 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id9` | 0.7806 ± 0.0509 | 0.7910 ± 0.0471 | 0.7753 ± 0.0496 | 0.7978 ± 0.0412 | 0.6434 ± 0.1129 | 0.9072 ± 0.0388 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+video_4` | 0.8001 ± 0.0365 | 0.8086 ± 0.0328 | 0.7960 ± 0.0396 | 0.8126 ± 0.0279 | 0.6987 ± 0.1080 | 0.8932 ± 0.0299 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.7815 ± 0.0615 | 0.7919 ± 0.0574 | 0.7764 ± 0.0607 | 0.7987 ± 0.0513 | 0.6455 ± 0.1252 | 0.9072 ± 0.0275 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.7791 ± 0.0660 | 0.7895 ± 0.0620 | 0.7741 ± 0.0651 | 0.7963 ± 0.0561 | 0.6435 ± 0.1276 | 0.9048 ± 0.0192 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.7701 ± 0.0689 | 0.7814 ± 0.0636 | 0.7663 ± 0.0670 | 0.7899 ± 0.0552 | 0.6266 ± 0.1467 | 0.9059 ± 0.0313 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.7476 ± 0.0733 | 0.7602 ± 0.0684 | 0.7444 ± 0.0699 | 0.7703 ± 0.0598 | 0.5922 ± 0.1463 | 0.8965 ± 0.0338 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.7666 ± 0.0710 | 0.7780 ± 0.0662 | 0.7626 ± 0.0687 | 0.7864 ± 0.0584 | 0.6230 ± 0.1427 | 0.9022 ± 0.0333 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.7745 ± 0.0684 | 0.7851 ± 0.0640 | 0.7699 ± 0.0672 | 0.7923 ± 0.0574 | 0.6383 ± 0.1342 | 0.9016 ± 0.0156 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.7387 ± 0.0738 | 0.7517 ± 0.0686 | 0.7362 ± 0.0702 | 0.7622 ± 0.0593 | 0.5833 ± 0.1487 | 0.8890 ± 0.0313 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.7592 ± 0.0748 | 0.7711 ± 0.0695 | 0.7559 ± 0.0720 | 0.7802 ± 0.0607 | 0.6134 ± 0.1502 | 0.8985 ± 0.0317 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.7686 ± 0.0660 | 0.7792 ± 0.0627 | 0.7638 ± 0.0644 | 0.7858 ± 0.0576 | 0.6347 ± 0.1185 | 0.8929 ± 0.0289 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.7486 ± 0.0750 | 0.7613 ± 0.0695 | 0.7459 ± 0.0716 | 0.7718 ± 0.0598 | 0.5934 ± 0.1556 | 0.8984 ± 0.0288 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.7699 ± 0.0595 | 0.7804 ± 0.0559 | 0.7657 ± 0.0589 | 0.7867 ± 0.0502 | 0.6417 ± 0.1216 | 0.8896 ± 0.0210 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.7472 ± 0.0726 | 0.7594 ± 0.0680 | 0.7444 ± 0.0702 | 0.7686 ± 0.0597 | 0.6024 ± 0.1495 | 0.8865 ± 0.0308 |
| `carewsm_bimodal_gated_late` | `audio_id1+video_4` | 0.8021 ± 0.0838 | 0.8122 ± 0.0784 | 0.7966 ± 0.0827 | 0.8208 ± 0.0704 | 0.6547 ± 0.1678 | 0.9386 ± 0.0308 |
| `carewsm_bimodal_gated_late` | `audio_id4+video_4` | 0.7995 ± 0.0529 | 0.8091 ± 0.0487 | 0.7940 ± 0.0555 | 0.8155 ± 0.0429 | 0.6684 ± 0.1317 | 0.9196 ± 0.0215 |
| `carewsm_bimodal_gated_late` | `audio_id2+video_4` | 0.7507 ± 0.0705 | 0.7644 ± 0.0651 | 0.7467 ± 0.0654 | 0.7772 ± 0.0545 | 0.5681 ± 0.1420 | 0.9253 ± 0.0179 |
| `carewsm_bimodal_gated_late` | `video_4+text_id10` | 0.6459 ± 0.0556 | 0.6638 ± 0.0495 | 0.6495 ± 0.0481 | 0.6788 ± 0.0365 | 0.4785 ± 0.1434 | 0.8204 ± 0.0601 |
| `carewsm_bimodal_task_aware` | `video_4+text_id9` | 0.6363 ± 0.0489 | 0.6532 ± 0.0438 | 0.6394 ± 0.0435 | 0.6655 ± 0.0336 | 0.4867 ± 0.1368 | 0.7921 ± 0.0660 |
| `carewsm_bimodal_task_aware` | `video_4+text_id10` | 0.6378 ± 0.0503 | 0.6543 ± 0.0448 | 0.6419 ± 0.0461 | 0.6666 ± 0.0346 | 0.4968 ± 0.1546 | 0.7870 ± 0.0801 |
| `carewsm_bimodal_gated_late` | `video_4+text_id9` | 0.6476 ± 0.0559 | 0.6655 ± 0.0503 | 0.6507 ± 0.0481 | 0.6807 ± 0.0382 | 0.4766 ± 0.1377 | 0.8248 ± 0.0599 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id9` | 0.6398 ± 0.0469 | 0.6558 ± 0.0426 | 0.6424 ± 0.0433 | 0.6663 ± 0.0358 | 0.5039 ± 0.1349 | 0.7809 ± 0.0773 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id10` | 0.6384 ± 0.0448 | 0.6541 ± 0.0396 | 0.6420 ± 0.0419 | 0.6648 ± 0.0321 | 0.5094 ± 0.1473 | 0.7746 ± 0.0850 |


## Detailed TEST metrics — five-fold mean ± sample SD

### Care Depression — TEST

| Model | Feature versions | Macro F1 | Weighted F1 | Macro Recall / UAR | Weighted Recall | Patient Recall | Healthy Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id10` | 0.8025 ± 0.0209 | 0.8379 ± 0.0163 | 0.7810 ± 0.0220 | 0.8461 ± 0.0142 | 0.6106 ± 0.0430 | 0.9514 ± 0.0052 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id10` | 0.8280 ± 0.0130 | 0.8579 ± 0.0093 | 0.8080 ± 0.0183 | 0.8638 ± 0.0068 | 0.6617 ± 0.0490 | 0.9543 ± 0.0129 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id9` | 0.8232 ± 0.0300 | 0.8553 ± 0.0238 | 0.7987 ± 0.0294 | 0.8632 ± 0.0213 | 0.6298 ± 0.0518 | 0.9676 ± 0.0109 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id10` | 0.8143 ± 0.0285 | 0.8483 ± 0.0225 | 0.7891 ± 0.0280 | 0.8572 ± 0.0200 | 0.6106 ± 0.0497 | 0.9676 ± 0.0085 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id9` | 0.8120 ± 0.0288 | 0.8467 ± 0.0230 | 0.7864 ± 0.0278 | 0.8559 ± 0.0209 | 0.6043 ± 0.0466 | 0.9686 ± 0.0110 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id10` | 0.7951 ± 0.0182 | 0.8330 ± 0.0140 | 0.7709 ± 0.0204 | 0.8434 ± 0.0124 | 0.5809 ± 0.0530 | 0.9610 ± 0.0222 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id9` | 0.8184 ± 0.0402 | 0.8515 ± 0.0313 | 0.7951 ± 0.0427 | 0.8599 ± 0.0271 | 0.6255 ± 0.0849 | 0.9648 ± 0.0099 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id9` | 0.8078 ± 0.0239 | 0.8427 ± 0.0191 | 0.7843 ± 0.0231 | 0.8513 ± 0.0175 | 0.6085 ± 0.0394 | 0.9600 ± 0.0110 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id9` | 0.8057 ± 0.0313 | 0.8413 ± 0.0235 | 0.7826 ± 0.0337 | 0.8507 ± 0.0190 | 0.6043 ± 0.0742 | 0.9610 ± 0.0123 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id10` | 0.8098 ± 0.0358 | 0.8445 ± 0.0275 | 0.7869 ± 0.0366 | 0.8533 ± 0.0231 | 0.6128 ± 0.0739 | 0.9610 ± 0.0109 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id10` | 0.8021 ± 0.0462 | 0.8384 ± 0.0353 | 0.7801 ± 0.0468 | 0.8480 ± 0.0295 | 0.6021 ± 0.0974 | 0.9581 ± 0.0209 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id10` | 0.8025 ± 0.0556 | 0.8379 ± 0.0452 | 0.7816 ± 0.0541 | 0.8461 ± 0.0425 | 0.6128 ± 0.0876 | 0.9505 ± 0.0281 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id9` | 0.8032 ± 0.0613 | 0.8390 ± 0.0495 | 0.7813 ± 0.0603 | 0.8480 ± 0.0460 | 0.6064 ± 0.1029 | 0.9562 ± 0.0317 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id9` | 0.7776 ± 0.0522 | 0.8199 ± 0.0415 | 0.7533 ± 0.0490 | 0.8329 ± 0.0373 | 0.5447 ± 0.0815 | 0.9619 ± 0.0221 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id10` | 0.8061 ± 0.0403 | 0.8396 ± 0.0334 | 0.7875 ± 0.0376 | 0.8461 ± 0.0323 | 0.6340 ± 0.0561 | 0.9410 ± 0.0281 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id9` | 0.8015 ± 0.0487 | 0.8376 ± 0.0371 | 0.7809 ± 0.0486 | 0.8467 ± 0.0306 | 0.6085 ± 0.0973 | 0.9533 ± 0.0119 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id10` | 0.7866 ± 0.0497 | 0.8258 ± 0.0400 | 0.7639 ± 0.0456 | 0.8362 ± 0.0369 | 0.5745 ± 0.0710 | 0.9533 ± 0.0257 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id9` | 0.8074 ± 0.0545 | 0.8408 ± 0.0453 | 0.7884 ± 0.0512 | 0.8474 ± 0.0440 | 0.6340 ± 0.0769 | 0.9429 ± 0.0388 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+video_4` | 0.6909 ± 0.0205 | 0.7492 ± 0.0170 | 0.6764 ± 0.0185 | 0.7664 ± 0.0169 | 0.4404 ± 0.0324 | 0.9124 ± 0.0212 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+video_4` | 0.7030 ± 0.0333 | 0.7584 ± 0.0262 | 0.6880 ± 0.0302 | 0.7743 ± 0.0239 | 0.4617 ± 0.0576 | 0.9143 ± 0.0263 |
| `carewsm_bimodal_task_aware` | `audio_id1+video_4` | 0.7177 ± 0.0267 | 0.7691 ± 0.0203 | 0.7031 ± 0.0264 | 0.7822 ± 0.0181 | 0.4957 ± 0.0658 | 0.9105 ± 0.0304 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id9` | 0.7022 ± 0.0323 | 0.7573 ± 0.0256 | 0.6877 ± 0.0295 | 0.7724 ± 0.0237 | 0.4660 ± 0.0560 | 0.9095 ± 0.0267 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id10` | 0.6831 ± 0.0139 | 0.7450 ± 0.0080 | 0.6688 ± 0.0148 | 0.7664 ± 0.0040 | 0.4128 ± 0.0512 | 0.9248 ± 0.0229 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id10` | 0.6992 ± 0.0337 | 0.7547 ± 0.0272 | 0.6852 ± 0.0303 | 0.7697 ± 0.0260 | 0.4638 ± 0.0595 | 0.9067 ± 0.0343 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id10` | 0.7006 ± 0.0292 | 0.7567 ± 0.0240 | 0.6853 ± 0.0260 | 0.7730 ± 0.0235 | 0.4553 ± 0.0460 | 0.9152 ± 0.0290 |
| `carewsm_bimodal_task_aware` | `audio_id4+video_4` | 0.7108 ± 0.0258 | 0.7643 ± 0.0191 | 0.6960 ± 0.0260 | 0.7789 ± 0.0163 | 0.4787 ± 0.0660 | 0.9133 ± 0.0278 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id10` | 0.7021 ± 0.0179 | 0.7567 ± 0.0145 | 0.6880 ± 0.0170 | 0.7711 ± 0.0148 | 0.4702 ± 0.0415 | 0.9057 ± 0.0257 |
| `carewsm_bimodal_task_aware` | `audio_id2+video_4` | 0.7007 ± 0.0371 | 0.7559 ± 0.0281 | 0.6880 ± 0.0360 | 0.7711 ± 0.0250 | 0.4702 ± 0.0885 | 0.9057 ± 0.0410 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id9` | 0.6804 ± 0.0332 | 0.7414 ± 0.0255 | 0.6674 ± 0.0308 | 0.7605 ± 0.0225 | 0.4234 ± 0.0644 | 0.9114 ± 0.0266 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id9` | 0.6920 ± 0.0337 | 0.7505 ± 0.0272 | 0.6772 ± 0.0309 | 0.7684 ± 0.0258 | 0.4383 ± 0.0529 | 0.9162 ± 0.0260 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id10` | 0.7000 ± 0.0350 | 0.7553 ± 0.0268 | 0.6863 ± 0.0332 | 0.7704 ± 0.0230 | 0.4660 ± 0.0679 | 0.9067 ± 0.0220 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id9` | 0.7031 ± 0.0403 | 0.7590 ± 0.0317 | 0.6878 ± 0.0369 | 0.7757 ± 0.0284 | 0.4574 ± 0.0664 | 0.9181 ± 0.0251 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id10` | 0.7007 ± 0.0149 | 0.7563 ± 0.0113 | 0.6861 ± 0.0144 | 0.7717 ± 0.0106 | 0.4617 ± 0.0381 | 0.9105 ± 0.0200 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id9` | 0.7089 ± 0.0188 | 0.7628 ± 0.0134 | 0.6939 ± 0.0186 | 0.7776 ± 0.0113 | 0.4745 ± 0.0530 | 0.9133 ± 0.0248 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id9` | 0.7002 ± 0.0246 | 0.7550 ± 0.0196 | 0.6865 ± 0.0230 | 0.7691 ± 0.0185 | 0.4702 ± 0.0466 | 0.9029 ± 0.0237 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+video_4` | 0.6561 ± 0.0231 | 0.7226 ± 0.0180 | 0.6454 ± 0.0205 | 0.7447 ± 0.0175 | 0.3851 ± 0.0472 | 0.9057 ± 0.0283 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6885 ± 0.0137 | 0.7474 ± 0.0109 | 0.6743 ± 0.0127 | 0.7651 ± 0.0113 | 0.4362 ± 0.0336 | 0.9124 ± 0.0217 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6813 ± 0.0376 | 0.7421 ± 0.0301 | 0.6679 ± 0.0329 | 0.7612 ± 0.0284 | 0.4234 ± 0.0604 | 0.9124 ± 0.0344 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6792 ± 0.0040 | 0.7412 ± 0.0036 | 0.6654 ± 0.0053 | 0.7618 ± 0.0089 | 0.4128 ± 0.0387 | 0.9181 ± 0.0296 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6768 ± 0.0403 | 0.7400 ± 0.0317 | 0.6631 ± 0.0353 | 0.7618 ± 0.0285 | 0.4043 ± 0.0606 | 0.9219 ± 0.0266 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6722 ± 0.0287 | 0.7354 ± 0.0254 | 0.6588 ± 0.0248 | 0.7559 ± 0.0277 | 0.4043 ± 0.0271 | 0.9133 ± 0.0352 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6809 ± 0.0383 | 0.7419 ± 0.0305 | 0.6673 ± 0.0342 | 0.7612 ± 0.0281 | 0.4213 ± 0.0576 | 0.9133 ± 0.0260 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6913 ± 0.0334 | 0.7503 ± 0.0269 | 0.6765 ± 0.0291 | 0.7691 ± 0.0258 | 0.4340 ± 0.0570 | 0.9190 ± 0.0353 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6699 ± 0.0294 | 0.7332 ± 0.0254 | 0.6569 ± 0.0255 | 0.7533 ± 0.0263 | 0.4043 ± 0.0271 | 0.9095 ± 0.0292 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6767 ± 0.0341 | 0.7385 ± 0.0268 | 0.6637 ± 0.0303 | 0.7579 ± 0.0240 | 0.4170 ± 0.0550 | 0.9105 ± 0.0239 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6775 ± 0.0272 | 0.7391 ± 0.0224 | 0.6642 ± 0.0230 | 0.7586 ± 0.0230 | 0.4170 ± 0.0460 | 0.9114 ± 0.0351 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6757 ± 0.0377 | 0.7382 ± 0.0299 | 0.6625 ± 0.0328 | 0.7586 ± 0.0272 | 0.4106 ± 0.0546 | 0.9143 ± 0.0254 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6703 ± 0.0389 | 0.7338 ± 0.0328 | 0.6573 ± 0.0336 | 0.7546 ± 0.0334 | 0.4021 ± 0.0460 | 0.9124 ± 0.0388 |
| `carewsm_bimodal_gated_late` | `audio_id1+video_4` | 0.6458 ± 0.0281 | 0.7147 ± 0.0246 | 0.6359 ± 0.0242 | 0.7382 ± 0.0276 | 0.3681 ± 0.0402 | 0.9038 ± 0.0413 |
| `carewsm_bimodal_gated_late` | `audio_id4+video_4` | 0.6198 ± 0.0262 | 0.6923 ± 0.0207 | 0.6140 ± 0.0229 | 0.7151 ± 0.0211 | 0.3489 ± 0.0529 | 0.8790 ± 0.0353 |
| `carewsm_bimodal_gated_late` | `audio_id2+video_4` | 0.6520 ± 0.0195 | 0.7191 ± 0.0138 | 0.6424 ± 0.0179 | 0.7414 ± 0.0146 | 0.3830 ± 0.0606 | 0.9019 ± 0.0377 |
| `carewsm_bimodal_gated_late` | `video_4+text_id10` | 0.6331 ± 0.0475 | 0.6995 ± 0.0364 | 0.6268 ± 0.0425 | 0.7158 ± 0.0309 | 0.3936 ± 0.0771 | 0.8600 ± 0.0209 |
| `carewsm_bimodal_task_aware` | `video_4+text_id9` | 0.6292 ± 0.0318 | 0.6970 ± 0.0257 | 0.6223 ± 0.0282 | 0.7145 ± 0.0246 | 0.3809 ± 0.0484 | 0.8638 ± 0.0275 |
| `carewsm_bimodal_task_aware` | `video_4+text_id10` | 0.6280 ± 0.0221 | 0.6953 ± 0.0173 | 0.6216 ± 0.0195 | 0.7118 ± 0.0170 | 0.3851 ± 0.0454 | 0.8581 ± 0.0281 |
| `carewsm_bimodal_gated_late` | `video_4+text_id9` | 0.6290 ± 0.0426 | 0.6958 ± 0.0317 | 0.6234 ± 0.0386 | 0.7118 ± 0.0255 | 0.3915 ± 0.0787 | 0.8552 ± 0.0206 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id9` | 0.6283 ± 0.0480 | 0.6975 ± 0.0353 | 0.6225 ± 0.0425 | 0.7171 ± 0.0272 | 0.3745 ± 0.0849 | 0.8705 ± 0.0132 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id10` | 0.6173 ± 0.0467 | 0.6893 ± 0.0348 | 0.6124 ± 0.0409 | 0.7105 ± 0.0277 | 0.3553 ± 0.0798 | 0.8695 ± 0.0183 |

### Care Parkinson — TEST

| Model | Feature versions | Macro F1 | Weighted F1 | Macro Recall / UAR | Weighted Recall | Patient Recall | Healthy Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id10` | 0.8246 ± 0.0468 | 0.9354 ± 0.0170 | 0.7954 ± 0.0518 | 0.9388 ± 0.0159 | 0.6121 ± 0.0991 | 0.9786 ± 0.0084 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id10` | 0.8152 ± 0.0217 | 0.9319 ± 0.0077 | 0.7855 ± 0.0266 | 0.9355 ± 0.0072 | 0.5939 ± 0.0550 | 0.9771 ± 0.0071 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id9` | 0.8325 ± 0.0386 | 0.9386 ± 0.0138 | 0.7999 ± 0.0442 | 0.9421 ± 0.0129 | 0.6182 ± 0.0873 | 0.9815 ± 0.0090 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id10` | 0.8303 ± 0.0312 | 0.9378 ± 0.0109 | 0.7968 ± 0.0356 | 0.9414 ± 0.0097 | 0.6121 ± 0.0691 | 0.9815 ± 0.0037 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id9` | 0.8348 ± 0.0346 | 0.9390 ± 0.0123 | 0.8052 ± 0.0376 | 0.9421 ± 0.0113 | 0.6303 ± 0.0723 | 0.9801 ± 0.0062 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id10` | 0.8043 ± 0.0239 | 0.9294 ± 0.0063 | 0.7665 ± 0.0410 | 0.9349 ± 0.0043 | 0.5515 ± 0.0919 | 0.9815 ± 0.0111 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id9` | 0.8075 ± 0.0338 | 0.9301 ± 0.0101 | 0.7745 ± 0.0486 | 0.9349 ± 0.0075 | 0.5697 ± 0.1037 | 0.9793 ± 0.0089 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id9` | 0.8258 ± 0.0493 | 0.9363 ± 0.0176 | 0.7934 ± 0.0556 | 0.9401 ± 0.0160 | 0.6061 ± 0.1071 | 0.9808 ± 0.0071 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id9` | 0.8190 ± 0.0337 | 0.9346 ± 0.0111 | 0.7798 ± 0.0421 | 0.9395 ± 0.0092 | 0.5758 ± 0.0857 | 0.9838 ± 0.0056 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id10` | 0.8210 ± 0.0350 | 0.9353 ± 0.0110 | 0.7828 ± 0.0457 | 0.9401 ± 0.0085 | 0.5818 ± 0.0944 | 0.9838 ± 0.0056 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id10` | 0.7961 ± 0.0409 | 0.9268 ± 0.0130 | 0.7574 ± 0.0494 | 0.9329 ± 0.0106 | 0.5333 ± 0.1019 | 0.9815 ± 0.0083 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id10` | 0.7847 ± 0.0357 | 0.9227 ± 0.0128 | 0.7446 ± 0.0316 | 0.9289 ± 0.0118 | 0.5091 ± 0.0583 | 0.9801 ± 0.0077 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id9` | 0.7848 ± 0.0526 | 0.9239 ± 0.0153 | 0.7434 ± 0.0598 | 0.9316 ± 0.0100 | 0.5030 ± 0.1242 | 0.9838 ± 0.0062 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id9` | 0.7753 ± 0.0577 | 0.9208 ± 0.0187 | 0.7313 ± 0.0565 | 0.9289 ± 0.0148 | 0.4788 ± 0.1101 | 0.9838 ± 0.0042 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id10` | 0.7804 ± 0.0251 | 0.9218 ± 0.0088 | 0.7366 ± 0.0215 | 0.9289 ± 0.0079 | 0.4909 ± 0.0395 | 0.9823 ± 0.0048 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id9` | 0.7971 ± 0.0467 | 0.9270 ± 0.0151 | 0.7601 ± 0.0552 | 0.9329 ± 0.0124 | 0.5394 ± 0.1122 | 0.9808 ± 0.0080 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id10` | 0.7873 ± 0.0407 | 0.9247 ± 0.0131 | 0.7411 ± 0.0411 | 0.9322 ± 0.0106 | 0.4970 ± 0.0819 | 0.9852 ± 0.0064 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id9` | 0.7808 ± 0.0501 | 0.9222 ± 0.0157 | 0.7396 ± 0.0538 | 0.9296 ± 0.0120 | 0.4970 ± 0.1084 | 0.9823 ± 0.0055 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+video_4` | 0.7028 ± 0.0942 | 0.9006 ± 0.0284 | 0.6585 ± 0.0774 | 0.9178 ± 0.0204 | 0.3273 ± 0.1536 | 0.9897 ± 0.0121 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+video_4` | 0.7224 ± 0.1021 | 0.9059 ± 0.0308 | 0.6812 ± 0.0916 | 0.9204 ± 0.0215 | 0.3758 ± 0.1826 | 0.9867 ± 0.0085 |
| `carewsm_bimodal_task_aware` | `audio_id1+video_4` | 0.6595 ± 0.0811 | 0.8880 ± 0.0235 | 0.6221 ± 0.0660 | 0.9099 ± 0.0155 | 0.2545 ± 0.1331 | 0.9897 ± 0.0092 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id9` | 0.6808 ± 0.0846 | 0.8940 ± 0.0251 | 0.6399 ± 0.0714 | 0.9132 ± 0.0172 | 0.2909 ± 0.1431 | 0.9889 ± 0.0094 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id10` | 0.6490 ± 0.0784 | 0.8850 ± 0.0226 | 0.6130 ± 0.0589 | 0.9079 ± 0.0151 | 0.2364 ± 0.1181 | 0.9897 ± 0.0099 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id10` | 0.6883 ± 0.0840 | 0.8964 ± 0.0246 | 0.6463 ± 0.0732 | 0.9151 ± 0.0168 | 0.3030 ± 0.1485 | 0.9897 ± 0.0109 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id10` | 0.6233 ± 0.0498 | 0.8760 ± 0.0128 | 0.5953 ± 0.0404 | 0.9000 ± 0.0106 | 0.2061 ± 0.0919 | 0.9845 ± 0.0168 |
| `carewsm_bimodal_task_aware` | `audio_id4+video_4` | 0.6761 ± 0.0956 | 0.8936 ± 0.0281 | 0.6353 ± 0.0785 | 0.9145 ± 0.0186 | 0.2788 ± 0.1566 | 0.9919 ± 0.0080 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id10` | 0.6572 ± 0.0769 | 0.8873 ± 0.0231 | 0.6191 ± 0.0625 | 0.9092 ± 0.0164 | 0.2485 ± 0.1238 | 0.9897 ± 0.0092 |
| `carewsm_bimodal_task_aware` | `audio_id2+video_4` | 0.6849 ± 0.0934 | 0.8948 ± 0.0266 | 0.6479 ± 0.0828 | 0.9132 ± 0.0162 | 0.3091 ± 0.1693 | 0.9867 ± 0.0077 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id9` | 0.6375 ± 0.0564 | 0.8823 ± 0.0158 | 0.6020 ± 0.0426 | 0.9072 ± 0.0110 | 0.2121 ± 0.0883 | 0.9919 ± 0.0109 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id9` | 0.6065 ± 0.0566 | 0.8710 ± 0.0161 | 0.5828 ± 0.0399 | 0.8967 ± 0.0154 | 0.1818 ± 0.0883 | 0.9838 ± 0.0201 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id10` | 0.6619 ± 0.1051 | 0.8879 ± 0.0281 | 0.6347 ± 0.0945 | 0.9086 ± 0.0140 | 0.2848 ± 0.1982 | 0.9845 ± 0.0109 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id9` | 0.6531 ± 0.1062 | 0.8861 ± 0.0286 | 0.6267 ± 0.0976 | 0.9086 ± 0.0146 | 0.2667 ± 0.2049 | 0.9867 ± 0.0116 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id10` | 0.6371 ± 0.0732 | 0.8813 ± 0.0191 | 0.6062 ± 0.0560 | 0.9053 ± 0.0110 | 0.2242 ± 0.1205 | 0.9882 ± 0.0141 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id9` | 0.6471 ± 0.0678 | 0.8840 ± 0.0213 | 0.6096 ± 0.0507 | 0.9066 ± 0.0169 | 0.2303 ± 0.0973 | 0.9889 ± 0.0117 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id9` | 0.6232 ± 0.0762 | 0.8778 ± 0.0203 | 0.5948 ± 0.0537 | 0.9039 ± 0.0126 | 0.2000 ± 0.1146 | 0.9897 ± 0.0151 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+video_4` | 0.6776 ± 0.1082 | 0.8936 ± 0.0324 | 0.6376 ± 0.0835 | 0.9138 ± 0.0226 | 0.2848 ± 0.1640 | 0.9904 ± 0.0118 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6887 ± 0.0782 | 0.8980 ± 0.0230 | 0.6402 ± 0.0618 | 0.9184 ± 0.0150 | 0.2848 ± 0.1223 | 0.9956 ± 0.0048 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6668 ± 0.0500 | 0.8899 ± 0.0142 | 0.6251 ± 0.0408 | 0.9105 ± 0.0097 | 0.2606 ± 0.0846 | 0.9897 ± 0.0092 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6782 ± 0.0884 | 0.8946 ± 0.0250 | 0.6361 ± 0.0727 | 0.9158 ± 0.0148 | 0.2788 ± 0.1475 | 0.9934 ± 0.0055 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6440 ± 0.1088 | 0.8827 ± 0.0323 | 0.6142 ± 0.0860 | 0.9053 ± 0.0222 | 0.2424 ± 0.1701 | 0.9860 ± 0.0115 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6692 ± 0.0649 | 0.8910 ± 0.0176 | 0.6285 ± 0.0529 | 0.9118 ± 0.0100 | 0.2667 ± 0.1101 | 0.9904 ± 0.0081 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6473 ± 0.0464 | 0.8843 ± 0.0121 | 0.6100 ± 0.0358 | 0.9072 ± 0.0063 | 0.2303 ± 0.0761 | 0.9897 ± 0.0071 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6555 ± 0.0914 | 0.8866 ± 0.0256 | 0.6214 ± 0.0711 | 0.9086 ± 0.0151 | 0.2545 ± 0.1447 | 0.9882 ± 0.0084 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6515 ± 0.0888 | 0.8861 ± 0.0247 | 0.6164 ± 0.0657 | 0.9092 ± 0.0148 | 0.2424 ± 0.1338 | 0.9904 ± 0.0100 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6674 ± 0.0437 | 0.8897 ± 0.0136 | 0.6248 ± 0.0335 | 0.9099 ± 0.0111 | 0.2606 ± 0.0664 | 0.9889 ± 0.0094 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6565 ± 0.0928 | 0.8868 ± 0.0268 | 0.6214 ± 0.0716 | 0.9086 ± 0.0172 | 0.2545 ± 0.1431 | 0.9882 ± 0.0084 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6558 ± 0.0425 | 0.8867 ± 0.0117 | 0.6160 ± 0.0334 | 0.9086 ± 0.0082 | 0.2424 ± 0.0711 | 0.9897 ± 0.0096 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6346 ± 0.0786 | 0.8798 ± 0.0213 | 0.6051 ± 0.0582 | 0.9033 ± 0.0129 | 0.2242 ± 0.1205 | 0.9860 ± 0.0112 |
| `carewsm_bimodal_gated_late` | `audio_id1+video_4` | 0.6994 ± 0.0862 | 0.8999 ± 0.0249 | 0.6558 ± 0.0744 | 0.9178 ± 0.0163 | 0.3212 ± 0.1509 | 0.9904 ± 0.0089 |
| `carewsm_bimodal_gated_late` | `audio_id4+video_4` | 0.6669 ± 0.0707 | 0.8893 ± 0.0194 | 0.6297 ± 0.0585 | 0.9092 ± 0.0115 | 0.2727 ± 0.1212 | 0.9867 ± 0.0089 |
| `carewsm_bimodal_gated_late` | `audio_id2+video_4` | 0.6305 ± 0.0705 | 0.8754 ± 0.0189 | 0.6060 ± 0.0566 | 0.8954 ± 0.0102 | 0.2364 ± 0.1181 | 0.9756 ± 0.0085 |
| `carewsm_bimodal_gated_late` | `video_4+text_id10` | 0.5768 ± 0.0582 | 0.8430 ± 0.0119 | 0.5809 ± 0.0596 | 0.8507 ± 0.0177 | 0.2364 ± 0.1475 | 0.9255 ± 0.0336 |
| `carewsm_bimodal_task_aware` | `video_4+text_id9` | 0.5979 ± 0.0437 | 0.8579 ± 0.0095 | 0.5878 ± 0.0439 | 0.8724 ± 0.0113 | 0.2242 ± 0.1041 | 0.9513 ± 0.0208 |
| `carewsm_bimodal_task_aware` | `video_4+text_id10` | 0.6028 ± 0.0507 | 0.8593 ± 0.0140 | 0.5938 ± 0.0495 | 0.8737 ± 0.0211 | 0.2364 ± 0.1238 | 0.9513 ± 0.0339 |
| `carewsm_bimodal_gated_late` | `video_4+text_id9` | 0.5765 ± 0.0449 | 0.8432 ± 0.0114 | 0.5786 ± 0.0475 | 0.8513 ± 0.0237 | 0.2303 ± 0.1278 | 0.9269 ± 0.0393 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id9` | 0.5957 ± 0.0377 | 0.8526 ± 0.0096 | 0.5902 ± 0.0389 | 0.8625 ± 0.0172 | 0.2424 ± 0.1005 | 0.9380 ± 0.0285 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id10` | 0.5854 ± 0.0453 | 0.8508 ± 0.0053 | 0.5826 ± 0.0457 | 0.8632 ± 0.0145 | 0.2242 ± 0.1185 | 0.9410 ± 0.0292 |

### Wsm Depression — TEST

| Model | Feature versions | Macro F1 | Weighted F1 | Macro Recall / UAR | Weighted Recall | Patient Recall | Healthy Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id10` | 0.7750 ± 0.0168 | 0.7828 ± 0.0179 | 0.7765 ± 0.0130 | 0.7825 ± 0.0189 | 0.7451 ± 0.0287 | 0.8079 ± 0.0476 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id10` | 0.7838 ± 0.0131 | 0.7910 ± 0.0125 | 0.7861 ± 0.0143 | 0.7904 ± 0.0126 | 0.7640 ± 0.0385 | 0.8083 ± 0.0259 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id9` | 0.7880 ± 0.0167 | 0.7950 ± 0.0165 | 0.7906 ± 0.0163 | 0.7943 ± 0.0170 | 0.7713 ± 0.0328 | 0.8099 ± 0.0319 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id10` | 0.7843 ± 0.0141 | 0.7925 ± 0.0141 | 0.7836 ± 0.0130 | 0.7929 ± 0.0146 | 0.7354 ± 0.0272 | 0.8318 ± 0.0308 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id9` | 0.7844 ± 0.0153 | 0.7927 ± 0.0153 | 0.7835 ± 0.0140 | 0.7931 ± 0.0158 | 0.7335 ± 0.0258 | 0.8335 ± 0.0314 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id10` | 0.7666 ± 0.0457 | 0.7738 ± 0.0472 | 0.7697 ± 0.0402 | 0.7736 ± 0.0484 | 0.7494 ± 0.0521 | 0.7901 ± 0.0930 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id9` | 0.7730 ± 0.0344 | 0.7808 ± 0.0353 | 0.7739 ± 0.0299 | 0.7808 ± 0.0368 | 0.7378 ± 0.0288 | 0.8099 ± 0.0698 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id9` | 0.7734 ± 0.0167 | 0.7822 ± 0.0167 | 0.7724 ± 0.0155 | 0.7828 ± 0.0174 | 0.7183 ± 0.0264 | 0.8264 ± 0.0330 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id9` | 0.7808 ± 0.0335 | 0.7888 ± 0.0339 | 0.7807 ± 0.0301 | 0.7889 ± 0.0352 | 0.7378 ± 0.0250 | 0.8236 ± 0.0599 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id10` | 0.7796 ± 0.0323 | 0.7876 ± 0.0327 | 0.7797 ± 0.0290 | 0.7877 ± 0.0340 | 0.7384 ± 0.0290 | 0.8211 ± 0.0600 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id10` | 0.7737 ± 0.0341 | 0.7814 ± 0.0332 | 0.7758 ± 0.0348 | 0.7810 ± 0.0335 | 0.7488 ± 0.0607 | 0.8029 ± 0.0464 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id10` | 0.7761 ± 0.0067 | 0.7840 ± 0.0061 | 0.7772 ± 0.0080 | 0.7837 ± 0.0058 | 0.7433 ± 0.0260 | 0.8112 ± 0.0148 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id9` | 0.7815 ± 0.0104 | 0.7891 ± 0.0097 | 0.7829 ± 0.0113 | 0.7887 ± 0.0096 | 0.7530 ± 0.0244 | 0.8128 ± 0.0114 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id9` | 0.7734 ± 0.0126 | 0.7816 ± 0.0122 | 0.7737 ± 0.0124 | 0.7815 ± 0.0123 | 0.7329 ± 0.0161 | 0.8145 ± 0.0143 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id10` | 0.7770 ± 0.0097 | 0.7846 ± 0.0096 | 0.7788 ± 0.0095 | 0.7840 ± 0.0098 | 0.7518 ± 0.0194 | 0.8058 ± 0.0181 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id9` | 0.7745 ± 0.0347 | 0.7819 ± 0.0347 | 0.7766 ± 0.0327 | 0.7813 ± 0.0355 | 0.7524 ± 0.0291 | 0.8008 ± 0.0509 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id10` | 0.7770 ± 0.0112 | 0.7850 ± 0.0110 | 0.7777 ± 0.0107 | 0.7847 ± 0.0113 | 0.7409 ± 0.0152 | 0.8145 ± 0.0175 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id9` | 0.7805 ± 0.0124 | 0.7878 ± 0.0121 | 0.7827 ± 0.0121 | 0.7872 ± 0.0123 | 0.7591 ± 0.0207 | 0.8062 ± 0.0198 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+video_4` | 0.7281 ± 0.0254 | 0.7372 ± 0.0263 | 0.7298 ± 0.0222 | 0.7369 ± 0.0275 | 0.6927 ± 0.0434 | 0.7669 ± 0.0611 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+video_4` | 0.7182 ± 0.0253 | 0.7280 ± 0.0257 | 0.7195 ± 0.0233 | 0.7281 ± 0.0267 | 0.6750 ± 0.0519 | 0.7640 ± 0.0590 |
| `carewsm_bimodal_task_aware` | `audio_id1+video_4` | 0.7331 ± 0.0343 | 0.7427 ± 0.0343 | 0.7334 ± 0.0321 | 0.7426 ± 0.0355 | 0.6854 ± 0.0248 | 0.7814 ± 0.0526 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id9` | 0.7366 ± 0.0335 | 0.7457 ± 0.0335 | 0.7375 ± 0.0313 | 0.7453 ± 0.0347 | 0.6970 ± 0.0296 | 0.7781 ± 0.0537 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id10` | 0.7361 ± 0.0256 | 0.7460 ± 0.0248 | 0.7361 ± 0.0255 | 0.7463 ± 0.0250 | 0.6829 ± 0.0457 | 0.7893 ± 0.0374 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id10` | 0.7324 ± 0.0310 | 0.7415 ± 0.0313 | 0.7335 ± 0.0287 | 0.7411 ± 0.0327 | 0.6939 ± 0.0314 | 0.7731 ± 0.0554 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id10` | 0.7207 ± 0.0270 | 0.7299 ± 0.0275 | 0.7228 ± 0.0251 | 0.7293 ± 0.0287 | 0.6890 ± 0.0418 | 0.7566 ± 0.0555 |
| `carewsm_bimodal_task_aware` | `audio_id4+video_4` | 0.7306 ± 0.0337 | 0.7407 ± 0.0331 | 0.7306 ± 0.0325 | 0.7414 ± 0.0336 | 0.6744 ± 0.0584 | 0.7868 ± 0.0575 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id10` | 0.7256 ± 0.0351 | 0.7347 ± 0.0355 | 0.7272 ± 0.0326 | 0.7342 ± 0.0371 | 0.6909 ± 0.0411 | 0.7636 ± 0.0648 |
| `carewsm_bimodal_task_aware` | `audio_id2+video_4` | 0.7225 ± 0.0289 | 0.7327 ± 0.0289 | 0.7225 ± 0.0272 | 0.7328 ± 0.0301 | 0.6689 ± 0.0255 | 0.7760 ± 0.0459 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id9` | 0.7327 ± 0.0389 | 0.7422 ± 0.0380 | 0.7333 ± 0.0382 | 0.7421 ± 0.0385 | 0.6872 ± 0.0500 | 0.7793 ± 0.0483 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id9` | 0.7298 ± 0.0301 | 0.7390 ± 0.0303 | 0.7314 ± 0.0280 | 0.7387 ± 0.0314 | 0.6933 ± 0.0435 | 0.7694 ± 0.0571 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id10` | 0.7225 ± 0.0339 | 0.7321 ± 0.0340 | 0.7235 ± 0.0318 | 0.7318 ± 0.0355 | 0.6805 ± 0.0332 | 0.7665 ± 0.0568 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id9` | 0.7252 ± 0.0288 | 0.7353 ± 0.0288 | 0.7251 ± 0.0272 | 0.7355 ± 0.0300 | 0.6713 ± 0.0297 | 0.7789 ± 0.0468 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id10` | 0.7205 ± 0.0407 | 0.7295 ± 0.0412 | 0.7224 ± 0.0379 | 0.7288 ± 0.0426 | 0.6890 ± 0.0334 | 0.7558 ± 0.0671 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id9` | 0.7307 ± 0.0404 | 0.7399 ± 0.0402 | 0.7318 ± 0.0385 | 0.7397 ± 0.0415 | 0.6909 ± 0.0439 | 0.7727 ± 0.0624 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id9` | 0.7188 ± 0.0336 | 0.7279 ± 0.0336 | 0.7209 ± 0.0318 | 0.7271 ± 0.0347 | 0.6884 ± 0.0347 | 0.7533 ± 0.0533 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+video_4` | 0.7099 ± 0.0270 | 0.7206 ± 0.0264 | 0.7103 ± 0.0266 | 0.7212 ± 0.0269 | 0.6537 ± 0.0567 | 0.7669 ± 0.0505 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.7234 ± 0.0450 | 0.7335 ± 0.0442 | 0.7236 ± 0.0437 | 0.7335 ± 0.0451 | 0.6720 ± 0.0479 | 0.7752 ± 0.0572 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.7202 ± 0.0352 | 0.7290 ± 0.0352 | 0.7228 ± 0.0337 | 0.7281 ± 0.0362 | 0.6951 ± 0.0422 | 0.7504 ± 0.0556 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.7254 ± 0.0380 | 0.7358 ± 0.0371 | 0.7251 ± 0.0374 | 0.7362 ± 0.0377 | 0.6671 ± 0.0491 | 0.7831 ± 0.0474 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.7238 ± 0.0348 | 0.7341 ± 0.0344 | 0.7234 ± 0.0334 | 0.7345 ± 0.0355 | 0.6659 ± 0.0360 | 0.7810 ± 0.0500 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.7250 ± 0.0379 | 0.7352 ± 0.0373 | 0.7248 ± 0.0370 | 0.7357 ± 0.0383 | 0.6683 ± 0.0521 | 0.7814 ± 0.0559 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.7186 ± 0.0387 | 0.7282 ± 0.0380 | 0.7201 ± 0.0381 | 0.7278 ± 0.0388 | 0.6799 ± 0.0562 | 0.7603 ± 0.0555 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.7212 ± 0.0373 | 0.7313 ± 0.0371 | 0.7212 ± 0.0356 | 0.7315 ± 0.0386 | 0.6677 ± 0.0415 | 0.7748 ± 0.0594 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.7147 ± 0.0470 | 0.7255 ± 0.0463 | 0.7142 ± 0.0456 | 0.7261 ± 0.0476 | 0.6524 ± 0.0533 | 0.7760 ± 0.0652 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.7074 ± 0.0407 | 0.7168 ± 0.0408 | 0.7096 ± 0.0389 | 0.7160 ± 0.0420 | 0.6762 ± 0.0455 | 0.7430 ± 0.0635 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.7175 ± 0.0361 | 0.7272 ± 0.0363 | 0.7186 ± 0.0338 | 0.7271 ± 0.0377 | 0.6744 ± 0.0441 | 0.7628 ± 0.0640 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.7085 ± 0.0392 | 0.7181 ± 0.0388 | 0.7104 ± 0.0382 | 0.7175 ± 0.0398 | 0.6738 ± 0.0532 | 0.7471 ± 0.0587 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.7226 ± 0.0283 | 0.7324 ± 0.0287 | 0.7232 ± 0.0263 | 0.7325 ± 0.0302 | 0.6750 ± 0.0418 | 0.7715 ± 0.0578 |
| `carewsm_bimodal_gated_late` | `audio_id1+video_4` | 0.6884 ± 0.0421 | 0.7012 ± 0.0411 | 0.6873 ± 0.0414 | 0.7032 ± 0.0421 | 0.6043 ± 0.0640 | 0.7702 ± 0.0617 |
| `carewsm_bimodal_gated_late` | `audio_id4+video_4` | 0.6807 ± 0.0393 | 0.6930 ± 0.0384 | 0.6807 ± 0.0389 | 0.6941 ± 0.0392 | 0.6110 ± 0.0653 | 0.7504 ± 0.0599 |
| `carewsm_bimodal_gated_late` | `audio_id2+video_4` | 0.6911 ± 0.0348 | 0.7036 ± 0.0328 | 0.6906 ± 0.0360 | 0.7054 ± 0.0326 | 0.6134 ± 0.0762 | 0.7678 ± 0.0481 |
| `carewsm_bimodal_gated_late` | `video_4+text_id10` | 0.6615 ± 0.0342 | 0.6741 ± 0.0334 | 0.6619 ± 0.0337 | 0.6749 ± 0.0339 | 0.5945 ± 0.0618 | 0.7293 ± 0.0553 |
| `carewsm_bimodal_task_aware` | `video_4+text_id9` | 0.6752 ± 0.0379 | 0.6867 ± 0.0378 | 0.6757 ± 0.0365 | 0.6865 ± 0.0390 | 0.6195 ± 0.0349 | 0.7318 ± 0.0540 |
| `carewsm_bimodal_task_aware` | `video_4+text_id10` | 0.6689 ± 0.0349 | 0.6810 ± 0.0336 | 0.6693 ± 0.0349 | 0.6810 ± 0.0338 | 0.6079 ± 0.0510 | 0.7306 ± 0.0382 |
| `carewsm_bimodal_gated_late` | `video_4+text_id9` | 0.6624 ± 0.0377 | 0.6758 ± 0.0367 | 0.6621 ± 0.0372 | 0.6773 ± 0.0371 | 0.5829 ± 0.0624 | 0.7413 ± 0.0551 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id9` | 0.6681 ± 0.0339 | 0.6799 ± 0.0339 | 0.6687 ± 0.0327 | 0.6798 ± 0.0351 | 0.6110 ± 0.0398 | 0.7264 ± 0.0533 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id10` | 0.6665 ± 0.0335 | 0.6785 ± 0.0335 | 0.6669 ± 0.0323 | 0.6786 ± 0.0347 | 0.6061 ± 0.0387 | 0.7277 ± 0.0524 |

### Wsm Parkinson — TEST

| Model | Feature versions | Macro F1 | Weighted F1 | Macro Recall / UAR | Weighted Recall | Patient Recall | Healthy Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id10` | 0.7804 ± 0.0461 | 0.8380 ± 0.0348 | 0.7747 ± 0.0457 | 0.8397 ± 0.0354 | 0.6458 ± 0.0735 | 0.9035 ± 0.0315 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id10` | 0.7753 ± 0.0472 | 0.8336 ± 0.0370 | 0.7699 ± 0.0407 | 0.8348 ± 0.0389 | 0.6412 ± 0.0474 | 0.8985 ± 0.0385 |
| `carewsm_bimodal_task_aware` | `audio_id2+text_id9` | 0.7864 ± 0.0478 | 0.8425 ± 0.0366 | 0.7792 ± 0.0426 | 0.8442 ± 0.0378 | 0.6504 ± 0.0554 | 0.9080 ± 0.0348 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id10` | 0.7808 ± 0.0331 | 0.8394 ± 0.0263 | 0.7703 ± 0.0264 | 0.8423 ± 0.0283 | 0.6275 ± 0.0364 | 0.9131 ± 0.0343 |
| `carewsm_bimodal_gated_late` | `audio_id2+text_id9` | 0.7802 ± 0.0316 | 0.8388 ± 0.0245 | 0.7708 ± 0.0273 | 0.8416 ± 0.0257 | 0.6305 ± 0.0391 | 0.9111 ± 0.0280 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id10` | 0.7465 ± 0.0351 | 0.8134 ± 0.0260 | 0.7404 ± 0.0342 | 0.8159 ± 0.0261 | 0.5908 ± 0.0544 | 0.8899 ± 0.0216 |
| `carewsm_bimodal_task_aware` | `audio_id1+text_id9` | 0.7509 ± 0.0354 | 0.8176 ± 0.0258 | 0.7424 ± 0.0350 | 0.8212 ± 0.0255 | 0.5863 ± 0.0597 | 0.8985 ± 0.0219 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+text_id9` | 0.7807 ± 0.0368 | 0.8386 ± 0.0268 | 0.7744 ± 0.0384 | 0.8408 ± 0.0265 | 0.6427 ± 0.0731 | 0.9060 ± 0.0267 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id9` | 0.7582 ± 0.0337 | 0.8244 ± 0.0251 | 0.7453 ± 0.0325 | 0.8295 ± 0.0258 | 0.5786 ± 0.0609 | 0.9121 ± 0.0300 |
| `carewsm_bimodal_gated_late` | `audio_id1+text_id10` | 0.7567 ± 0.0323 | 0.8233 ± 0.0244 | 0.7441 ± 0.0308 | 0.8284 ± 0.0254 | 0.5771 ± 0.0590 | 0.9111 ± 0.0313 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id10` | 0.7545 ± 0.0394 | 0.8194 ± 0.0281 | 0.7490 ± 0.0428 | 0.8219 ± 0.0272 | 0.6046 ± 0.0809 | 0.8935 ± 0.0223 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id10` | 0.8017 ± 0.0267 | 0.8487 ± 0.0211 | 0.8163 ± 0.0263 | 0.8454 ± 0.0221 | 0.7588 ± 0.0406 | 0.8739 ± 0.0218 |
| `carewsm_bimodal_task_aware` | `audio_id4+text_id9` | 0.8085 ± 0.0258 | 0.8546 ± 0.0210 | 0.8198 ± 0.0254 | 0.8522 ± 0.0227 | 0.7557 ± 0.0483 | 0.8839 ± 0.0294 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id9` | 0.8175 ± 0.0210 | 0.8637 ± 0.0169 | 0.8181 ± 0.0172 | 0.8635 ± 0.0182 | 0.7282 ± 0.0284 | 0.9080 ± 0.0236 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id10` | 0.8076 ± 0.0380 | 0.8539 ± 0.0310 | 0.8173 ± 0.0304 | 0.8514 ± 0.0335 | 0.7496 ± 0.0302 | 0.8849 ± 0.0379 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+text_id9` | 0.7583 ± 0.0465 | 0.8223 ± 0.0333 | 0.7526 ± 0.0497 | 0.8250 ± 0.0322 | 0.6092 ± 0.0897 | 0.8960 ± 0.0232 |
| `carewsm_bimodal_gated_late` | `audio_id4+text_id10` | 0.8057 ± 0.0241 | 0.8554 ± 0.0180 | 0.8052 ± 0.0253 | 0.8556 ± 0.0182 | 0.7053 ± 0.0482 | 0.9050 ± 0.0194 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+text_id9` | 0.8107 ± 0.0251 | 0.8562 ± 0.0194 | 0.8224 ± 0.0240 | 0.8537 ± 0.0201 | 0.7603 ± 0.0322 | 0.8844 ± 0.0167 |
| `carewsm_bimodal_shared_mtl` | `audio_id1+video_4` | 0.6363 ± 0.0420 | 0.7392 ± 0.0398 | 0.6268 ± 0.0338 | 0.7520 ± 0.0521 | 0.3786 ± 0.0647 | 0.8749 ± 0.0793 |
| `carewsm_bimodal_shared_mtl` | `audio_id2+video_4` | 0.6184 ± 0.0398 | 0.7241 ± 0.0371 | 0.6121 ± 0.0339 | 0.7346 ± 0.0488 | 0.3695 ± 0.0640 | 0.8548 ± 0.0732 |
| `carewsm_bimodal_task_aware` | `audio_id1+video_4` | 0.6524 ± 0.0524 | 0.7571 ± 0.0359 | 0.6397 ± 0.0508 | 0.7769 ± 0.0366 | 0.3679 ± 0.1088 | 0.9116 ± 0.0485 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id9` | 0.6212 ± 0.0422 | 0.7303 ± 0.0374 | 0.6130 ± 0.0393 | 0.7459 ± 0.0487 | 0.3496 ± 0.0800 | 0.8764 ± 0.0728 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id10` | 0.6464 ± 0.0226 | 0.7482 ± 0.0090 | 0.6389 ± 0.0265 | 0.7626 ± 0.0169 | 0.3939 ± 0.1014 | 0.8839 ± 0.0525 |
| `carewsm_trimodal_cross_attention` | `audio_id2+video_4+text_id10` | 0.6117 ± 0.0424 | 0.7228 ± 0.0353 | 0.6049 ± 0.0401 | 0.7376 ± 0.0441 | 0.3420 ± 0.0794 | 0.8678 ± 0.0636 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id10` | 0.6419 ± 0.0312 | 0.7460 ± 0.0291 | 0.6315 ± 0.0252 | 0.7622 ± 0.0401 | 0.3725 ± 0.0826 | 0.8905 ± 0.0724 |
| `carewsm_bimodal_task_aware` | `audio_id4+video_4` | 0.6800 ± 0.0160 | 0.7748 ± 0.0071 | 0.6649 ± 0.0214 | 0.7909 ± 0.0134 | 0.4153 ± 0.0816 | 0.9146 ± 0.0420 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id10` | 0.6165 ± 0.0317 | 0.7258 ± 0.0266 | 0.6092 ± 0.0287 | 0.7395 ± 0.0339 | 0.3511 ± 0.0620 | 0.8673 ± 0.0517 |
| `carewsm_bimodal_task_aware` | `audio_id2+video_4` | 0.6271 ± 0.0595 | 0.7367 ± 0.0439 | 0.6183 ± 0.0561 | 0.7546 ± 0.0478 | 0.3481 ± 0.1072 | 0.8884 ± 0.0601 |
| `carewsm_trimodal_cross_attention` | `audio_id4+video_4+text_id9` | 0.6406 ± 0.0513 | 0.7431 ± 0.0395 | 0.6321 ± 0.0501 | 0.7561 ± 0.0447 | 0.3863 ± 0.0984 | 0.8779 ± 0.0593 |
| `carewsm_shared_private_trimodal` | `audio_id4+video_4+text_id9` | 0.6550 ± 0.0300 | 0.7554 ± 0.0207 | 0.6442 ± 0.0305 | 0.7705 ± 0.0250 | 0.3939 ± 0.0909 | 0.8945 ± 0.0505 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id10` | 0.6101 ± 0.0472 | 0.7180 ± 0.0380 | 0.6068 ± 0.0481 | 0.7289 ± 0.0481 | 0.3649 ± 0.1119 | 0.8487 ± 0.0753 |
| `carewsm_shared_private_trimodal` | `audio_id2+video_4+text_id9` | 0.6146 ± 0.0546 | 0.7221 ± 0.0421 | 0.6106 ± 0.0555 | 0.7338 ± 0.0497 | 0.3664 ± 0.1176 | 0.8548 ± 0.0708 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id10` | 0.6132 ± 0.0375 | 0.7247 ± 0.0290 | 0.6077 ± 0.0379 | 0.7410 ± 0.0381 | 0.3435 ± 0.1071 | 0.8719 ± 0.0692 |
| `carewsm_trimodal_cross_attention` | `audio_id1+video_4+text_id9` | 0.6170 ± 0.0268 | 0.7265 ± 0.0232 | 0.6102 ± 0.0251 | 0.7410 ± 0.0329 | 0.3511 ± 0.0708 | 0.8693 ± 0.0570 |
| `carewsm_shared_private_trimodal` | `audio_id1+video_4+text_id9` | 0.6191 ± 0.0455 | 0.7304 ± 0.0301 | 0.6127 ± 0.0461 | 0.7478 ± 0.0319 | 0.3450 ± 0.1126 | 0.8804 ± 0.0518 |
| `carewsm_bimodal_shared_mtl` | `audio_id4+video_4` | 0.6549 ± 0.0498 | 0.7521 ± 0.0421 | 0.6440 ± 0.0422 | 0.7633 ± 0.0503 | 0.4076 ± 0.0684 | 0.8804 ± 0.0686 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6247 ± 0.0287 | 0.7357 ± 0.0160 | 0.6172 ± 0.0321 | 0.7546 ± 0.0203 | 0.3450 ± 0.1006 | 0.8894 ± 0.0489 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6494 ± 0.0472 | 0.7504 ± 0.0327 | 0.6402 ± 0.0460 | 0.7645 ± 0.0345 | 0.3939 ± 0.1047 | 0.8864 ± 0.0509 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id10` | 0.6273 ± 0.0264 | 0.7371 ± 0.0200 | 0.6188 ± 0.0285 | 0.7554 ± 0.0291 | 0.3481 ± 0.0911 | 0.8894 ± 0.0579 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6173 ± 0.0454 | 0.7263 ± 0.0363 | 0.6115 ± 0.0447 | 0.7406 ± 0.0464 | 0.3557 ± 0.1038 | 0.8673 ± 0.0729 |
| `carewsm_audio_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6180 ± 0.0309 | 0.7291 ± 0.0269 | 0.6104 ± 0.0295 | 0.7459 ± 0.0383 | 0.3420 ± 0.0796 | 0.8789 ± 0.0651 |
| `carewsm_audio_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6457 ± 0.0392 | 0.7487 ± 0.0282 | 0.6353 ± 0.0375 | 0.7641 ± 0.0315 | 0.3802 ± 0.0863 | 0.8905 ± 0.0480 |
| `carewsm_audio_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.5925 ± 0.0391 | 0.7099 ± 0.0312 | 0.5884 ± 0.0395 | 0.7267 ± 0.0412 | 0.3145 ± 0.0931 | 0.8623 ± 0.0653 |
| `carewsm_text_guided_dual_gated` | `audio_id1+video_4+text_id9` | 0.6192 ± 0.0301 | 0.7306 ± 0.0193 | 0.6125 ± 0.0311 | 0.7482 ± 0.0256 | 0.3435 ± 0.0929 | 0.8814 ± 0.0517 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id10` | 0.6503 ± 0.0364 | 0.7491 ± 0.0256 | 0.6428 ± 0.0378 | 0.7607 ± 0.0301 | 0.4092 ± 0.0992 | 0.8764 ± 0.0532 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id10` | 0.6162 ± 0.0325 | 0.7284 ± 0.0210 | 0.6107 ± 0.0353 | 0.7463 ± 0.0290 | 0.3420 ± 0.1081 | 0.8794 ± 0.0602 |
| `carewsm_text_guided_dual_gated` | `audio_id4+video_4+text_id9` | 0.6476 ± 0.0375 | 0.7480 ± 0.0273 | 0.6387 ± 0.0368 | 0.7607 ± 0.0312 | 0.3969 ± 0.0865 | 0.8804 ± 0.0486 |
| `carewsm_text_guided_dual_gated` | `audio_id2+video_4+text_id9` | 0.6215 ± 0.0461 | 0.7279 ± 0.0401 | 0.6144 ± 0.0434 | 0.7403 ± 0.0511 | 0.3649 ± 0.0826 | 0.8638 ± 0.0736 |
| `carewsm_bimodal_gated_late` | `audio_id1+video_4` | 0.6761 ± 0.0223 | 0.7730 ± 0.0139 | 0.6601 ± 0.0239 | 0.7905 ± 0.0177 | 0.4015 ± 0.0765 | 0.9186 ± 0.0403 |
| `carewsm_bimodal_gated_late` | `audio_id4+video_4` | 0.6789 ± 0.0641 | 0.7669 ± 0.0559 | 0.6664 ± 0.0534 | 0.7747 ± 0.0657 | 0.4519 ± 0.0621 | 0.8809 ± 0.0840 |
| `carewsm_bimodal_gated_late` | `audio_id2+video_4` | 0.6101 ± 0.0209 | 0.7249 ± 0.0207 | 0.6023 ± 0.0180 | 0.7437 ± 0.0316 | 0.3221 ± 0.0559 | 0.8824 ± 0.0550 |
| `carewsm_bimodal_gated_late` | `video_4+text_id10` | 0.4996 ± 0.0446 | 0.6216 ± 0.0616 | 0.5017 ± 0.0430 | 0.6216 ± 0.0875 | 0.2641 ± 0.0557 | 0.7392 ± 0.1325 |
| `carewsm_bimodal_task_aware` | `video_4+text_id9` | 0.5017 ± 0.0471 | 0.6209 ± 0.0589 | 0.5022 ± 0.0452 | 0.6170 ± 0.0800 | 0.2748 ± 0.0270 | 0.7296 ± 0.1144 |
| `carewsm_bimodal_task_aware` | `video_4+text_id10` | 0.4993 ± 0.0377 | 0.6194 ± 0.0548 | 0.5015 ± 0.0350 | 0.6166 ± 0.0781 | 0.2733 ± 0.0524 | 0.7296 ± 0.1207 |
| `carewsm_bimodal_gated_late` | `video_4+text_id9` | 0.5004 ± 0.0472 | 0.6242 ± 0.0641 | 0.5021 ± 0.0466 | 0.6261 ± 0.0924 | 0.2565 ± 0.0519 | 0.7477 ± 0.1382 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id9` | 0.5069 ± 0.0405 | 0.6234 ± 0.0495 | 0.5089 ± 0.0398 | 0.6178 ± 0.0689 | 0.2931 ± 0.0547 | 0.7246 ± 0.1019 |
| `carewsm_bimodal_shared_mtl` | `video_4+text_id10` | 0.5069 ± 0.0491 | 0.6232 ± 0.0615 | 0.5091 ± 0.0485 | 0.6189 ± 0.0870 | 0.2916 ± 0.0590 | 0.7266 ± 0.1282 |


## VAL-only best bimodal and trimodal variants

- **Bimodal:** `bimodal_shared_mtl__A2_T10`, VAL UAR 0.8218 ± 0.0327; descriptive TEST UAR 0.7819 ± 0.0259
- **Trimodal:** `cross_attention__A2_V4_T9`, VAL UAR 0.6995 ± 0.0465; descriptive TEST UAR 0.6695 ± 0.0275

