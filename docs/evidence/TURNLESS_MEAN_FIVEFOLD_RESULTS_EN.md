# TURNLESS MEAN Five-Fold Results

Training source commit: `8a91a06e52d4710f54503b824f76a969cf73942b`; reporting commit: `5cf80d37525552cd4afba63a6b817c5d0cf64aef`; seed 5; 20/20 runs completed.

Generated exclusively by `scripts/collect_turnless_mean_results.py` from runtime summaries, retained validation checkpoints, and local MLflow. Validation selects the checkpoint; TEST is descriptive and was not used for selection.

Historical pooling verification used `test_care` and `test_wsm` feature artifacts only for arithmetic (CARE 3 turns, WSM 1 turn; max_abs_error=0.0; output shape `[1, 2048]`). Labels and predictions were not used.

## Table A — Validation Results

Displayed names spell out the feature families and model type; E1–E4 are immutable run identifiers retained in the machine-readable ledger.

| Experiment | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± SD | Delta vs Weighted-F | Paired wins | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (64) [E1] | 0.8332 | 0.8396 | 0.7992 | 0.7870 | 0.8637 | 0.8245 ± 0.0312 | -0.0025 | 2/5 | REJECTED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (64) [E2] | 0.8311 | 0.8119 | 0.7161 | 0.8091 | 0.8414 | 0.8019 ± 0.0498 | -0.0251 | 1/5 | REJECTED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (32) [E3] | 0.8589 | 0.8343 | 0.7972 | 0.8026 | 0.8724 | 0.8331 ± 0.0333 | +0.0060 | 3/5 | REJECTED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (32) [E4] | 0.7977 | 0.7994 | 0.7200 | 0.8016 | 0.8311 | 0.7900 ± 0.0414 | -0.0371 | 1/5 | REJECTED |
| Weighted-F historical control | 0.8321 | 0.8400 | 0.7979 | 0.8010 | 0.8643 | 0.8271 ± 0.0279 | — | — | CONTROL |

## Table B — Descriptive TEST Results — not used for model selection

| Experiment | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± SD | Delta vs Weighted-F |
|---|---:|---:|---:|---:|---:|---:|---:|
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (64) [E1] | 0.7777 | 0.7285 | 0.7614 | 0.7692 | 0.7890 | 0.7652 ± 0.0229 | +0.0009 |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (64) [E2] | 0.7595 | 0.7565 | 0.7262 | 0.7629 | 0.7772 | 0.7565 ± 0.0187 | -0.0078 |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (32) [E3] | 0.7681 | 0.7549 | 0.7216 | 0.7701 | 0.7719 | 0.7573 ± 0.0211 | -0.0070 |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (32) [E4] | 0.7437 | 0.7409 | 0.6920 | 0.7755 | 0.7695 | 0.7443 ± 0.0330 | -0.0200 |
| Weighted-F historical control | 0.7547 | 0.7325 | 0.7609 | 0.7846 | 0.7888 | 0.7643 ± 0.0231 | — |

Acceptance is validation-only: +0.010 mean VAL UAR was required, with no component mean deterioration above 0.020. All four configurations are rejected; no significance or clinical superiority is claimed.

## Table C — Five-fold mean Macro Recall by corpus/task

| Experiment | CARE Depression VAL/TEST | CARE Parkinson VAL/TEST | WSM Depression VAL/TEST | WSM Parkinson VAL/TEST |
|---|---:|---:|---:|---:|
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (64) [E1] | 0.8296/0.7512 | 0.8124/0.7711 | 0.7609/0.7516 | 0.8953/0.7866 |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (64) [E2] | 0.8190/0.7573 | 0.7979/0.7703 | 0.7302/0.7556 | 0.8606/0.7427 |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (32) [E3] | 0.8454/0.7631 | 0.8364/0.7615 | 0.7464/0.7640 | 0.9043/0.7408 |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (32) [E4] | 0.7962/0.7243 | 0.7659/0.7680 | 0.7331/0.7378 | 0.8647/0.7472 |

## Table D — Patient/Healthy Recall by corpus/task

| Experiment | CARE Depression VAL patient/healthy | CARE Depression TEST patient/healthy | CARE Parkinson VAL patient/healthy | CARE Parkinson TEST patient/healthy | WSM Depression VAL patient/healthy | WSM Depression TEST patient/healthy | WSM Parkinson VAL patient/healthy | WSM Parkinson TEST patient/healthy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (64) [E1] | 0.7454/0.9138 | 0.5650/0.9376 | 0.6852/0.9395 | 0.5824/0.9599 | 0.7458/0.7759 | 0.7540/0.7492 | 0.9133/0.8773 | 0.7203/0.8530 |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (64) [E2] | 0.7554/0.8826 | 0.6186/0.8959 | 0.7060/0.8898 | 0.6294/0.9113 | 0.7102/0.7502 | 0.7558/0.7553 | 0.9130/0.8083 | 0.7774/0.7079 |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (32) [E3] | 0.7675/0.9232 | 0.5732/0.9530 | 0.7186/0.9542 | 0.5764/0.9465 | 0.7236/0.7693 | 0.7540/0.7740 | 0.9072/0.9013 | 0.6271/0.8545 |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (32) [E4] | 0.7025/0.8900 | 0.5382/0.9104 | 0.6664/0.8655 | 0.6353/0.9007 | 0.7191/0.7471 | 0.7678/0.7077 | 0.9109/0.8185 | 0.7925/0.7020 |

## Table E — Reproducibility

| Experiment | Fold | MLflow run ID | Selected epoch | Selected checkpoint | Status |
|---|---:|---|---:|---|---|
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (64) [E1] | 0 | 2a470983fbf34cd283b2cdd5cd5e68c0 | 8 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E1_gated64_fold0_2026-11-12_13-38_carewsm_gated_fusion_38c44458/checkpoints/epoch=8_val_mean_corpora_tasks_macro_recall=0.8332.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (64) [E1] | 1 | d2ee7b73ef4a49f082946a888bc9286e | 3 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E1_gated64_fold1_2026-11-12_13-40_carewsm_gated_fusion_878d2eff/checkpoints/epoch=3_val_mean_corpora_tasks_macro_recall=0.8396.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (64) [E1] | 2 | 0217d3a87b7a4387b8faed5ca3066846 | 11 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E1_gated64_fold2_2026-11-12_13-41_carewsm_gated_fusion_e9ae3d7a/checkpoints/epoch=11_val_mean_corpora_tasks_macro_recall=0.7992.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (64) [E1] | 3 | e5f1d41cec4b48908f60e3101f3f6824 | 18 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E1_gated64_fold3_2026-11-12_13-43_carewsm_gated_fusion_7bf5e384/checkpoints/epoch=18_val_mean_corpora_tasks_macro_recall=0.7870.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (64) [E1] | 4 | c9cd45d0c0fb4e7cb1bdb580e8af97ce | 7 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E1_gated64_fold4_2026-11-12_13-45_carewsm_gated_fusion_de80556f/checkpoints/epoch=7_val_mean_corpora_tasks_macro_recall=0.8637.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (64) [E2] | 0 | 544a8357d52841c49c619730adaddeed | 12 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E2_mlp64_fold0_2026-11-12_13-47_carewsm_pooled_mlp_86a3bf6c/checkpoints/epoch=12_val_mean_corpora_tasks_macro_recall=0.8311.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (64) [E2] | 1 | 8a98646a24f246afb1a309699dd72543 | 12 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E2_mlp64_fold1_2026-11-12_13-49_carewsm_pooled_mlp_ee88e918/checkpoints/epoch=12_val_mean_corpora_tasks_macro_recall=0.8119.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (64) [E2] | 2 | 302d460674ab4270b3b2dac135c2fd5d | 5 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E2_mlp64_fold2_2026-11-12_13-51_carewsm_pooled_mlp_850c8ac9/checkpoints/epoch=5_val_mean_corpora_tasks_macro_recall=0.7161.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (64) [E2] | 3 | ae6d957055ed40a5870fd11ed8ac7b66 | 15 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E2_mlp64_fold3_2026-11-12_13-52_carewsm_pooled_mlp_37358cbb/checkpoints/epoch=15_val_mean_corpora_tasks_macro_recall=0.8091.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (64) [E2] | 4 | 56bb22ee50144e9e9765505cbb09fc28 | 19 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E2_mlp64_fold4_2026-11-12_13-55_carewsm_pooled_mlp_88a8c0f4/checkpoints/epoch=19_val_mean_corpora_tasks_macro_recall=0.8414.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (32) [E3] | 0 | 7b9ee7c1d0564a088c627d3f9a1c627c | 15 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E3_gated32_fold0_2026-11-12_13-57_carewsm_gated_fusion_57f0c709/checkpoints/epoch=15_val_mean_corpora_tasks_macro_recall=0.8590.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (32) [E3] | 1 | 3abd7055d53841f38454835fde298f76 | 17 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E3_gated32_fold1_2026-11-12_14-00_carewsm_gated_fusion_4d317e3f/checkpoints/epoch=17_val_mean_corpora_tasks_macro_recall=0.8343.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (32) [E3] | 2 | 0579cd3e9d934391ba3b99f6de6f2eca | 30 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E3_gated32_fold2_2026-11-12_14-02_carewsm_gated_fusion_e42bfa4a/checkpoints/epoch=30_val_mean_corpora_tasks_macro_recall=0.7972.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (32) [E3] | 3 | 707208da71ca4b63a8dc2a78286dc95d | 15 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E3_gated32_fold3_2026-11-12_14-06_carewsm_gated_fusion_0f6792c4/checkpoints/epoch=15_val_mean_corpora_tasks_macro_recall=0.8026.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Gated Fusion (32) [E3] | 4 | 547754d2b9c748259b7ba57f5d214321 | 18 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E3_gated32_fold4_2026-11-12_14-09_carewsm_gated_fusion_aa044de6/checkpoints/epoch=18_val_mean_corpora_tasks_macro_recall=0.8724.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (32) [E4] | 0 | c6f3a4b2b64849e08fe29d710f38d5d5 | 8 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E4_mlp32_fold0_2026-11-12_14-11_carewsm_pooled_mlp_bc449d1c/checkpoints/epoch=8_val_mean_corpora_tasks_macro_recall=0.7977.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (32) [E4] | 1 | 86e1545be618485fa2bce0f5c2a68ba1 | 13 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E4_mlp32_fold1_2026-11-12_14-13_carewsm_pooled_mlp_9878520d/checkpoints/epoch=13_val_mean_corpora_tasks_macro_recall=0.7994.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (32) [E4] | 2 | e0b2331f1c4b4842a48a1ee925266689 | 5 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E4_mlp32_fold2_2026-11-12_14-15_carewsm_pooled_mlp_bf683ff6/checkpoints/epoch=5_val_mean_corpora_tasks_macro_recall=0.7200.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (32) [E4] | 3 | 798ecd034c274548a1c37fb917219656 | 20 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E4_mlp32_fold3_2026-11-12_14-16_carewsm_pooled_mlp_6bd226e0/checkpoints/epoch=20_val_mean_corpora_tasks_macro_recall=0.8017.pt` | COMPLETED |
| EmotionMSP Embeddings + TRILLsson — Mean, Pooled MLP (32) [E4] | 4 | d0c9a005ed4e4b149fecb7e10f98bac9 | 12 | `logs/audio_representation_pilot/turnless_mean/audio_representation_pilot_turnless_mean/E4_mlp32_fold4_2026-11-12_14-19_carewsm_pooled_mlp_5a04628f/checkpoints/epoch=12_val_mean_corpora_tasks_macro_recall=0.8311.pt` | COMPLETED |

TEST is previously observed descriptive evidence, not a pristine independent holdout.

Feature families: `emotionmsp_embeds` + `trillsson4_1`; arithmetic mean over valid turns followed by concatenation; joint CARE+WSM five-fold training; seed 5; AdamW; learning rate 1e-4; batch size 64; dropout 0.1; maximum 50 epochs; patience 7. Parkinson class weights were TRAIN-only.

The collector fails on missing or ambiguous runtimes/checkpoints, monitor disagreement, incomplete MLflow provenance, or non-finite metrics; it never falls back to an arbitrary checkpoint.
