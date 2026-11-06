# DEEP-008 Fold-0 Representation Pilot

## Outcome

The six approved DEEP-008 fold-0 arms completed sequentially on the fixed joint
CARE+WSM training population. All six runs finished successfully with MLflow
status `FINISHED`. Selection used only the canonical validation monitor
`val_mean_corpora_tasks/macro_recall`; TEST values are descriptive only.

The best fold-0 validation result was arm F (gated EmotionMSP+TRILLsson native
fusion), with Macro Recall **0.8023** at epoch 14. This is a one-fold pilot
result and does not establish a five-fold improvement or guarantee a target in
the 0.82--0.85 range.

## Reproducibility and runtime

- Training source commit: `2d9102cedbdc32dbc74901ce088561d7fd92e8c1`
- Branch: `codex/deep-008-fold0-pilot`
- Required base commit: `0ed3ad0c8f6a1b4c57c83d4329470e852ad5545b`
- Seed: 5
- Fold: 0
- Maximum epochs: 50
- Early stopping patience: 7
- Device: CUDA, NVIDIA GeForce RTX 4080
- PyTorch: `2.11.0+cu128` (CUDA 12.8)
- Chimera-ML: `0.3.0`
- MLflow: `2.22.5`
- Training population: joint train 6,998; CARE validation 352; WSM validation 1,429
- Launcher: `scripts/run_audio_deep008_pilot.sh --all`
- Launcher progress: `logs/audio_representation_pilot/deep008_launcher/progress.tsv`
- MLflow database: `logs/audio_representation_pilot/mlflow.db`

The canonical fixed splits, native feature artifacts, ordered turn IDs, masks,
and masked multitask loss were unchanged. No MantisV2 run or additional fold,
search, or experiment was launched.

## Runs and selected validation evidence

The selected epoch is the row with the maximum canonical validation monitor in
the run's `summary.txt`. The four component values are CARE Depression, CARE
Parkinson, WSM Depression, and WSM Parkinson, respectively. The aggregate is
their equal-weight arithmetic mean.

| Arm | Representation/model | Runtime directory | MLflow run ID | Status | Epoch | Best checkpoint | Component validation Macro Recall | Aggregate | Patient Recall components | Healthy Recall components |
|---|---|---|---|---|---:|---|---|---:|---|---|
| A | Native EmotionMSP, MLP-64 | `A_native_mlp64_fold0_2026-11-06_13-11_carewsm_pooled_mlp_5ce940d8` | `08a203a1aec044779c60d8a3f06d6d7b` | FINISHED | 15 | `epoch=15_val_mean_corpora_tasks_macro_recall=0.7713.pt` | 0.8034 / 0.7070 / 0.7849 / 0.7899 | 0.7713 | 0.6744 / 0.4706 / 0.7588 / 0.6943 | 0.9323 / 0.9434 / 0.8110 / 0.8854 |
| B | Input masked mean, MLP-64 | `B_mean_mlp64_fold0_2026-11-06_13-13_carewsm_pooled_mlp_c825ec6e` | `1e40a71373414150893249d8056d2b9a` | FINISHED | 15 | `epoch=15_val_mean_corpora_tasks_macro_recall=0.7695.pt` | 0.8129 / 0.7007 / 0.7758 / 0.7886 | 0.7695 | 0.6860 / 0.4706 / 0.7425 / 0.6856 | 0.9398 / 0.9308 / 0.8091 / 0.8916 |
| C | Input masked mean+std, MLP-64 | `C_mean_std_mlp64_fold0_2026-11-06_13-15_carewsm_pooled_mlp_6b13de49` | `b1c32fe199804f918a4569b92c9d3b7f` | FINISHED | 12 | `epoch=12_val_mean_corpora_tasks_macro_recall=0.7483.pt` | 0.8268 / 0.6313 / 0.7588 / 0.7763 | 0.7483 | 0.7326 / 0.2941 / 0.7263 / 0.6332 | 0.9211 / 0.9686 / 0.7913 / 0.9195 |
| D | Learned masked attentive mean, MLP-64 | `D_attentive_mlp64_fold0_2026-11-06_13-16_carewsm_attentive_mlp_bccd7a95` | `e8ecccc4a8944aee84854a8eda608115` | FINISHED | 20 | `epoch=20_val_mean_corpora_tasks_macro_recall=0.7741.pt` | 0.7743 / 0.7264 / 0.7829 / 0.8127 | 0.7741 | 0.6163 / 0.5000 / 0.7371 / 0.7555 | 0.9323 / 0.9528 / 0.8287 / 0.8700 |
| E | Native EmotionMSP, MLP-128 | `E_native_mlp128_fold0_2026-11-06_13-19_carewsm_pooled_mlp_2b9e06f9` | `40b2e29f5b2344029af09a9c13637a93` | FINISHED | 9 | `epoch=9_val_mean_corpora_tasks_macro_recall=0.7606.pt` | 0.8210 / 0.6235 / 0.7783 / 0.8197 | 0.7606 | 0.7209 / 0.2941 / 0.7967 / 0.7293 | 0.9211 / 0.9528 / 0.7598 / 0.9102 |
| F | Gated EmotionMSP+TRILLsson native, hidden-64 | `F_gated_emotion_trillsson_native64_fold0_2026-11-06_13-20_carewsm_gated_fusion_9ab3fc26` | `ddf7c9e3e72641abafca3e1a426cd58f` | FINISHED | 14 | `epoch=14_val_mean_corpora_tasks_macro_recall=0.8023.pt` | 0.8225 / 0.6739 / 0.7823 / 0.9304 | 0.8023 | 0.6977 / 0.3824 / 0.7615 / 0.9258 | 0.9474 / 0.9654 / 0.8031 / 0.9350 |

Within each slash-separated sequence, the order is CARE Depression / CARE
Parkinson / WSM Depression / WSM Parkinson. The largest class-balance concern
is CARE Parkinson patient recall: it ranges from 0.2941 to 0.5000 across the
arms. Arm F's aggregate is driven especially by WSM Parkinson (0.9304), while
CARE Parkinson remains lower (0.6739; patient recall 0.3824).

The selected TEST aggregate Macro Recall values, retained only as descriptive
statistics, were A 0.7377, B 0.7364, C 0.7363, D 0.7197, E 0.7177, and F
0.7427. They were not used for selection, stopping, or interpretation of the
pilot winner.

## Historical context and limitations

The evidence records the previously reported fold-level context: EmotionMSP
Pooled MLP hidden-16 0.7461, EmotionMSP MD-Former 0.7650, EmotionMSP+TRILLsson
Pooled MLP 0.8293, and historical five-fold mean approximately 0.7952. These
numbers are not recomputed or altered by this pilot. The present 0.8023 is a
single fold-0 validation result under the DEEP-008 isolated protocol, so it
cannot be compared as a replacement for the five-fold mean without running the
approved follow-up protocol.

This pilot did not evaluate MantisV2 feasibility, other folds, threshold
selection, preprocessing changes, or a larger hyperparameter search. No claim
of generalization or guaranteed 0.82--0.85 performance is made.
