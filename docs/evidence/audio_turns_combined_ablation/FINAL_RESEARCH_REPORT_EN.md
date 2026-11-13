# CAREWSM EmotionMSP Embeddings + TRILLsson — Static Gated Fusion (64) Ablation

## Scope

Exactly one isolated five-fold configuration was executed: `EmotionMSP Embeddings + TRILLsson — Mean, Static Gated Fusion (64) [AC/64]`. No historical run or completed `audio_turns_combined` slot was rerun.

Critical correction: `pooling: mean` is applied by the Dataset before the historical E3 model. E3 therefore receives a singleton `[B, 1, 2048]` representation; it is not meaningful turn-level temporal modeling. E3 differs from the newer static model mainly through architecture, normalization, gate inputs, dropout placement, and related implementation details.

## Ablation result

Displayed model names are human-readable. Original canonical run IDs and configuration paths are preserved in the machine-readable CSV ledgers.

VAL UAR: **0.821620 ± 0.032782**
Descriptive TEST UAR: **0.754020 ± 0.012444**

| Fold | VAL UAR | TEST UAR | Epoch | Checkpoint | MLflow run ID |
|---:|---:|---:|---:|---|---|
| 0 | 0.8252 | 0.7410 | 5 | `logs/audio_turns_combined_ablation/audio_turns_combined_ablation/EmotionMSP Embeddings + TRILLsson — Mean, Static Gated Fusion (64) [AC/64]_fold0_2026-11-13_11-31_carewsm_static_gated_fusion_38d0ca80/checkpoints/epoch=5_val_mean_corpora_tasks_macro_recall=0.8252.pt` | `06c8a13e34684872b0f068cabdf94b5d` |
| 1 | 0.8433 | 0.7591 | 14 | `logs/audio_turns_combined_ablation/audio_turns_combined_ablation/EmotionMSP Embeddings + TRILLsson — Mean, Static Gated Fusion (64) [AC/64]_fold1_2026-11-13_11-32_carewsm_static_gated_fusion_b96dff6e/checkpoints/epoch=14_val_mean_corpora_tasks_macro_recall=0.8433.pt` | `8656a224e6e44dbf9d1cac7c1750306f` |
| 2 | 0.7796 | 0.7473 | 12 | `logs/audio_turns_combined_ablation/audio_turns_combined_ablation/EmotionMSP Embeddings + TRILLsson — Mean, Static Gated Fusion (64) [AC/64]_fold2_2026-11-13_11-35_carewsm_static_gated_fusion_e3c8fb76/checkpoints/epoch=12_val_mean_corpora_tasks_macro_recall=0.7796.pt` | `e804408119ce4aa1ae15b7e30d7e60e0` |
| 3 | 0.7991 | 0.7497 | 14 | `logs/audio_turns_combined_ablation/audio_turns_combined_ablation/EmotionMSP Embeddings + TRILLsson — Mean, Static Gated Fusion (64) [AC/64]_fold3_2026-11-13_11-37_carewsm_static_gated_fusion_6c5f7f43/checkpoints/epoch=14_val_mean_corpora_tasks_macro_recall=0.7991.pt` | `6662e696b190456dba36704dafeb1847` |
| 4 | 0.8609 | 0.7730 | 13 | `logs/audio_turns_combined_ablation/audio_turns_combined_ablation/EmotionMSP Embeddings + TRILLsson — Mean, Static Gated Fusion (64) [AC/64]_fold4_2026-11-13_11-39_carewsm_static_gated_fusion_80c49367/checkpoints/epoch=13_val_mean_corpora_tasks_macro_recall=0.8609.pt` | `f883d27fed984d0faac4e503f881bf8f` |

## Corpus/task metrics

| Corpus/task | VAL UAR | TEST UAR | VAL Patient/Healthy | TEST Patient/Healthy |
|---|---:|---:|---:|---:|
| CARE Depression | 0.8412 | 0.7702 | 0.7562 / 0.9261 | 0.5794 / 0.9611 |
| CARE Parkinson | 0.8196 | 0.7736 | 0.6949 / 0.9444 | 0.5824 / 0.9648 |
| WSM Depression | 0.7497 | 0.7337 | 0.7196 / 0.7798 | 0.7194 / 0.7480 |
| WSM Parkinson | 0.8759 | 0.7387 | 0.8899 / 0.8619 | 0.6602 / 0.8173 |

## Matched-fold comparisons

| Comparison | VAL delta | TEST delta | Paired VAL differences | Component-safety result |
|---|---:|---:|---|---|
| EmotionMSP Embeddings + TRILLsson — Static Gated Fusion (32) | +0.0063 | -0.0199 | +0.0022, +0.0055, -0.0026, +0.0112, +0.0150 | PASS; max mean component drop +0.0111 |
| EmotionMSP Embeddings + Trajectories + TRILLsson + XLS-R — Static Gated Fusion (32) | -0.0023 | -0.0266 | -0.0164, +0.0034, -0.0014, +0.0185, -0.0158 | FAIL; max mean component drop +0.0233 |
| EmotionMSP Embeddings + Trajectories + TRILLsson + XLS-R — Static Gated Fusion (64), Equal | -0.0092 | -0.0282 | -0.0206, +0.0002, -0.0218, +0.0185, -0.0224 | FAIL; max mean component drop +0.0229 |

## Historical E3 comparison

E3 VAL UAR was `0.8331 ± 0.0333`; EmotionMSP Embeddings + TRILLsson — Static Gated Fusion (64) is `0.8216 ± 0.0328`. The matched fold differences (EmotionMSP Embeddings + TRILLsson / Gated-64 minus E3) are -0.0337, +0.0090, -0.0176, -0.0035, -0.0115, mean `-0.0115`. This is an architectural comparison, not a pure hidden-size comparison.

E3 remains **RETAINED — HIGH-VAL MTL CANDIDATE**. Its historical acceptance status remains REJECTED because its original improvement gate was not met.

## Consolidated research leaderboard

| Candidate | Category | VAL UAR | Descriptive TEST UAR | Status |
|---|---|---:|---:|---|
| EmotionMSP Embeddings + TRILLsson — Mean, Historical Gated Fusion (32) [E3] | Historical joint multitask gated fusion | 0.8331 ± 0.0333 | 0.7573 ± 0.0211 | RETAINED — HIGH-VAL MTL CANDIDATE; historical gate REJECTED |
| Native-turn DB-MTL | Historical joint multitask | 0.8317 ± 0.0371 | 0.784105 ± 0.022360 | Historical reference |
| EmotionMSP Embeddings + Trajectories + TRILLsson + XLS-R — Static Gated Fusion (64), Equal | Current joint multitask static gated fusion | 0.830855 ± 0.040364 | 0.782235 ± 0.022355 | VAL-selected current control |
| Weighted-F native-turn | Historical joint multitask control | 0.8271 ± 0.0279 | 0.7643 ± 0.0231 | Historical control |
| Independent Single-Task H1 | Separately trained Depression/Parkinson models paired by fold | 0.834070 ± 0.032738 | 0.782660 ± 0.025709 | Separate category; historical gate REJECTED |

H1 is not a single jointly trained multitask model and is therefore not ranked as architecture-equivalent to E3, DB-MTL, or Static Gated64 Equal.

## Decision

A claimed substantive improvement requires at least +0.010 mean VAL UAR and no component mean VAL deterioration greater than 0.020. The EmotionMSP Embeddings + TRILLsson / Gated-64 result is evaluated against EmotionMSP Embeddings + TRILLsson / Gated-32 for the hidden-dimension question and against ABCD/64 for the feature-set comparison. TEST is descriptive only; no statistical significance is claimed from five folds.

## Artifacts

- `logs/audio_turns_combined_ablation/reports/all_results_by_fold.csv`
- `logs/audio_turns_combined_ablation/reports/all_results_summary.csv`
- `logs/audio_turns_combined_ablation/reports/run_ledger.csv`
- `logs/audio_turns_combined_ablation/mlflow.db`
- `configs/audio_turns_combined_ablation/ablation/`
- `scripts/audio_turns_combined_ablation/run_all.sh`
- `scripts/audio_turns_combined_ablation/run_worker.py`
