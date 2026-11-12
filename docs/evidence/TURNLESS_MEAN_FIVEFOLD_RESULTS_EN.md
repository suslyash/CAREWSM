# TURNLESS MEAN Five-Fold Results

Source commit: 8a91a06e52d4710f54503b824f76a969cf73942b; seed 5; 20/20 runs completed successfully.

Mean pooling was verified on real artifacts: a CARE multi-turn sample (3 turns) and a WSM single-turn sample (1 turn) both had exact arithmetic agreement (max_abs_error=0.0), with output shape [1, 2048].

## Table A — Validation Results

| Experiment | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± SD | Delta vs Weighted-F | Paired wins | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 gated 64 | 0.8332 | 0.8396 | 0.7992 | 0.7870 | 0.8637 | 0.8245 ± 0.0312 | -0.0025 | 2/5 | REJECTED |
| E2 pooled MLP 64 | 0.8311 | 0.8119 | 0.7161 | 0.8091 | 0.8414 | 0.8019 ± 0.0498 | -0.0251 | 1/5 | REJECTED |
| E3 gated 32 | 0.8590 | 0.8343 | 0.7972 | 0.8026 | 0.8724 | 0.8331 ± 0.0333 | +0.0060 | 3/5 | REJECTED |
| E4 pooled MLP 32 | 0.7977 | 0.7994 | 0.7200 | 0.8017 | 0.8311 | 0.7900 ± 0.0415 | -0.0371 | 1/5 | REJECTED |
| Weighted-F historical control | 0.8321 | 0.8400 | 0.7979 | 0.8010 | 0.8643 | 0.8271 ± 0.0279 | — | — | CONTROL |

Acceptance is validation-only: required improvement was +0.010 mean UAR, with no component mean deterioration over 0.020.

## Table B — Descriptive TEST Results — not used for model selection

| Experiment | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± SD | Delta vs Weighted-F |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 gated 64 | 0.7777 | 0.7285 | 0.7614 | 0.7692 | 0.7890 | 0.7652 ± 0.0229 | +0.0009 |
| E2 pooled MLP 64 | 0.7595 | 0.7566 | 0.7262 | 0.7629 | 0.7771 | 0.7565 ± 0.0187 | -0.0078 |
| E3 gated 32 | 0.7681 | 0.7549 | 0.7215 | 0.7701 | 0.7719 | 0.7596 ± 0.0162 | -0.0047 |
| E4 pooled MLP 32 | 0.7437 | 0.7409 | 0.6920 | 0.7755 | 0.7695 | 0.7443 ± 0.0330 | -0.0200 |
| Weighted-F historical control | 0.7547 | 0.7325 | 0.7609 | 0.7846 | 0.7888 | 0.7643 ± 0.0231 | — |

TEST is previously observed descriptive evidence, not a pristine independent holdout.

## Table C — Five-fold mean Macro Recall by corpus/task

| Experiment | CARE Depression VAL/TEST | CARE Parkinson VAL/TEST | WSM Depression VAL/TEST | WSM Parkinson VAL/TEST |
|---|---:|---:|---:|---:|
| E1 gated 64 | 0.8296/0.7512 | 0.8124/0.7711 | 0.7609/0.7516 | 0.8953/0.7866 |
| E2 pooled MLP 64 | 0.8190/0.7573 | 0.7979/0.7703 | 0.7302/0.7556 | 0.8606/0.7427 |
| E3 gated 32 | 0.8454/0.7631 | 0.8364/0.7615 | 0.7464/0.7640 | 0.9043/0.7408 |
| E4 pooled MLP 32 | 0.7962/0.7243 | 0.7659/0.7680 | 0.7331/0.7378 | 0.8647/0.7472 |

## Table D — Patient/Healthy Recall

| Experiment | CARE Parkinson VAL patient/healthy | CARE Parkinson TEST patient/healthy |
|---|---:|---:|
| E1 gated 64 | 0.6852/0.9395 | 0.5824/0.9599 |
| E2 pooled MLP 64 | 0.7060/0.8898 | 0.6294/0.9113 |
| E3 gated 32 | 0.7186/0.9542 | 0.5764/0.9465 |
| E4 pooled MLP 32 | 0.6664/0.8655 | 0.6353/0.9007 |

The full six-metric blocks for every corpus/task/fold are in turnless_mean_results_by_fold.csv: Macro F1, Weighted F1, Macro Recall, Weighted Recall, Patient Recall, and Healthy Recall. Confusion counts were not fabricated because the existing summary callback does not persist TP/TN/FP/FN.

## Table E — Reproducibility

All 20 run identities, actual MLflow run IDs, selected epochs, selected checkpoints, commit, and status are in turnless_mean_run_ledger.csv. Every row is COMPLETED; selected checkpoints are VAL-selected and TEST metrics are read from the same selected epoch.

- Runtime root: logs/audio_representation_pilot/turnless_mean/
- MLflow database: logs/audio_representation_pilot/turnless_mean/mlflow.db
- Configurations: configs/audio_representation_pilot/turnless_mean/
- Pooling audit: logs/audio_representation_pilot/turnless_mean/data_verification.json
- Fold results: turnless_mean_results_by_fold.csv
- Summary: turnless_mean_results_summary.csv
- Ledger: turnless_mean_run_ledger.csv

## Scientific interpretation

Early mean pooling does not establish a validation improvement over native-turn Weighted-F. Its effect is primarily testable on CARE because valid WSM samples contain one turn; WSM differences are optimization/generalization effects.

Learned feature-family gates remain beneficial relative to the pooled MLP: E1 exceeds E2 by 0.0226 mean VAL UAR, and E3 exceeds E4 by 0.0431. Reducing hidden dimension helps the gated model (E3 > E1 by 0.0086) but hurts the pooled MLP (E4 < E2 by 0.0119). The best new configuration, E3, is still below the +0.010 acceptance threshold and is therefore rejected. No statistical significance or clinical superiority is claimed. A further research step requires a new manager-approved, validation-controlled task.
### All 20 run identities

| Experiment | Fold | MLflow run ID | Epoch | Status |
|---|---:|---|---:|---|
| E1_gated64 | 0 | 2a470983fbf34cd283b2cdd5cd5e68c0 | 8 | COMPLETED |
| E1_gated64 | 1 | d2ee7b73ef4a49f082946a888bc9286e | 3 | COMPLETED |
| E1_gated64 | 2 | 0217d3a87b7a4387b8faed5ca3066846 | 11 | COMPLETED |
| E1_gated64 | 3 | e5f1d41cec4b48908f60e3101f3f6824 | 18 | COMPLETED |
| E1_gated64 | 4 | c9cd45d0c0fb4e7cb1bdb580e8af97ce | 7 | COMPLETED |
| E2_mlp64 | 0 | 544a8357d52841c49c619730adaddeed | 12 | COMPLETED |
| E2_mlp64 | 1 | 8a98646a24f246afb1a309699dd72543 | 12 | COMPLETED |
| E2_mlp64 | 2 | 302d460674ab4270b3b2dac135c2fd5d | 5 | COMPLETED |
| E2_mlp64 | 3 | ae6d957055ed40a5870fd11ed8ac7b66 | 15 | COMPLETED |
| E2_mlp64 | 4 | 56bb22ee50144e9e9765505cbb09fc28 | 19 | COMPLETED |
| E3_gated32 | 0 | 7b9ee7c1d0564a088c627d3f9a1c627c | 15 | COMPLETED |
| E3_gated32 | 1 | 3abd7055d53841f38454835fde298f76 | 17 | COMPLETED |
| E3_gated32 | 2 | 0579cd3e9d934391ba3b99f6de6f2eca | 30 | COMPLETED |
| E3_gated32 | 3 | 707208da71ca4b63a8dc2a78286dc95d | 15 | COMPLETED |
| E3_gated32 | 4 | 547754d2b9c748259b7ba57f5d214321 | 18 | COMPLETED |
| E4_mlp32 | 0 | c6f3a4b2b64849e08fe29d710f38d5d5 | 8 | COMPLETED |
| E4_mlp32 | 1 | 86e1545be618485fa2bce0f5c2a68ba1 | 13 | COMPLETED |
| E4_mlp32 | 2 | e0b2331f1c4b4842a48a1ee925266689 | 5 | COMPLETED |
| E4_mlp32 | 3 | 798ecd034c274548a1c37fb917219656 | 20 | COMPLETED |
| E4_mlp32 | 4 | d0c9a005ed4e4b149fecb7e10f98bac9 | 12 | COMPLETED |
