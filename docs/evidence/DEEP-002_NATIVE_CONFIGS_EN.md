# DEEP-002 — Native-Turn Baseline Configuration Evidence

Date: 2026-11-05
Base: `519cfe1f9f09cf207c61d3bc0d83ae7cb9aada11`
Branch: `codex/deep-002-native-configs`

## Outcome

Prepared exactly 40 bounded baseline YAML configurations for the approved native-turn rerun:

- four feature families;
- pooled MLP and MD-Former;
- five canonical folds per family/model;
- no training or GPU experiment launched.

The configurations use the existing singular string-valued `feature_family` interface. They preserve the native artifact representation `[T,D]`: variable `T` in CARE and `T=1` in WSM. No input-side pooling, truncation, normalization, feature concatenation, or NPZ `std` channel was added.

## Matrix and files

| Family | Input dimension | MLP files | MD-Former files | Total |
|---|---:|---:|---:|---:|
| `emotionmsp_embeds` | 1,024 | 5 | 5 | 10 |
| `emotionmsp_trajectories` | 3 | 5 | 5 | 10 |
| `trillsson4_1` | 1,024 | 5 | 5 | 10 |
| `wav2vec_xlsr300m` | 1,024 | 5 | 5 | 10 |
| **Total** | — | **20** | **20** | **40** |

Files are under `configs/audio_deep_features/native_turns/<family>/`, named `mlp_fold0.yaml` through `mlp_fold4.yaml` and `mdformer_fold0.yaml` through `mdformer_fold4.yaml`.

Each file retains the current batch size, optimizer, seed, mixed precision, 50-epoch budget, model parameters, masked multitask loss, six-metric configuration, joint CARE+WSM fold construction, and separate `val_care`, `val_wsm`, `test_care`, and `test_wsm` populations.

## Monitoring and logging

All 40 configurations use the current validation-only monitor `dev_mean_macro_recall` for both checkpointing and early stopping. TEST populations are not used for selection.

Each configuration has a unique experiment/run identity containing `native_turns`, the family, model, and fold. Console/file logs, checkpoints, and snapshots use `logs/audio_deep_features`. MLflow uses:

```text
sqlite:///logs/audio_deep_features/mlflow.db
```

The historical `joint/average_macro_f1` monitor was not reintroduced; it remains historical evidence only.

## Validation

- Exactly 40 YAML files found.
- Exactly 10 files per family, with five MLP and five MD-Former files.
- All 40 identities are unique.
- Every `feature_family` is a string and matches its directory.
- Fold values, model names, native dimensions, monitor names, evaluation populations, logging paths, MLflow URI, and 50-epoch budget passed structural checks.
- Representative native Chimera validation passed for `emotionmsp_embeds/mlp_fold0.yaml`.
- Full native Chimera validation was run for all 40 YAMLs; no training was launched.
- The existing read-only check for all 110 original configurations was run; original configuration files were not modified.
- `git diff --check` passed.

## Native input contract

These are single-family baselines only. The active loader continues to select:

- `embed` for `emotionmsp_embeds`;
- `mean` for `trillsson4_1`;
- `mean` for `wav2vec_xlsr300m`.

The trajectory CSV remains `[T,3]`. The NPZ families remain `[T,1024]` under the selected key. The existing pooled MLP performs only its historical internal aggregation; the YAMLs do not alter that architecture. MD-Former receives the same native sequence contract and does not receive fabricated temporal tokens.

No multi-feature list is used. Native-turn concatenation and a separate mean+std-over-turns comparison remain future, separately authorized work.

## Blockers and risks

The DEEP-001 audit found that CARE has variable turn counts while WSM is single-turn for these artifacts. The native rerun therefore compares the existing model behavior under a corpus-dependent `T` distribution; it must not be described as a pooled-static experiment. The registry’s declared mean+std dimensionality for TRILLsson/XLS-R is intentionally not used here because the current loader contract selects `mean` only.

No scientific performance claim is made by this configuration-preparation task. Independent manager review is required before any of the 40 configurations are run.
