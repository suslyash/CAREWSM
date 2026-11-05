# DEEP-004 — Native Turn-Aligned Feature Fusion Evidence

## Outcome

DEEP-004 implements native turn-aligned concatenation in the existing CAREWSM Dataset/DataModule and prepares exactly 110 new configurations. No training, CUDA experiment, feature extraction, or source-artifact modification was performed.

The implementation preserves the existing string-valued singleton interface and adds ordered list-valued selections of two to four families. Fused inputs are concatenated along the channel dimension only; no pooling, interpolation, resampling, timestamp fabrication, or padding is performed at input load time.

## Native turn alignment contract

For every selected family, the Dataset loads the native two-dimensional array and its original turn identifiers:

- `emotionmsp_embeds`: NPZ `embed`, IDs from `turn_ids`, dimension 1,024;
- `emotionmsp_trajectories`: CSV arousal/dominance/valence, IDs from `turn_id`, dimension 3;
- `trillsson4_1`: NPZ `mean`, IDs from `turn_ids`, dimension 1,024;
- `wav2vec_xlsr300m`: NPZ `mean`, IDs from `turn_ids`, dimension 1,024.

The selected list order is the concatenation order. Every component must have a non-empty, one-dimensional, unique ID sequence with one ID per native row. The sequences must match exactly, including order; a same-length sequence with different IDs is rejected. Missing, duplicate, malformed, or absent IDs are rejected. A `VALID` manifest status does not override a failed runtime alignment check.

For a valid selection, the result is:

```text
[T, D_1] + [T, D_2] + ... -> [T, sum(D_i)]
```

The original `T` is retained. CARE remains variable-T and WSM remains single-turn for the inspected live artifacts. The sequence mask is generated from the resulting native `T`, and observed target masks, labels, corpus/profile identities, source paths, dimensions, selected family order, and turn-alignment metadata are preserved.

Matching turn IDs establish turn-level correspondence only. They do not establish synchronized one-second windows or a common frame rate.

## Configuration matrix

The 11 approved subsets use the canonical family order below. Dimensions are the exact concatenated channel dimensions.

| Subset | Input dimension | Configurations |
|---|---:|---:|
| `emotionmsp_embeds__emotionmsp_trajectories` | 1,027 | 10 |
| `emotionmsp_embeds__trillsson4_1` | 2,048 | 10 |
| `emotionmsp_embeds__wav2vec_xlsr300m` | 2,048 | 10 |
| `emotionmsp_trajectories__trillsson4_1` | 1,027 | 10 |
| `emotionmsp_trajectories__wav2vec_xlsr300m` | 1,027 | 10 |
| `trillsson4_1__wav2vec_xlsr300m` | 2,048 | 10 |
| `emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1` | 2,051 | 10 |
| `emotionmsp_embeds__emotionmsp_trajectories__wav2vec_xlsr300m` | 2,051 | 10 |
| `emotionmsp_embeds__trillsson4_1__wav2vec_xlsr300m` | 3,072 | 10 |
| `emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m` | 2,051 | 10 |
| `emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m` | 3,075 | 10 |
| **Total** | — | **110** |

Each subset has pooled MLP and MD-Former configurations for folds 0–4. New YAMLs live only under `configs/audio_deep_features/native_turn_fusion/`. Existing historical 110 configs and DEEP-002 native-turn 40 configs were not modified.

New configurations use:

- `val_mean_corpora_tasks/macro_recall` for checkpointing and early stopping;
- `logs/audio_deep_features` for console/file logs, checkpoints, and snapshots;
- `sqlite:///logs/audio_deep_features/mlflow.db` for MLflow;
- unique `native_turn_fusion/<subset>/<model>/fold` experiment identities;
- the existing model, optimizer, seed, batch size, epoch budget, metrics, split, and evaluation contracts.

## Synthetic test evidence

The extended `tests/test_refactor_audio_data.py` suite passed **15 tests**. It covers:

- unchanged singleton string behavior;
- exact two-, three-, and four-family concatenation order and dimensions;
- variable-T inputs and sequence masks;
- strict ordered turn-ID comparison;
- same-T mismatches;
- missing, duplicated, and malformed IDs;
- complete-case filtering for every selected family;
- invalid empty, singleton-list, duplicate, and unknown selections;
- profile/fold/corpus and observed-label preservation;
- pooled MLP and MD-Former forward/backward on fused synthetic batches.

The full repository test suite passed **30 tests**, with five dependency/model warnings and no failures.

## Real-data read-only verification

The live CARE/WSM registry and availability manifest were read without modification. DEEP-001 reported 10,461 valid rows for the four-family complete-case intersection; the 11 fused selections retain that same intersection. The new DataModule produced the following counts for every one of the 11 subsets and every fold:

| Fold | Joint train | CARE val | WSM val | CARE test | WSM test |
|---:|---:|---:|---:|---:|---:|
| 0 | 6,998 | 352 | 1,429 | 318 | 1,364 |
| 1 | 7,027 | 345 | 1,407 | 318 | 1,364 |
| 2 | 7,045 | 335 | 1,399 | 318 | 1,364 |
| 3 | 7,020 | 331 | 1,428 | 318 | 1,364 |
| 4 | 7,026 | 350 | 1,403 | 318 | 1,364 |

Representative live artifacts were loaded for a pair, a triple, and the four-family selection. CARE variable-T examples produced `[2, 1027]`, `[2, 2051]`, and `[2, 3075]`; WSM single-turn examples produced `[1, 1027]`, `[1, 2051]`, and `[1, 3075]`. The observed IDs were ordered and aligned, with representative sequences `['1', '2']` for CARE and `['1']` for WSM. Values were finite.

The prior four-family audit found the same ordered turn-ID correspondence across the inspected valid artifacts and confirmed CARE `T=1..55` variability versus WSM `T=1`. This remains turn-level correspondence rather than a common timestamp axis.

## Configuration and code verification

- Exactly 110 new YAML files and 11 subsets were structurally validated.
- All 110 new configs have correct ordered family lists, summed dimensions, unique identities, canonical monitor, and filesystem/MLflow roots.
- Native Chimera validation passed for all **260** checked configs: 110 new fusion, 110 historical, and 40 DEEP-002 native-turn configs.
- Python compilation passed for the modified Dataset, DataModule, and test module.
- `git diff --check` passed.
- No training command, GPU benchmark, or experiment rerun was executed.

## Remaining risks

- Turn-ID equality is not synchronized one-second temporal alignment. A future temporal-native scientific comparison still requires an approved timestamp/window contract.
- Runtime alignment validation occurs when a fused sample is loaded; invalid artifacts fail explicitly rather than being silently accepted or filtered by `VALID` alone.
- No performance, fusion benefit, or model superiority claim is made by this preparation task.
- The unrelated local `run-mlflow.sh` modification was preserved and is outside this task.
