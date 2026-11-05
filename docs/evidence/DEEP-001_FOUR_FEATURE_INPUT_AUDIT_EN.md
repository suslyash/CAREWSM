# DEEP-001 — Four-Family Input Shape and Fusion Readiness Audit

Date: 2026-11-05  
Base: `c7e64c428ebf9b8a81571c7067cd7eb9fdf9d271`  
Branch: `codex/deep-001`

## Outcome

This is a read-only audit of the real CARE and WSM artifacts for:

- `emotionmsp_embeds`;
- `emotionmsp_trajectories`;
- `trillsson4_1`;
- `wav2vec_xlsr300m`.

No feature extraction, training, configuration generation, model change, or data modification was performed. The existing manifest contains 10,464 rows. Four-family artifact validity is identical for these families: 10,461 valid rows and three WSM missing rows. The three missing rows are excluded by complete-case filtering; no labels or samples were rewritten.

## Actual artifact findings

The registry declares the NPZ families as turn-level embeddings and the trajectory family as turn-level data. Direct inspection of valid artifacts found these native structures:

| Family | Artifact/key or columns | Observed native shape | Dtype | Temporal evidence |
|---|---|---|---|---|
| `emotionmsp_embeds` | `embed`, `turn_ids` | `[T, 1024]`, `turn_ids [T]` | `float32`, `int64` | Turn-indexed; CARE `T=1..55` (35 distinct T), WSM `T=1` |
| `emotionmsp_trajectories` | `arousal`, `dominance`, `valence`; `turn_id`, `turn_start_sec`, `turn_end_sec` | `[T, 3]` | parsed `float32` | Explicit turn timestamps; CARE `T=1..55` (35 distinct T), WSM `T=1` |
| `trillsson4_1` | `mean`, `std`, `turn_ids` | `[T, 1024]` per selected `mean`; both arrays are present | `float32`, `int64` | Turn-indexed; CARE `T=1..55` (35 distinct T), WSM `T=1` |
| `wav2vec_xlsr300m` | `mean`, `std`, `turn_ids` | `[T, 1024]` per selected `mean`; both arrays are present | `float32`, `int64` | Turn-indexed; CARE `T=1..55` (35 distinct T), WSM `T=1` |

All inspected valid arrays were non-empty, finite, and readable. NPZ keys were stable: `emotionmsp_embeds` had `embed, turn_ids`; the other two NPZ families had `mean, std, turn_ids`. The registry describes the latter two as 2,048-dimensional mean+std representations, but the active loader selects only `mean`, giving native channel dimension 1,024. That registry/loader distinction must be made explicit before configuration generation.

The four families use the same turn cardinality in each inspected sample. NPZ `turn_ids` agree with one another and with trajectory `turn_id`; trajectory start times are nondecreasing and end times are not before start times. This establishes turn-index correspondence, not a common frame-rate or a common 1.0-second time axis. Only `emotionmsp_trajectories` contains explicit timestamps. The NPZ embeddings do not provide timestamps, so a temporal-native four-family model cannot assume aligned 1.0-second windows.

The observed T pattern is corpus-dependent: CARE contains variable-length turn sequences, while WSM artifacts are all single-turn under the live files. Thus neither a filename nor an existing YAML `input_dim` justifies a native `T=1` claim.

## Four-family validity intersection

The complete-case intersection is the same for every nonempty subset because all four families have the same three missing WSM rows and no additional invalid rows in this selection. The table therefore reports all 15 subsets explicitly; counts are valid rows in the full manifest, not TEST-based selection.

| Subset | Valid rows | Coverage |
|---|---:|---:|
| `emotionmsp_embeds` | 10,461 | 99.97% |
| `emotionmsp_trajectories` | 10,461 | 99.97% |
| `trillsson4_1` | 10,461 | 99.97% |
| `wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_embeds + emotionmsp_trajectories` | 10,461 | 99.97% |
| `emotionmsp_embeds + trillsson4_1` | 10,461 | 99.97% |
| `emotionmsp_embeds + wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_trajectories + trillsson4_1` | 10,461 | 99.97% |
| `emotionmsp_trajectories + wav2vec_xlsr300m` | 10,461 | 99.97% |
| `trillsson4_1 + wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_embeds + emotionmsp_trajectories + trillsson4_1` | 10,461 | 99.97% |
| `emotionmsp_embeds + emotionmsp_trajectories + wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_embeds + trillsson4_1 + wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_trajectories + trillsson4_1 + wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_embeds + emotionmsp_trajectories + trillsson4_1 + wav2vec_xlsr300m` | 10,461 | 99.97% |

## Per-fold and per-corpus retained populations

Because the four validity masks coincide, the following counts apply to every one of the 15 subsets. `Dep obs` and `PD obs` are original observed-label counts after filtering; missing labels remain missing.

| Fold | Population | Expected | Retained | Dep obs | PD obs |
|---:|---|---:|---:|---:|---:|
| 0 | CARE train | 1,361 | 1,361 | 1,361 | 1,361 |
| 0 | WSM train | 5,640 | 5,637 | 3,404 | 2,233 |
| 0 | CARE val | 352 | 352 | 352 | 352 |
| 0 | WSM val | 1,429 | 1,429 | 877 | 552 |
| 1 | CARE train | 1,368 | 1,368 | 1,368 | 1,368 |
| 1 | WSM train | 5,661 | 5,659 | 3,429 | 2,230 |
| 1 | CARE val | 345 | 345 | 345 | 345 |
| 1 | WSM val | 1,408 | 1,407 | 852 | 555 |
| 2 | CARE train | 1,378 | 1,378 | 1,378 | 1,378 |
| 2 | WSM train | 5,670 | 5,667 | 3,434 | 2,233 |
| 2 | CARE val | 335 | 335 | 335 | 335 |
| 2 | WSM val | 1,399 | 1,399 | 847 | 552 |
| 3 | CARE train | 1,382 | 1,382 | 1,382 | 1,382 |
| 3 | WSM train | 5,640 | 5,638 | 3,429 | 2,209 |
| 3 | CARE val | 331 | 331 | 331 | 331 |
| 3 | WSM val | 1,429 | 1,428 | 852 | 576 |
| 4 | CARE train | 1,363 | 1,363 | 1,363 | 1,363 |
| 4 | WSM train | 5,665 | 5,663 | 3,428 | 2,235 |
| 4 | CARE val | 350 | 350 | 350 | 350 |
| 4 | WSM val | 1,404 | 1,403 | 853 | 550 |
| all folds | CARE TEST | 318 | 318 | 318 | 318 |
| all folds | WSM TEST | 1,364 | 1,364 | 827 | 537 |

The joint retained TRAIN/VAL totals are respectively 6,998/1,781, 7,027/1,752, 7,045/1,734, 7,020/1,759, and 7,026/1,753 for folds 0–4. Fixed TEST populations are unchanged. Profile, corpus, split, fold, and sample identities come from the authoritative source split metadata and are preserved.

## Recommended static fusion representation

The minimal defensible representation for the proposed static series is explicit per-family turn pooling followed by ordered feature concatenation:

1. load each family independently;
2. pool its valid `[T, D]` values over the observed turns to one `[1, D]` vector;
3. concatenate the four pooled vectors in configured order along the feature dimension;
4. retain family names, source dimensions, pooling rule, and validity metadata in run metadata.

This is a pooled-static representation, not raw-native fusion. It avoids repeating static vectors along time and does not claim temporal dynamics. A simple arithmetic mean is the minimal contract. The existing registry's `mean+std` dimensions should not be silently mixed into this contract: using per-family mean+std would be a separate representation and must be named and specified explicitly.

The minimal proposed component dimensions are:

| Component | Pooled dimension |
|---|---:|
| `emotionmsp_embeds_mean` | 1,024 |
| `emotionmsp_trajectories_mean` | 3 |
| `trillsson4_1_mean` | 1,024 |
| `wav2vec_xlsr300m_mean` | 1,024 |

The corresponding ordered configuration names and dimensions are:

| Configuration name | Components | Dimension |
|---|---|---:|
| `emotionmsp_embeds` | emotion | 1,024 |
| `emotionmsp_trajectories` | trajectory | 3 |
| `trillsson4_1` | TRILLsson | 1,024 |
| `wav2vec_xlsr300m` | XLS-R | 1,024 |
| `emotionmsp_embeds__emotionmsp_trajectories` | first + second | 1,027 |
| `emotionmsp_embeds__trillsson4_1` | first + third | 2,048 |
| `emotionmsp_embeds__wav2vec_xlsr300m` | first + fourth | 2,048 |
| `emotionmsp_trajectories__trillsson4_1` | second + third | 1,027 |
| `emotionmsp_trajectories__wav2vec_xlsr300m` | second + fourth | 1,027 |
| `trillsson4_1__wav2vec_xlsr300m` | third + fourth | 2,048 |
| `emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1` | first + second + third | 2,051 |
| `emotionmsp_embeds__emotionmsp_trajectories__wav2vec_xlsr300m` | first + second + fourth | 2,051 |
| `emotionmsp_embeds__trillsson4_1__wav2vec_xlsr300m` | first + third + fourth | 3,072 |
| `emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m` | second + third + fourth | 2,051 |
| `emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m` | all four | 3,075 |

These dimensions are valid only for the explicitly proposed mean-pooled contract. The raw-native dimensions are `[T,1024]`, `[T,3]`, `[T,1024]`, and `[T,1024]`; raw concatenation is not a static `[1,D]` input and needs a separate temporal batching/alignment contract.

## Temporal-native comparison and model readiness

No four-family temporal-native model comparison is scientifically justified yet. The trajectory family has timestamps, but the three NPZ families expose turn IDs without timestamps. Matching turn IDs support turn-level correspondence, not the required common 1.0-second/0.5-second axis. Mantis temporalization remains deferred, as required by the project contract.

The existing `carewsm_pooled_mlp` and `carewsm_md_former` can consume the proposed pooled-static representation at the tensor-contract level: both receive `[batch, time, features]`, and the proposed representation is `[batch, 1, D]` with the sequence mask true at that position. MLP then pools the singleton sequence; MD-Former can process the singleton sequence, but this is not evidence of temporal modelling. Their source code does not need a change for a correctly implemented pooled `[1,D]` input, but the current Dataset does not perform the required per-family pooling or multi-family concatenation. Therefore this is model compatibility, not current pipeline readiness.

## Remaining blockers and ambiguities

- The active loader selects only NPZ `mean` for TRILLsson and XLS-R although the registry also declares `std` and an original 2,048-dimensional mean+std representation. The next configuration contract must choose one representation explicitly.
- The current Dataset preserves native `T`; it does not implement the recommended pooled-static fusion. No fusion implementation was added in DEEP-001.
- A mean over turns is a static summary and discards within-video temporal order. It must not be described as temporal modelling.
- The exact policy for trajectory pooling, including treatment of turn duration and any normalization, should be frozen before experiment YAML generation. No global scaler or PCA may be fitted outside each fold's TRAIN population.
- The four-family turn IDs are compatible at the observed turn level, but timestamps are absent from the NPZ families. A temporal-native comparison requires an owner-approved temporalization contract and evidence that all families can be mapped to it.
- All reported counts are validity/coverage evidence only. No TEST result or performance claim was used for feature selection.

## Audit provenance

The inspection used the existing `reports/feature_audit/sample_feature_availability.csv`, the authoritative CARE registry `/media/maxim/Databases/CARE/splits/features.json`, source split CSVs, and direct `numpy`/CSV reads of the live artifacts under `/media/maxim/Databases/CARE/` and `/media/maxim/Databases/WSM_NEW/`. The existing feature-audit evidence and manager experiment report were used as contextual evidence; no new audit framework or dependency was added.
