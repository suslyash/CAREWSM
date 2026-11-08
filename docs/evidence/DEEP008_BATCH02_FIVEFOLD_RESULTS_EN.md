# DEEP-008 Batch 02: Five-Fold Results

## 1. Scope and decision order

Batch 02 was completed under the approved five-fold protocol. The ten H1 runs were reused, and the five remaining H2 runs were executed sequentially with `--resume-h2`: the fold-0 recovery configuration followed by the original H2 folds 1–4. No H1 run was rerun.

Validation checkpoint selection and the preregistered acceptance decisions were frozen before reading TEST results. TEST metrics below are descriptive only and did not affect selection, tuning, continuation, or rejection.

The original failed H2 fold-0 attempt remains preserved and is excluded from the valid-run analysis:

`H2_corpus_heads_fold0_2026-11-08_18-04_carewsm_gated_corpus_heads_42a41a9d`, MLflow run `eed7325995504d32b6f85e00d85d426b`.

## 2. Valid run ledger

All 15 valid runs had a completed launcher state, a completed training log, a summary, a selected non-`last.pt` checkpoint, and a FINISHED MLflow run in the dedicated Batch 02 experiment.

| Hypothesis | Fold | Runtime directory | MLflow run ID | Selected epoch | Selected checkpoint |
|---|---:|---|---|---:|---|
| H1 Depression | 0 | `H1_depression_fold0_2026-11-08_17-44_carewsm_gated_single_task_aa64ac7a` | `fb36340edf0b405986bd56b4ec3f1b64` | 9 | `epoch=9_val_mean_corpora_depression_macro_recall=0.8003.pt` |
| H1 Depression | 1 | `H1_depression_fold1_2026-11-08_17-45_carewsm_gated_single_task_c13fe6ca` | `a7e7e17a4a62428cb756316fb7a61088` | 2 | `epoch=2_val_mean_corpora_depression_macro_recall=0.8243.pt` |
| H1 Depression | 2 | `H1_depression_fold2_2026-11-08_17-47_carewsm_gated_single_task_d70070ee` | `2e90e2998be94eb893b3cf60a9e33dd6` | 6 | `epoch=6_val_mean_corpora_depression_macro_recall=0.7187.pt` |
| H1 Depression | 3 | `H1_depression_fold3_2026-11-08_17-48_carewsm_gated_single_task_4bb85004` | `dd5cce3e4b694cad97f20de0fc8f6888` | 6 | `epoch=6_val_mean_corpora_depression_macro_recall=0.8080.pt` |
| H1 Depression | 4 | `H1_depression_fold4_2026-11-08_17-50_carewsm_gated_single_task_83cb33e7` | `143dd63b49e445fd92fc5ba8361c51c6` | 7 | `epoch=7_val_mean_corpora_depression_macro_recall=0.8392.pt` |
| H1 Parkinson | 0 | `H1_parkinson_fold0_2026-11-08_17-52_carewsm_gated_single_task_0e112034` | `322643a624974f00a0301a2143967e31` | 14 | `epoch=14_val_mean_corpora_parkinson_macro_recall=0.8810.pt` |
| H1 Parkinson | 1 | `H1_parkinson_fold1_2026-11-08_17-54_carewsm_gated_single_task_51d3f890` | `055188fcdfce41289019c1027090771c` | 15 | `epoch=15_val_mean_corpora_parkinson_macro_recall=0.8864.pt` |
| H1 Parkinson | 2 | `H1_parkinson_fold2_2026-11-08_17-56_carewsm_gated_single_task_43d995f7` | `623b12fcfaa6478a879e56bcec6252eb` | 9 | `epoch=9_val_mean_corpora_parkinson_macro_recall=0.8648.pt` |
| H1 Parkinson | 3 | `H1_parkinson_fold3_2026-11-08_17-58_carewsm_gated_single_task_2bb0b2d9` | `b87ce3b1959e4bfabac446490ca1db71` | 10 | `epoch=10_val_mean_corpora_parkinson_macro_recall=0.8129.pt` |
| H1 Parkinson | 4 | `H1_parkinson_fold4_2026-11-08_18-00_carewsm_gated_single_task_fa20e550` | `6d729265b30e49d0acc728bac589358e` | 22 | `epoch=22_val_mean_corpora_parkinson_macro_recall=0.9049.pt` |
| H2 Corpus heads | 0 | `H2_corpus_heads_fold0_recovery_2026-11-08_18-27_carewsm_gated_corpus_heads_ebb93484` | `447b733f80f648288c433478dddaf8f0` | 9 | `epoch=9_val_mean_corpora_tasks_macro_recall=0.8117.pt` |
| H2 Corpus heads | 1 | `H2_corpus_heads_fold1_2026-11-08_18-29_carewsm_gated_corpus_heads_73f614e4` | `943147ca998f4799bcfb5bb747e05d55` | 16 | `epoch=16_val_mean_corpora_tasks_macro_recall=0.8034.pt` |
| H2 Corpus heads | 2 | `H2_corpus_heads_fold2_2026-11-08_18-32_carewsm_gated_corpus_heads_11b4a31c` | `38432ea423d44d7080f17f8366212e9c` | 22 | `epoch=22_val_mean_corpora_tasks_macro_recall=0.8199.pt` |
| H2 Corpus heads | 3 | `H2_corpus_heads_fold3_2026-11-08_18-35_carewsm_gated_corpus_heads_2497e80a` | `9029dcbe0064499da65ea33b9a376707` | 11 | `epoch=11_val_mean_corpora_tasks_macro_recall=0.8056.pt` |
| H2 Corpus heads | 4 | `H2_corpus_heads_fold4_2026-11-08_18-37_carewsm_gated_corpus_heads_e1b1d771` | `825f9ba86bf64d5aadc5583fd16bfb12` | 16 | `epoch=16_val_mean_corpora_tasks_macro_recall=0.8782.pt` |

## 3. Validation results and frozen decisions

For H1, each fold combines the Depression component from its independently VAL-selected Depression run with the Parkinson component from its independently VAL-selected Parkinson run. The four components are CARE Depression, CARE Parkinson, WSM Depression, and WSM Parkinson.

| Fold | Weighted-F control | H1 paired UAR | H1 − control | H2 UAR | H2 − control |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.832100 | 0.840650 | +0.008550 | 0.811675 | −0.020425 |
| 1 | 0.840000 | 0.855375 | +0.015375 | 0.803325 | −0.036675 |
| 2 | 0.797900 | 0.791775 | −0.006125 | 0.819925 | +0.022025 |
| 3 | 0.801000 | 0.810475 | +0.009475 | 0.805625 | +0.004625 |
| 4 | 0.864300 | 0.872075 | +0.007775 | 0.878175 | +0.013875 |
| **Mean ± sample SD** | **0.827060 ± 0.027880** | **0.834070 ± 0.032738** | **+0.007010** | **0.823745 ± 0.031097** | **−0.003315** |

| Hypothesis | Mean improvement | Component degradation > 0.020 | Paired fold wins | Decision |
|---|---:|---|---:|---|
| H1 independent task optimization | +0.007010 | No | 4/5 | **Rejected**: below the required +0.010 |
| H2 corpus-specific heads | −0.003315 | Yes: CARE Parkinson (−0.06490) | 3/5 | **Rejected**: overall improvement and component-preservation rules both fail |

The historical DB-MTL reference is `0.8317 ± 0.0371`. H1 is descriptively `+0.00237` above its mean; H2 is `−0.007955` below it. Neither hypothesis satisfies the preregistered overall improvement criterion.

### Validation Patient and Healthy Recall

Values are five-fold mean ± sample SD for the fixed VAL-selected checkpoints.

| Hypothesis | Corpus/task | Patient Recall | Healthy Recall | Macro Recall |
|---|---|---:|---:|---:|
| H1 | CARE Depression | 0.771940 ± 0.124294 | 0.919900 ± 0.032965 | 0.845940 ± 0.060564 |
| H1 | CARE Parkinson | 0.767360 ± 0.182997 | 0.938660 ± 0.036254 | 0.853020 ± 0.084540 |
| H1 | WSM Depression | 0.740520 ± 0.081260 | 0.760160 ± 0.079346 | 0.750340 ± 0.048334 |
| H1 | WSM Parkinson | 0.920300 ± 0.060507 | 0.853720 ± 0.071179 | 0.886980 ± 0.031983 |
| H2 | CARE Depression | 0.823080 ± 0.099195 | 0.941220 ± 0.052907 | 0.882140 ± 0.041519 |
| H2 | CARE Parkinson | 0.556640 ± 0.225760 | 0.958980 ± 0.033835 | 0.757800 ± 0.098735 |
| H2 | WSM Depression | 0.730320 ± 0.096223 | 0.766980 ± 0.072787 | 0.748640 ± 0.043522 |
| H2 | WSM Parkinson | 0.930060 ± 0.023312 | 0.882740 ± 0.033962 | 0.906400 ± 0.022304 |

Relative to the historical weighted-F VAL component means (CARE Depression 0.82642, CARE Parkinson 0.82270, WSM Depression 0.75624, WSM Parkinson 0.90276), H1 changes were +0.01952, +0.03032, −0.00590, and −0.01578, respectively. H2 changes were +0.05572, **−0.06490**, −0.00760, and +0.00364. H1 satisfies the component-preservation rule but fails the required +0.010 overall VAL gain. **H2 fails the component-preservation rule**: CARE Parkinson UAR declines by 0.06490, well beyond the permitted 0.020, and it also fails the overall VAL gain rule. On CARE Parkinson, H2 Patient Recall is 0.55664 versus weighted-F 0.69884, an absolute decline of 0.14220.

## 4. Descriptive TEST results

The following values come from the recorded TEST rows at exactly the VAL-selected epochs listed in the run ledger. No checkpoint was selected or changed using TEST.

| Fold | H1 TEST UAR | H2 TEST UAR |
|---:|---:|---:|
| 0 | 0.790400 | 0.747600 |
| 1 | 0.738075 | 0.716175 |
| 2 | 0.788750 | 0.743450 |
| 3 | 0.791475 | 0.786275 |
| 4 | 0.804600 | 0.768975 |
| **Mean ± sample SD** | **0.782660 ± 0.025709** | **0.752495 ± 0.026635** |

| Reference | TEST UAR | Difference from 0.8500 target |
|---|---:|---:|
| H1 | 0.782660 | −0.067340 |
| H2 | 0.752495 | −0.097505 |
| Historical weighted-F | ≈0.7643 | ≈−0.0857 |
| Historical DB-MTL | 0.784105 | −0.065895 |

Descriptively, H1 is +0.01836 above weighted-F and −0.001445 below DB-MTL. H2 is −0.011805 below weighted-F and −0.031610 below DB-MTL. These TEST differences do not alter the VAL-based rejection decisions.

### TEST Patient and Healthy Recall

Values are five-fold mean ± sample SD at the fixed VAL-selected epochs.

| Hypothesis | Corpus/task | Patient Recall | Healthy Recall | Macro Recall |
|---|---|---:|---:|---:|
| H1 | CARE Depression | 0.585560 ± 0.072326 | 0.937560 ± 0.048464 | 0.761560 ± 0.057577 |
| H1 | CARE Parkinson | 0.694100 ± 0.105235 | 0.945780 ± 0.024613 | 0.819940 ± 0.046981 |
| H1 | WSM Depression | 0.794620 ± 0.033175 | 0.755680 ± 0.052106 | 0.775140 ± 0.032121 |
| H1 | WSM Parkinson | 0.762380 ± 0.057918 | 0.785660 ± 0.067226 | 0.774000 ± 0.032382 |
| H2 | CARE Depression | 0.628880 ± 0.062689 | 0.968340 ± 0.027553 | 0.798600 ± 0.021762 |
| H2 | CARE Parkinson | 0.364720 ± 0.144977 | 0.974660 ± 0.004613 | 0.669680 ± 0.073881 |
| H2 | WSM Depression | 0.737320 ± 0.074947 | 0.744720 ± 0.039798 | 0.741020 ± 0.036125 |
| H2 | WSM Parkinson | 0.790960 ± 0.043627 | 0.810380 ± 0.051538 | 0.800680 ± 0.036259 |

## 5. Completion evidence and provenance

- Original ten completed H1 runs used source commit `78e3eb3e50e09119246dfa7849c0f0b23745f320`; five valid H2 recovery runs used source commit `607d6afc46c6a4dc631b7fae69a5d3aa8df00c0f`, which adds only the H2 mixed-precision dtype compatibility fix and recovery launcher/config. No H1 run was retrained.
- H2-only preflight passed before execution.
- The launcher completed exactly five recovery runs in deterministic order and reported `selected=5`.
- The ten H1 runs were reused; no H1 process was launched.
- The original failed H2 fold-0 runtime and MLflow history were preserved and remain explicitly invalid.
- No TEST sample inspection, new inference, feature extraction, audio access, or CUDA run beyond the five authorized H2 recoveries was performed.

## 6. Scientific conclusion

Both hypotheses are rejected under the frozen VAL acceptance contract. H1 produced a modest mean VAL increase of `+0.007010`, four paired fold wins, and no component degradation beyond `0.020`, but it did not reach the required `+0.010`. H2 produced three paired wins but failed **both** required gates: its mean VAL UAR was `0.003315` below weighted-F, and CARE Parkinson VAL UAR deteriorated by `0.06490` (Patient Recall by `0.14220`).

The recorded TEST results are descriptive follow-up evidence only. H1 averaged `0.782660 ± 0.025709` and H2 averaged `0.752495 ± 0.026635`, both below the long-term `0.8500` target. No additional experiments are authorized by this batch.
