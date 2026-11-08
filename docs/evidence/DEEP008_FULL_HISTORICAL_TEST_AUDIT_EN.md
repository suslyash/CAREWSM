# DEEP-008 Complete Historical TEST Results Audit

## 1. Scope and selection rule

This retrospective audit covers the locally available CAREWSM audio experiment records at source commit `37c18cc6c4688855140576e5e66e0a002f8591ec`. It includes the original 110-fold feature-screening matrix, native and fused representations, gated fusion, multitask optimization methods, weighted-F, task attention, Batch 01, Batch 02, one-fold pilots, and other historical records discovered under `logs/`.

No training, checkpoint loading, inference, audio access, raw TEST inspection, or feature extraction was performed. TEST values were read from existing summaries only.

Every ranked value below uses the four-component arithmetic TEST UAR:

```text
(CARE Depression + CARE Parkinson + WSM Depression + WSM Parkinson) / 4
```

Only the original validation-selected epoch is used. Historical selection monitors are preserved as follows:

- Original 110-run feature-screening matrix: `joint/average_macro_f1` (historical VAL Macro F1 selection). Later deep-feature and DEEP-008 run groups may use `dev_mean_macro_recall` (an alias of the modern four-component VAL Macro Recall monitor), but **that is not an alias of the original feature-screening Macro F1 selector**. Keep the two selection protocols distinct.
- Modern joint records: `val_mean_corpora_tasks/macro_recall`.
- Single-task Batch 02 records: task-specific VAL monitor, paired by fold before H1 scoring.

## 2. Historical coverage and exclusions

The local audit found 267 `summary.txt` records. The committed source manifest covers 205 historical report cells and 200 unique physical MLflow runs; the five-cell Stage-G Equal block reuses the weighted-F physical runs and is deduplicated in rankings.

Additional local records were inventoried as follows:

- Batch 01: 20 completed five-fold runs across four hypotheses.
- Task-specific attention: 5 completed folds.
- Batch 02: 10 completed single-task H1 runs, paired into 5 four-component candidates; 5 valid H2 corpus-head runs.
- The original H2 fold-0 failure is excluded from rankings and retained as invalid provenance.
- Four one-fold DEEP-008 pilot variants (B–E), two one-fold launch/verification records, and one one-fold MantisV2 record are valid or complete artifacts but are not complete five-fold configurations.
- Superseded Stage-G records, duplicate fold-zero records, and diagnostic smoke records remain inventoried but are excluded from the scientific ranking when corrected or complete replacements exist.
- Records without a complete selected-checkpoint/completion chain are non-comparable or incomplete and are not promoted into either leaderboard.

The 200 manifest-backed runs are supported by [DEEP008_HISTORICAL_TEST_SOURCE_MANIFEST.csv](tables/DEEP008_HISTORICAL_TEST_SOURCE_MANIFEST.csv) and [DEEP008_HISTORICAL_TEST_METRICS_PER_FOLD.csv](tables/DEEP008_HISTORICAL_TEST_METRICS_PER_FOLD.csv). Modern run provenance is retained in the Batch 01, task-attention, Batch 02, and Stage-G evidence reports and the local runtime summaries.

## 3. Leaderboard A — top 20 individual fold results

`P/H` is Patient Recall / Healthy Recall in the component order CARE Depression, CARE Parkinson, WSM Depression, WSM Parkinson. Checkpoint values and epochs are the selected VAL identities, never TEST-selected.

| Rank | Experiment/configuration | Fold | VAL UAR / monitor | TEST UAR | Component TEST UARs: CARE-D / CARE-P / WSM-D / WSM-P | P/H tuples in same order | MLflow run ID |
|---:|---|---:|---|---:|---|---|---|
| 1 | Stage-G DB-MTL | 1 | 0.865600 / `val_mean_corpora_tasks/macro_recall` | **0.810075** | 0.7691 / 0.8272 / 0.8085 / 0.8355 | 0.7010/0.8371; 0.7353/0.9190; 0.8507/0.7663; 0.8195/0.8515 | `7add8f3965c347ba88f20facac5a66df` |
| 2 | Batch 02 H1 paired | 4 | 0.872075 / paired task monitors | 0.804600 | 0.7895 / 0.8647 / 0.8134 / 0.7508 | 0.6289/0.9502; 0.7647/0.9648; 0.8179/0.8089; 0.7368/0.7649 | `143dd63b49e445fd92fc5ba8361c51c6` + `6d729265b30e49d0acc728bac589358e` |
| 3 | Batch 01 H4 hidden-32 | 4 | 0.860200 / `val_mean_corpora_tasks/macro_recall` | 0.796175 | 0.8337 / 0.7706 / 0.8048 / 0.7756 | 0.7216/0.9457; 0.5588/0.9824; 0.8149/0.7947; 0.7293/0.8218 | `3debccce04c24b74ae317513f1324719` |
| 4 | Stage-G STCH | 4 | 0.859900 / `val_mean_corpora_tasks/macro_recall` | 0.794800 | 0.7538 / 0.8318 / 0.7805 / 0.8131 | 0.5258/0.9819; 0.7059/0.9577; 0.7134/0.8476; 0.7895/0.8366 | `74399d3faae447adbc66425529052fe4` |
| 5 | Stage-G corrected RA-STCH | 4 | 0.855900 / `val_mean_corpora_tasks/macro_recall` | 0.794650 | 0.7567 / 0.8189 / 0.7899 / 0.8131 | 0.5361/0.9774; 0.6765/0.9613; 0.7343/0.8455; 0.7895/0.8366 | `96df82073f9d43fb81aac99c0eea3b8d` |
| 6 | Batch 01 H3 dropout-0.3 | 4 | 0.853600 / `val_mean_corpora_tasks/macro_recall` | 0.793675 | 0.7496 / 0.8153 / 0.7943 / 0.8155 | 0.5670/0.9321; 0.6765/0.9542; 0.8000/0.7886; 0.7895/0.8416 | `621feea4d02d4079aff12ced7f39f561` |
| 7 | Stage-G corrected RA-STCH | 3 | 0.799400 / `val_mean_corpora_tasks/macro_recall` | 0.793150 | 0.8491 / 0.7418 / 0.7538 / 0.8279 | 0.7526/0.9457; 0.5294/0.9542; 0.8328/0.6748; 0.7820/0.8738 | `cedb5b0f4f564274b346d270f1ffdea5` |
| 8 | Stage-G DB-MTL | 0 | 0.816500 / `val_mean_corpora_tasks/macro_recall` | 0.792875 | 0.7673 / 0.8020 / 0.7905 / 0.8117 | 0.5979/0.9367; 0.7941/0.8099; 0.7313/0.8496; 0.7669/0.8564 | `9577904d62e84d54bc4564f91ec69b7e` |
| 9 | Batch 01 H2 depression-weighted | 4 | 0.861600 / `val_mean_corpora_tasks/macro_recall` | 0.792325 | 0.7770 / 0.7712 / 0.7955 / 0.8256 | 0.6082/0.9457; 0.5882/0.9542; 0.8269/0.7642; 0.8120/0.8391 | `b1413b6b1f164600977b696e817916b4` |
| 10 | Stage-G PCGrad | 4 | 0.861200 / `val_mean_corpora_tasks/macro_recall` | 0.791825 | 0.7535 / 0.7930 / 0.8054 / 0.8154 | 0.5567/0.9502; 0.6176/0.9683; 0.8060/0.8049; 0.7669/0.8639 | `ac9e55aed1af4c6c96c3682ba9d3834e` |
| 11 | Batch 02 H1 paired | 3 | 0.810475 / paired task monitors | 0.791475 | 0.7899 / 0.7748 / 0.7732 / 0.8280 | 0.5979/0.9819; 0.5882/0.9613; 0.7761/0.7703; 0.8120/0.8441 | `dd5cce3e4b694cad97f20de0fc8f6888` + `b87ce3b1959e4bfabac446490ca1db71` |
| 12 | Batch 02 H1 paired | 0 | 0.840650 / paired task monitors | 0.790400 | 0.7735 / 0.8107 / 0.7969 / 0.7805 | 0.5876/0.9593; 0.7059/0.9155; 0.7970/0.7967; 0.7293/0.8317 | `fb36340edf0b405986bd56b4ec3f1b64` + `322643a624974f00a0301a2143967e31` |
| 13 | Stage-G CAGrad | 0 | 0.828800 / `val_mean_corpora_tasks/macro_recall` | 0.790150 | 0.8188 / 0.7683 / 0.7607 / 0.8128 | 0.7010/0.9367; 0.6176/0.9190; 0.8060/0.7154; 0.7519/0.8738 | `fe6f132700db4677a17939f08eb891fe` |
| 14 | Batch 01 H1 corpus-balanced | 4 | 0.863300 / `val_mean_corpora_tasks/macro_recall` | 0.789950 | 0.7825 / 0.7947 / 0.7811 / 0.8015 | 0.5876/0.9774; 0.6176/0.9718; 0.7493/0.8130; 0.7293/0.8738 | `2e02d0c111e0400180a6b03ec77b8257` |
| 15 | Stage-G corrected Progress | 4 | 0.860500 / `val_mean_corpora_tasks/macro_recall` | 0.789500 | 0.7493 / 0.8006 / 0.7975 / 0.8106 | 0.5258/0.9729; 0.6471/0.9542; 0.7433/0.8516; 0.7970/0.8243 | `92e0f5f91bc74a4099ad91373f5d2502` |
| 16 | Weighted-F | 4 | 0.864300 / `val_mean_corpora_tasks/macro_recall` | 0.788825 | 0.7483 / 0.7912 / 0.8054 / 0.8104 | 0.5464/0.9502; 0.6176/0.9648; 0.8060/0.8049; 0.7669/0.8540 | `9e15409719554d2daa5819df7b7b8546` |
| 17 | Batch 02 H1 paired | 2 | 0.791775 / paired task monitors | 0.788750 | 0.7953 / 0.8730 / 0.7303 / 0.7564 | 0.6495/0.9412; 0.8235/0.9225; 0.7493/0.7114; 0.8346/0.6782 | `2e90e2998be94eb893b3cf60a9e33dd6` + `623b12fcfaa6478a879e56bcec6252eb` |
| 18 | Task-specific attention | 4 | 0.857000 / `val_mean_corpora_tasks/macro_recall` | 0.788125 | 0.7310 / 0.8294 / 0.7804 / 0.8117 | 0.4845/0.9774; 0.6765/0.9824; 0.7194/0.8415; 0.7669/0.8564 | `3ef2e7d7eb1e476ba1aefa660677dfcf3` |
| 19 | Stage-G corrected GradNorm | 4 | 0.856100 / `val_mean_corpora_tasks/macro_recall` | 0.787425 | 0.7448 / 0.7842 / 0.8087 / 0.8120 | 0.5258/0.9638; 0.6176/0.9507; 0.7821/0.8354; 0.8120/0.8119 | `688a28433b3c447d995df011e836814d` |
| 20 | Batch 01 H3 dropout-0.3 | 3 | 0.798900 / `val_mean_corpora_tasks/macro_recall` | 0.786825 | 0.8388 / 0.7401 / 0.7579 / 0.8105 | 0.7320/0.9457; 0.5294/0.9507; 0.7881/0.7276; 0.7744/0.8465 | `3184b3caec854b50853e29b4e33174c6` |

Selected checkpoint names and runtime paths for these rows are the exact `epoch=...` files under the corresponding run directories. For the original matrix and Stage-G rows, the full paths are in the source manifest; Batch 01, task-attention, and Batch 02 paths are in their committed evidence reports and local summary directories.

### Top-20 selected checkpoint identities

This index makes the checkpoint evidence explicit for every individual-fold row above.

```text
01 dbmtl/fold1: epoch=12_val_mean_corpora_tasks_macro_recall=0.8656.pt
02 batch02/H1-paired/fold4: epoch=7_val_mean_corpora_depression_macro_recall=0.8392.pt + epoch=22_val_mean_corpora_parkinson_macro_recall=0.9049.pt
03 batch01/H4/fold4: epoch=11_val_mean_corpora_tasks_macro_recall=0.8602.pt
04 stch/fold4: epoch=7_val_mean_corpora_tasks_macro_recall=0.8599.pt
05 corrected-ra-stch/fold4: epoch=7_val_mean_corpora_tasks_macro_recall=0.8559.pt
06 batch01/H3/fold4: epoch=7_val_mean_corpora_tasks_macro_recall=0.8536.pt
07 corrected-ra-stch/fold3: epoch=14_val_mean_corpora_tasks_macro_recall=0.7994.pt
08 dbmtl/fold0: epoch=11_val_mean_corpora_tasks_macro_recall=0.8165.pt
09 batch01/H2/fold4: epoch=7_val_mean_corpora_tasks_macro_recall=0.8616.pt
10 pcgrad/fold4: epoch=7_val_mean_corpora_tasks_macro_recall=0.8612.pt
11 batch02/H1-paired/fold3: epoch=6_val_mean_corpora_depression_macro_recall=0.8080.pt + epoch=10_val_mean_corpora_parkinson_macro_recall=0.8129.pt
12 batch02/H1-paired/fold0: epoch=9_val_mean_corpora_depression_macro_recall=0.8003.pt + epoch=14_val_mean_corpora_parkinson_macro_recall=0.8810.pt
13 cagrad/fold0: epoch=9_val_mean_corpora_tasks_macro_recall=0.8288.pt
14 batch01/H1/fold4: epoch=7_val_mean_corpora_tasks_macro_recall=0.8633.pt
15 corrected-progress/fold4: epoch=7_val_mean_corpora_tasks_macro_recall=0.8605.pt
16 weighted-F/fold4: epoch=7_val_mean_corpora_tasks_macro_recall=0.8643.pt
17 batch02/H1-paired/fold2: epoch=6_val_mean_corpora_depression_macro_recall=0.7187.pt + epoch=9_val_mean_corpora_parkinson_macro_recall=0.8648.pt
18 task-attention/fold4: epoch=18_val_mean_corpora_tasks_macro_recall=0.8570.pt
19 corrected-gradnorm/fold4: epoch=7_val_mean_corpora_tasks_macro_recall=0.8561.pt
20 batch01/H3/fold3: epoch=18_val_mean_corpora_tasks_macro_recall=0.7989.pt
```

## 4. Leaderboard B — top 20 complete five-fold configurations

The TEST UAR columns are fold 0 through fold 4. `VAL–TEST gap` is mean VAL UAR minus mean TEST UAR. Component values are five-fold TEST component means.

| Rank | Configuration | TEST mean ± SD | VAL mean ± SD | VAL–TEST gap | Fold TEST UARs | CARE-D / CARE-P / WSM-D / WSM-P TEST means |
|---:|---|---:|---:|---:|---|---|
| 1 | Stage-G DB-MTL | **0.784105 ± 0.022360** | 0.831680 ± 0.037077 | +0.047575 | 0.792875, 0.810075, 0.748775, 0.784200, 0.784600 | 0.769060 / 0.792960 / 0.776680 / 0.797720 |
| 2 | Batch 02 H1 paired | 0.782660 ± 0.025709 | 0.834070 ± 0.032738 | +0.051410 | 0.790400, 0.738075, 0.788750, 0.791475, 0.804600 | 0.761560 / 0.819940 / 0.775140 / 0.774000 |
| 3 | Stage-G corrected RA-STCH | 0.774885 ± 0.020183 | 0.821740 ± 0.029444 | +0.046855 | 0.777500, 0.748550, 0.760575, 0.793150, 0.794650 | 0.755060 / 0.782300 / 0.755480 / 0.806700 |
| 4 | Stage-G STCH | 0.772610 ± 0.018587 | 0.826560 ± 0.026686 | +0.053950 | 0.775225, 0.743950, 0.769350, 0.779725, 0.794800 | 0.748880 / 0.789700 / 0.751120 / 0.800740 |
| 5 | Batch 01 H1 corpus-balanced | 0.771260 ± 0.020104 | 0.826820 ± 0.033126 | +0.055560 | 0.781575, 0.780425, 0.765375, 0.738975, 0.789950 | 0.768520 / 0.788260 / 0.746740 / 0.781520 |
| 6 | Stage-G corrected Progress | 0.770955 ± 0.021197 | 0.824480 ± 0.030391 | +0.053525 | 0.779675, 0.737650, 0.762800, 0.785150, 0.789500 | 0.761800 / 0.770400 / 0.757460 / 0.794160 |
| 7 | Batch 01 H4 hidden-32 | 0.770180 ± 0.023836 | 0.828640 ± 0.024704 | +0.058460 | 0.776700, 0.758375, 0.735325, 0.784325, 0.796175 | 0.793200 / 0.779020 / 0.762580 / 0.745920 |
| 8 | Batch 01 H2 depression-weighted | 0.769855 ± 0.021793 | 0.827260 ± 0.025117 | +0.057405 | 0.783500, 0.738700, 0.756750, 0.778000, 0.792325 | 0.765980 / 0.749100 / 0.756940 / 0.807400 |
| 9 | Batch 01 H3 dropout-0.3 | 0.769195 ± 0.021059 | 0.822460 ± 0.026674 | +0.053265 | 0.768000, 0.745175, 0.752300, 0.786825, 0.793675 | 0.740940 / 0.777820 / 0.757560 / 0.800460 |
| 10 | Stage-G CAGrad | 0.768610 ± 0.022259 | 0.818980 ± 0.031191 | +0.050370 | 0.790150, 0.734200, 0.765350, 0.766850, 0.786500 | 0.750020 / 0.764520 / 0.752620 / 0.807280 |
| 11 | Task-specific attention | 0.766730 ± 0.018514 | 0.822980 ± 0.030914 | +0.056250 | 0.765175, 0.737775, 0.767350, 0.775225, 0.788125 | 0.762900 / 0.778520 / 0.742000 / 0.783500 |
| 12 | Stage-G corrected GradNorm | 0.766090 ± 0.017841 | 0.823300 ± 0.027003 | +0.057210 | 0.776400, 0.740950, 0.757425, 0.768250, 0.787425 | 0.746060 / 0.760300 / 0.753340 / 0.804660 |
| 13 | Stage-G PCGrad | 0.764980 ± 0.026095 | 0.825720 ± 0.024291 | +0.060740 | 0.775400, 0.722425, 0.761725, 0.773525, 0.791825 | 0.748180 / 0.762300 / 0.756100 / 0.793340 |
| 14 | Weighted-F | 0.764310 ± 0.023065 | 0.827060 ± 0.027880 | +0.062750 | 0.754725, 0.732500, 0.760900, 0.784600, 0.788825 | 0.740280 / 0.768660 / 0.757680 / 0.790620 |
| 15 | Batch 02 H2 corpus-heads | 0.752495 ± 0.026635 | 0.823745 ± 0.031097 | +0.071250 | 0.747600, 0.716175, 0.743450, 0.786275, 0.768975 | 0.798600 / 0.669680 / 0.741020 / 0.800680 |
| 16 | DEEP-003 EmotionMSP embeddings MLP | 0.751020 ± 0.025894 | 0.791880 ± 0.039534 | +0.040860 | 0.722550, 0.737500, 0.738000, 0.776775, 0.780275 | 0.754680 / 0.705660 / 0.768980 / 0.774760 |
| 17 | DEEP-008 F gated fusion | 0.750475 ± 0.029723 | 0.808360 ± 0.033308 | +0.057885 | 0.742675, 0.706175, 0.752500, 0.764275, 0.786750 | 0.734060 / 0.753200 / 0.756560 / 0.758080 |
| 18 | DEEP-006 EmotionMSP embeddings + trajectories MD-Former | 0.746740 ± 0.011309 | 0.791940 ± 0.036498 | +0.045200 | 0.749175, 0.735775, 0.750925, 0.735525, 0.762300 | 0.730260 / 0.697040 / 0.776360 / 0.783300 |
| 19 | DEEP-006 EmotionMSP embeddings + TRILLsson MLP | 0.737400 ± 0.029732 | 0.795180 ± 0.054572 | +0.057780 | 0.731475, 0.753300, 0.688025, 0.761225, 0.752975 | 0.746680 / 0.745840 / 0.751200 / 0.705880 |
| 20 | DEEP-006 EmotionMSP embeddings + trajectories MLP | 0.735265 ± 0.020390 | 0.790620 ± 0.037013 | +0.055355 | 0.707175, 0.733650, 0.728025, 0.745475, 0.762000 | 0.706400 / 0.707120 / 0.761400 / 0.766140 |

Complete five-fold provenance is available in the committed source manifest for the original matrix and Stage-G entries. The modern configurations are supported by `docs/evidence/DEEP008_BATCH01_FOUR_HYPOTHESES_RESULTS_EN.md`, `docs/evidence/DEEP008_TASK_ATTENTION_FIVEFOLD_RESULTS_EN.md`, `docs/evidence/DEEP008_BATCH02_FIVEFOLD_RESULTS_EN.md`, and their corresponding local `summary.txt` and checkpoint paths. Each selected fold has one run identity and one original VAL-selected checkpoint.

### Top-20 complete-configuration MLflow provenance

The following fold-ordered IDs identify every run in the complete configurations above. The corresponding selected checkpoint names are in the committed source manifest or the modern evidence reports.

```text
Stage-G DB-MTL: 9577904d62e84d54bc4564f91ec69b7e, 7add8f3965c347ba88f20facac5a66df, 80a098e23c144683a7185c46d61ccb1c, ccd497e18e854c4caf8ea98075103df6, 047b19dc4cbd459eba57adf4b3fd574d
Batch 02 H1 paired: fb36340edf0b405986bd56b4ec3f1b64+322643a624974f00a0301a2143967e31; a7e7e17a4a62428cb756316fb7a61088+055188fcdfce41289019c1027090771c; 2e90e2998be94eb893b3cf60a9e33dd6+623b12fcfaa6478a879e56bcec6252eb; dd5cce3e4b694cad97f20de0fc8f6888+b87ce3b1959e4bfabac446490ca1db71; 143dd63b49e445fd92fc5ba8361c51c6+6d729265b30e49d0acc728bac589358e
Stage-G corrected RA-STCH: fc5c9a7f48c54148b8db141469175b37, 833ec87ccad34881af8b5e6403825587, 4850fa04ffa6458a802d11e9d89f6fc8, cedb5b0f4f564274b346d270f1ffdea5, 96df82073f9d43fb81aac99c0eea3b8d
Stage-G STCH: 03a0d4c7d99741f5a6f4de369943d840, 091ae97c22424743ad3c2f1038ff3d4a, dd1d68a4be7c46068d75502fc784c4c0, 28a7ab859bb24ae084420b95be66890f, 74399d3faae447adbc66425529052fe4
Batch 01 H1: e0343b25417241558065c8791edd785a, befcb053af4a4a4d8680ea9e158afc11, 2753edec66864bd4bb8c7ce23eff5a4d, 8b2ab9d70cad4d2488cf08543921f49c, 2e02d0c111e0400180a6b03ec77b8257
Stage-G corrected Progress: ae158a4ca28e48be90f5180bd2ae01df, f67b079b360e40f5a24b41a05bf2e1ad, 876017b8e3614c0490678463df90bd68, 63ec2b2dab46483594f25d423291d18c, 92e0f5f91bc74a4099ad91373f5d2502
Batch 01 H4: b589c3fb24394fb2a46b6239d6a7d67e, 4fc5e49a75a94992ab440b1780bf7468, 691510075f234b6e848679dc6b85b2c8, efbf0a61e6c3431fa153663619b18556, 3debccce04c24b74ae317513f1324719
Batch 01 H2: c9cd55202e8f430f8730974a1692a8aa, 496ac464421a4211aca61a8a4b611040, ede9aaff1c9449a7afe619f6d1bb16f1, 05c42097cea54b9aa4303464df245120, b1413b6b1f164600977b696e817916b4
Batch 01 H3: 703710dffd994c72ab1823181447c968, 58d7faaae8e44ed3a67920a4836c9684, cfdf3b9e559d4283bf11a2a66f4a529b, 3184b3caec854b50853e29b4e33174c6, 621feea4d02d4079aff12ced7f39f561
Stage-G CAGrad: fe6f132700db4677a17939f08eb891fe, e3f259d4448c42cdb80b2cecfcffac72, f785cf770c23467fbce9455680825824, 38a65526d8684df89d2c24dd0c7f0c02, c1951c3bd4ff433aa7973a9129c1ad9f
Task attention: c2c4dae180bd4d21936a8182bb723c56, d6053a4ab09342fab64a5a871e6bbebb, 0d8123e34f0745d58660f873ab0dccf2, 896e3ae6b0354c65a7094df5b59b07b1, 3ef2e7d7eb1e476ba1aefa660677dfcf
Stage-G corrected GradNorm: 707670be07684a98ab30807e9d175b88, 201bb24f68794809991dbdd2e4eafac3, 491c5bd634c5409aa7878ea900bd0316, f3e3dca3fa864a6cb3abfb87fd6b5e23, 688a28433b3c447d995df011e836814d
Stage-G PCGrad: e241d06c55ee4b509979f62b7e87f6ee, b086cf2380f5440a8723bcdf7a7b92df, 2de59fa8a2504ca0a2f8200c776000c2, 6738b37f02e34f44a12e87f193125d03, ac9e55aed1af4c6c96c3682ba9d3834e
Weighted-F: 542b03733f9b45f39e40ab0ec0051c0d, bf9395ed33d640d28653102fc6620619, 4afa8dfef8144666a14e4a44017c77bb, 93c47d48889b4445bc293cab24b40735, 9e15409719554d2daa5819df7b7b8546
Batch 02 H2: 447b733f80f648288c433478dddaf8f0, 943147ca998f4799bcfb5bb747e05d55, 38432ea423d44d7080f17f8366212e9c, 9029dcbe0064499da65ea33b9a376707, 825f9ba86bf64d5aadc5583fd16bfb12
DEEP-003 EmotionMSP embeddings MLP: 53d37837e0724d4d9652cbc8d91c205e, edb55a570cde4e179c901a1f150b2e3d, 42122aeda15342c1811abde068c0ae81, d96994abca124fc6822c211e38b2e4df, 9868ea8a42a04bf4ad92a5af50acc792
DEEP-008 F: ddf7c9e3e72641abafca3e1a426cd58f, bf9d5deb4c7b42a6a89ee978a0c88182, 108ee477891c46749bd1a0847b980bc3, 302d74b7ab764bcc8980d9ecd4a8cacf, c15776880971492e863401560de384de
DEEP-006 EmotionMSP embeddings + trajectories MD-Former: f33d10ae894f458d82342c8b67c22e27, 032f7ebfb88b4f678e76af6d759f682c, 1554aa873dfc4b33828f55185871b5dc, d6bb7bad835a4e0985adc616577cc6aa, 8f8157cbdca74c418f2efc8b9be75fa5
DEEP-006 EmotionMSP embeddings + TRILLsson MLP: f9da32597cc44bec9dccbe67a9aa95db, d413a07e1ab540c2b581a9c36e788244, ad4e6ca8f6694955b0bf8e4c0e2d6c47, 7a20fb09cddc43d390c45b79b1e3cd32, 322a830e2a2e4a218a009327e9c439bd
DEEP-006 EmotionMSP embeddings + trajectories MLP: b8adacf98cd648a5b1e5d7e6f1a4c477, bdfb616637764d5a94426bf317374ad6, ec39c6543e7945d5a046fe0aab7cb8b3, 169699367b6d4ef1a1997b6c11075f93, 26cc00d2ff844488839f11112cd0bf8f
```

## 5. Direct answers

1. **Highest valid individual-fold TEST UAR:** `0.810075`.
2. **Configuration:** Stage-G DB-MTL, fold 1; VAL UAR `0.865600`, selected checkpoint `epoch=12_val_mean_corpora_tasks_macro_recall=0.8656.pt`, MLflow run `7add8f3965c347ba88f20facac5a66df`.
3. **Highest complete five-fold mean TEST UAR:** Stage-G DB-MTL, `0.784105 ± 0.022360`.
4. **Did any individual fold reach 0.8500 TEST UAR?** No. The maximum was `0.810075`, `0.039925` below target.
5. **Did any complete five-fold configuration reach 0.8500 mean TEST UAR?** No. The best mean was `0.784105`, `0.065895` below target.
6. **High TEST with weak corresponding VAL:** the largest individual TEST values are generally fold-specific and have substantial VAL–TEST gaps. The clearest example is DB-MTL fold 1 (`0.865600` VAL versus `0.810075` TEST). Corrected RA-STCH fold 3 has `0.799400` VAL and `0.793150` TEST, while Batch 02 H1 fold 2 has `0.791775` VAL and `0.788750` TEST. These are descriptive instability/generalization observations, not selection recommendations.
7. **Previously missing results:** yes. The prior manager summaries did not provide one unified ranking across the original 110-run matrix, Stage-G methods, Batch 01, task attention, and Batch 02. This audit adds that retrospective coverage and explicitly separates individual fold peaks from complete five-fold evidence.

## 6. Non-selectable TEST oracle

No non-selected-epoch TEST oracle ranking was used or added. Any historical TEST peak from an epoch other than the original VAL-selected checkpoint is not a legitimate result for this audit and must not be used for model selection, promotion, tuning, or reruns.

## 7. Interpretation and limitations

The highest historical evidence is a DB-MTL individual fold at `0.810075`, not an 85% result. The strongest robust configuration is DB-MTL at `0.784105 ± 0.022360`; its average VAL–TEST gap is `0.047575`. The Batch 02 H1 pairing is second by mean TEST UAR at `0.782660 ± 0.025709`, but it is not a single jointly trained model and remains subject to its original VAL decision rules.

These are descriptive historical TEST statistics. They do not reverse any prior VAL-based rejection, establish statistical significance, or justify further experiments. The audit does not claim that the previously inspected TEST sets are pristine holdouts. No TEST-driven recommendation is made.

## 8. Reproducibility and evidence paths

- Base commit: `37c18cc6c4688855140576e5e66e0a002f8591ec`.
- Historical per-fold table: `docs/evidence/tables/DEEP008_HISTORICAL_TEST_METRICS_PER_FOLD.csv`.
- Historical provenance manifest: `docs/evidence/tables/DEEP008_HISTORICAL_TEST_SOURCE_MANIFEST.csv`.
- Historical aggregate table: `docs/evidence/tables/DEEP008_HISTORICAL_TEST_METRICS_AGGREGATED.csv`.
- Existing export/audit method: `scripts/export_historical_test_metrics.py`.
- Modern evidence: Batch 01, task-attention, Batch 02, and Stage-G reports under `docs/evidence/`.
- Local runtime evidence: existing `logs/**/summary.txt`, `train.log`, checkpoint filenames, launcher ledgers, and MLflow databases; these artifacts were read-only inputs and were not committed.
- Statistics use arithmetic component means and sample standard deviation (`ddof=1`).
