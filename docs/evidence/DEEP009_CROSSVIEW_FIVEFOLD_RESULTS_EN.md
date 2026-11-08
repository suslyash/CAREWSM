# DEEP-009 CrossView five-fold results

## Scope and protocol

This report records the five authorized DEEP-009 CrossView runs executed
sequentially from source commit `8a91a06e52d4710f54503b824f76a969cf73942b`.
The runs used the existing joint CARE+WSM Chimera-ML pipeline, native
EmotionMSP and TRILLsson turn-level features, fixed five-fold splits, seed 5,
and the prepared CrossView configuration. No model code, configuration,
feature artifact, checkpoint, or runtime artifact was changed during analysis.

Checkpoint selection used only
`val_mean_corpora_tasks/macro_recall`. TEST values below are read from the
same selected epoch and are descriptive only.

## Run completion and selected checkpoints

The dedicated MLflow experiment was
`audio_representation_pilot_deep009_crossview` in the local, uncommitted
database `logs/audio_representation_pilot/deep009_crossview/mlflow.db`.
Each fold had one runtime directory, one `train.log` containing
`Training complete`, one `summary.txt`, one selected non-`last.pt` checkpoint,
and one FINISHED MLflow run.

| Fold | Runtime directory | MLflow run ID | Selected epoch | Checkpoint |
|---:|---|---|---:|---|
| 0 | `F_crossview_fold0_2026-11-08_20-05_carewsm_crossview_b4cb34a6` | `946e8bdc2c4e43118b6cf9a70f66e374` | 10 | `epoch=10_val_mean_corpora_tasks_macro_recall=0.8308.pt` |
| 1 | `F_crossview_fold1_2026-11-08_20-07_carewsm_crossview_53ab35bf` | `15620228cf04434e8668afbd93a88c79` | 9 | `epoch=9_val_mean_corpora_tasks_macro_recall=0.8428.pt` |
| 2 | `F_crossview_fold2_2026-11-08_20-09_carewsm_crossview_4ba8df85` | `d45be4674c1d40329ff04ee438d6be50` | 16 | `epoch=16_val_mean_corpora_tasks_macro_recall=0.7933.pt` |
| 3 | `F_crossview_fold3_2026-11-08_20-12_carewsm_crossview_e2b3ad7d` | `bd30ae8f536a4bd8831b91c0fc2127d9` | 16 | `epoch=16_val_mean_corpora_tasks_macro_recall=0.7962.pt` |
| 4 | `F_crossview_fold4_2026-11-08_20-15_carewsm_crossview_f61de61c` | `229741fad57a4b51a891c3c56ee5de18` | 20 | `epoch=20_val_mean_corpora_tasks_macro_recall=0.8721.pt` |

## VAL results and frozen decision

Component order is CARE Depression, CARE Parkinson, WSM Depression, WSM
Parkinson. Each cell is `Macro Recall (Patient Recall / Healthy Recall)`.

| Fold | CARE Depression | CARE Parkinson | WSM Depression | WSM Parkinson | Overall VAL UAR |
|---:|---|---|---|---|---:|
| 0 | 0.8435 (0.7209 / 0.9662) | 0.7999 (0.6471 / 0.9528) | 0.7786 (0.8347 / 0.7224) | 0.9012 (0.9170 / 0.8854) | 0.830800 |
| 1 | 0.8943 (0.8791 / 0.9094) | 0.8876 (0.7949 / 0.9804) | 0.7144 (0.6261 / 0.8028) | 0.8748 (0.8596 / 0.8899) | 0.842775 |
| 2 | 0.7355 (0.5517 / 0.9194) | 0.8591 (0.7742 / 0.9441) | 0.6847 (0.6337 / 0.7356) | 0.8939 (0.9520 / 0.8359) | 0.793300 |
| 3 | 0.8589 (0.9024 / 0.8153) | 0.6987 (0.4375 / 0.9599) | 0.7119 (0.7762 / 0.6476) | 0.9152 (0.9558 / 0.8746) | 0.796175 |
| 4 | 0.8876 (0.8444 / 0.9308) | 0.9274 (0.9118 / 0.9430) | 0.7783 (0.7471 / 0.8094) | 0.8951 (0.8772 / 0.9130) | 0.872100 |

CrossView VAL UAR is `0.827030 ± 0.033103` (mean ± sample SD). The
historical weighted-F control is `0.827060 ± 0.027880`; historical DB-MTL is
`0.831680 ± 0.037077`. The paired CrossView-minus-weighted-F differences are
`-0.001300`, `+0.002775`, `-0.004600`, `-0.004825`, and `+0.007800`, giving
2/5 fold wins.

| Component | CrossView mean | Weighted-F mean | Difference |
|---|---:|---:|---:|
| CARE Depression | 0.843960 | 0.826420 | +0.017540 |
| CARE Parkinson | 0.834540 | 0.822700 | +0.011840 |
| WSM Depression | 0.733580 | 0.756240 | -0.022660 |
| WSM Parkinson | 0.896040 | 0.902760 | -0.006720 |

The pre-registered decision is **REJECT**. All five runs are valid, but the
mean does not reach the required `0.837060`, there are only two paired fold
wins, and WSM Depression declines by more than the allowed `0.020`. This
decision was frozen before reading the TEST rows.

## Descriptive TEST results

These values are taken from the recorded TEST metrics at each VAL-selected
epoch. No TEST inference, checkpoint selection, threshold tuning, or further
experiment decision was performed.

| Fold | CARE Depression | CARE Parkinson | WSM Depression | WSM Parkinson | TEST UAR |
|---:|---|---|---|---|---:|
| 0 | 0.7603 (0.5567 / 0.9638) | 0.7730 (0.5882 / 0.9577) | 0.7583 (0.7910 / 0.7256) | 0.8128 (0.7519 / 0.8738) | 0.776100 |
| 1 | 0.7728 (0.5773 / 0.9683) | 0.8077 (0.6471 / 0.9683) | 0.7551 (0.7134 / 0.7967) | 0.7918 (0.7594 / 0.8243) | 0.781850 |
| 2 | 0.8098 (0.7010 / 0.9186) | 0.7859 (0.6176 / 0.9542) | 0.6979 (0.7433 / 0.6524) | 0.7569 (0.7218 / 0.7921) | 0.762625 |
| 3 | 0.8430 (0.7629 / 0.9231) | 0.7313 (0.5294 / 0.9331) | 0.7497 (0.7881 / 0.7114) | 0.7831 (0.7519 / 0.8144) | 0.776775 |
| 4 | 0.7357 (0.5258 / 0.9457) | 0.7541 (0.5294 / 0.9789) | 0.8114 (0.8179 / 0.8049) | 0.7750 (0.6466 / 0.9035) | 0.769050 |

The descriptive TEST UAR is `0.773280 ± 0.007501`. The mean VAL-to-TEST gap
is `0.053750`, and the mean is `0.076720` below the 0.850000 target.

| Component | TEST mean | Patient Recall mean | Healthy Recall mean |
|---|---:|---:|---:|
| CARE Depression | 0.784320 | 0.624740 | 0.943900 |
| CARE Parkinson | 0.770400 | 0.582340 | 0.958440 |
| WSM Depression | 0.754480 | 0.770740 | 0.738200 |
| WSM Parkinson | 0.783920 | 0.726320 | 0.841620 |

## Conclusion and limitations

CrossView does not establish a VAL-supported improvement over weighted-F and
is rejected under the frozen protocol. The TEST numbers are descriptive
records of the five completed, VAL-selected runs; they cannot reverse that
decision or authorize another experiment. This report does not claim that the
TEST sets are pristine holdouts.

Runtime directories, checkpoints and the MLflow database remain local and
uncommitted. Only aggregate metrics, run identifiers and checkpoint names are
reported here; no raw audio, sample-level records, embeddings or identities
are included.
