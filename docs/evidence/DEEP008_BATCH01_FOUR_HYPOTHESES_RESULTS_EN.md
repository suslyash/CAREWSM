# DEEP-008 Batch 01 — Four Hypotheses, Five-Fold Results

## Scope and protocol

This report records the approved Batch 01 experiment from source commit
`a4fd2f8ff8fc655fbd63f047c3030e8f2f9bbb00`. The production launcher executed
exactly 20 sequential CUDA runs: H1, H2, H3, and H4 on folds 0–4. No control
run was retrained and no implementation, configuration, split, metric, or
feature artifact was changed.

The hypotheses were:

- H1 — corpus-balanced weighted BCE;
- H2 — depression- and Parkinson-weighted BCE;
- H3 — weighted Parkinson BCE with dropout 0.3;
- H4 — weighted Parkinson BCE with hidden dimension 32.

All runs used the existing EmotionMSP + TRILLsson turn-level features, fixed
joint CARE+WSM folds, seed 5, and the native Chimera-ML Dataset, DataModule,
Trainer, callbacks, metrics, and logging interfaces. The canonical validation
monitor was `val_mean_corpora_tasks/macro_recall`. The historical weighted-F
five-fold runs are the control and were reused without retraining.

The batch runtime namespace was
`logs/audio_representation_pilot/deep008_batch01/`, with its dedicated
MLflow database and sequential launcher progress file. Runtime artifacts are
local experiment evidence and are not committed to the repository.

## VAL-first selection and acceptance

The selected epoch in every run was determined from the maximum canonical VAL
monitor row only. The launcher completion checks passed for all 20 runs:
exactly one runtime directory, completed training log, valid summary, one
non-`last.pt` selected checkpoint whose filename value matched the selected VAL
row to filename precision, and one FINISHED MLflow run.

| configuration | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean | sample SD | difference vs F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weighted-F control | 0.8321 | 0.8400 | 0.7979 | 0.8010 | 0.8643 | 0.8271 | 0.0279 | — |
| H1 corpus-balanced | 0.8351 | 0.8506 | 0.7864 | 0.7987 | 0.8633 | 0.8268 | 0.0331 | -0.0002 |
| H2 depression-weighted | 0.8318 | 0.8376 | 0.8010 | 0.8043 | 0.8616 | 0.8273 | 0.0251 | +0.0002 |
| H3 dropout 0.3 | 0.8322 | 0.8370 | 0.7906 | 0.7989 | 0.8536 | 0.8225 | 0.0267 | -0.0046 |
| H4 hidden 32 | 0.8428 | 0.8320 | 0.8003 | 0.8079 | 0.8602 | 0.8286 | 0.0247 | +0.0016 |

The paired fold differences versus weighted-F were:

- H1: `+0.0030, +0.0106, -0.0115, -0.0023, -0.0010`;
- H2: `-0.0003, -0.0024, +0.0031, +0.0033, -0.0027`;
- H3: `+0.0001, -0.0030, -0.0073, -0.0021, -0.0107`;
- H4: `+0.0107, -0.0080, +0.0024, +0.0069, -0.0041`.

None of the four hypotheses satisfies the provisional VAL acceptance rule of
at least `+0.010` mean overall VAL UAR. Therefore all four are rejected for
further architecture-selection purposes. The component no-deterioration rule
was not sufficient to change that decision: all component mean changes stayed
within the allowed `0.020` absolute bound, but the required overall gain was
absent. CARE was evaluated explicitly; WSM remains a useful control population
but has one valid turn per record and therefore cannot provide temporal
variation evidence.

For reference, the historical DB-MTL VAL result was `0.8317 ± 0.0371`. Batch
H4 is `0.8286 ± 0.0247`, below that reference and without meeting the
pre-registered improvement criterion.

### VAL component means

Each cell is five-fold mean `Macro Recall (Patient Recall / Healthy Recall)`.

| configuration | CARE Depression | CARE Parkinson | WSM Depression | WSM Parkinson |
|---|---:|---:|---:|---:|
| Weighted-F | 0.8264 (0.7451 / 0.9077) | 0.8227 (0.6988 / 0.9466) | 0.7562 (0.7377 / 0.7748) | 0.9028 (0.9203 / 0.8853) |
| H1 | 0.8471 (0.7629 / 0.9314) | 0.8217 (0.6930 / 0.9505) | 0.7432 (0.6995 / 0.7869) | 0.8953 (0.8942 / 0.8964) |
| H2 | 0.8304 (0.7843 / 0.8765) | 0.8194 (0.6928 / 0.9461) | 0.7572 (0.7755 / 0.7390) | 0.9021 (0.9176 / 0.8865) |
| H3 | 0.8296 (0.7568 / 0.9025) | 0.8153 (0.6916 / 0.9389) | 0.7522 (0.7391 / 0.7653) | 0.8926 (0.9185 / 0.8668) |
| H4 | 0.8514 (0.7838 / 0.9191) | 0.8150 (0.6939 / 0.9361) | 0.7462 (0.7200 / 0.7724) | 0.9019 (0.9284 / 0.8755) |

## Descriptive TEST results after VAL freeze

TEST values below were read only from the already selected VAL epoch for each
fold. They were not used for checkpoint, threshold, architecture, or
acceptance decisions.

| configuration | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean | sample SD | difference vs weighted-F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weighted-F control | 0.7547 | 0.7325 | 0.7609 | 0.7846 | 0.7888 | 0.7643 | 0.0231 | — |
| H1 corpus-balanced | 0.7816 | 0.7804 | 0.7654 | 0.7390 | 0.7900 | 0.7713 | 0.0201 | +0.0070 |
| H2 depression-weighted | 0.7835 | 0.7387 | 0.7568 | 0.7780 | 0.7923 | 0.7699 | 0.0218 | +0.0056 |
| H3 dropout 0.3 | 0.7680 | 0.7452 | 0.7523 | 0.7868 | 0.7937 | 0.7692 | 0.0211 | +0.0049 |
| H4 hidden 32 | 0.7767 | 0.7584 | 0.7353 | 0.7843 | 0.7962 | 0.7702 | 0.0238 | +0.0059 |

The historical DB-MTL TEST UAR was `0.784105`; the long-term target is
`0.8500`. The best Batch 01 descriptive mean is H1 at `0.7713`, which is
`-0.0787` below the target and `-0.0128` below DB-MTL. Relative to weighted-F,
the largest descriptive increase is H1 at `+0.0070`, still below the target
and not evidence that can reverse the VAL decision.

### TEST component means

Each cell is five-fold mean `Macro Recall (Patient Recall / Healthy Recall)`.

| configuration | CARE Depression | CARE Parkinson | WSM Depression | WSM Parkinson |
|---|---:|---:|---:|---:|
| Weighted-F | 0.7403 (0.5340 / 0.9466) | 0.7687 (0.5824 / 0.9549) | 0.7577 (0.7552 / 0.7602) | 0.7906 (0.7248 / 0.8564) |
| H1 | 0.7685 (0.5588 / 0.9783) | 0.7883 (0.6118 / 0.9648) | 0.7467 (0.7045 / 0.7890) | 0.7815 (0.6962 / 0.8668) |
| H2 | 0.7660 (0.6062 / 0.9258) | 0.7491 (0.5412 / 0.9570) | 0.7569 (0.8054 / 0.7085) | 0.8074 (0.7534 / 0.8614) |
| H3 | 0.7409 (0.5443 / 0.9376) | 0.7778 (0.6000 / 0.9556) | 0.7576 (0.7582 / 0.7569) | 0.8005 (0.7549 / 0.8460) |
| H4 | 0.7932 (0.6371 / 0.9493) | 0.7790 (0.6235 / 0.9345) | 0.7626 (0.7463 / 0.7789) | 0.7459 (0.6587 / 0.8332) |

## Interpretation and decision

Batch 01 did not establish a VAL-supported improvement over weighted-F for any
hypothesis. H4 has the highest mean VAL UAR, but its gain is only `+0.0016`.
The TEST differences are descriptive only and are reported to preserve the
pre-registered evidence record; they do not alter the VAL-based rejection.
There is no basis in this batch for selecting a new model, changing the
canonical protocol, or launching additional variants.

## Reproducibility record

- Source commit: `a4fd2f8ff8fc655fbd63f047c3030e8f2f9bbb00`.
- Batch launcher: `scripts/run_audio_deep008_batch01.sh`.
- Batch configurations: `configs/audio_representation_pilot/deep008/batch01/`.
- Runtime namespace: `logs/audio_representation_pilot/deep008_batch01/`.
- Selection monitor: `val_mean_corpora_tasks/macro_recall`.
- Historical control monitor: `dev_mean_macro_recall`.
- Statistics: arithmetic five-fold means; sample SD uses `ddof=1`.
- Data access was limited to existing aggregate training/validation/test
  summary rows and completion evidence. No raw audio, sample-level TEST
  records, identities, embeddings, or new features were inspected.

