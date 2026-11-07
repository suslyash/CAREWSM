# DEEP008 Stage G — WSM-inspired CAREWSM multitask suite

## Outcome

The complete five-fold, eight-method suite is finished: 40/40 method/fold
combinations have standard Chimera-ML artifacts. The five existing weighted-
Parkinson-BCE runs are the Equal control; the seven new methods were run once
per fold. No training run was restarted after successful completion and no
completed control or new run was overwritten.

The experiment uses the best existing CAREWSM model (`carewsm_gated_fusion`),
EmotionMSP embeddings plus TRILLsson 4.1, joint CARE+WSM folds, seed 5,
original AdamW settings, 50-epoch maximum with early stopping, TRAIN-only
Parkinson `pos_weight`, and validation-only checkpoint selection. The Equal
implementation is content-equivalent to the existing weighted-F control:
the same observed-label BCE objectives, Parkinson weight, model, data,
optimizer, seed, fold and monitor are used.

## Methods and implementation

| Method | CAREWSM implementation | WSM-inspired mechanism |
|---|---|---|
| Equal | existing weighted-F control | equal observed-task average |
| Static-STCH | `carewsm_stageg_mtl` | fixed temperature-softmax task weights |
| Progress | `carewsm_stageg_mtl` + callback | validation-only relative progress |
| RA-STCH | `carewsm_stageg_mtl` + callback | observed-only reliability-adjusted progress |
| GradNorm | `carewsm_stageg_mtl` | shared-gradient norm balancing |
| PCGrad | `carewsm_stageg_mtl` | conflict projection on shared parameters |
| CAGrad | `carewsm_stageg_mtl` | two-task simplex solve on shared gradients |
| DB-MTL | `carewsm_stageg_mtl` | EMA/max-norm shared-gradient update |

The registered loss binds to the actual gated-fusion ownership: 12 shared
projection/gate tensors, six depression-head tensors, and six
Parkinson-head tensors. Private task gradients remain private; balancing is
applied only to the shared representation. No standalone trainer, logger,
metric, callback, or checkpoint subsystem was added.

## Run accounting and selected validation checkpoints

Each row is the checkpoint selected by `dev_mean_macro_recall`. Run IDs are
the standard MLflow IDs. `VAL` is the selected equal-weight mean across CARE
and WSM × Depression and Parkinson Macro Recall.

| method | fold | epoch | VAL | MLflow run |
|---|---:|---:|---:|---|
| Equal | 0 | 14 | 0.8321 | 18e922c2 |
| Equal | 1 | 3 | 0.8400 | 309e95a2 |
| Equal | 2 | 11 | 0.7979 | 151e5750 |
| Equal | 3 | 14 | 0.8010 | f7e04dbe |
| Equal | 4 | 7 | 0.8643 | 410a2700 |
| Static-STCH | 0 | 8 | 0.8363 | e7902c8d |
| Static-STCH | 1 | 3 | 0.8383 | ed43ba64 |
| Static-STCH | 2 | 11 | 0.8001 | c2bd2d9e |
| Static-STCH | 3 | 18 | 0.7982 | f353fa84 |
| Static-STCH | 4 | 7 | 0.8599 | 9eb4affb |
| Progress | 0 | 9 | 0.8422 | 4c10bb7e |
| Progress | 1 | 3 | 0.8352 | 10ebaaf7 |
| Progress | 2 | 11 | 0.7889 | 8263e07a |
| Progress | 3 | 14 | 0.7966 | 59e4db96 |
| Progress | 4 | 17 | 0.8618 | 41a999d2 |
| RA-STCH | 0 | 9 | 0.8386 | ac7a0cb8 |
| RA-STCH | 1 | 3 | 0.8358 | 0143b81d |
| RA-STCH | 2 | 11 | 0.8002 | 65a9a61b |
| RA-STCH | 3 | 21 | 0.7994 | 784d60a2 |
| RA-STCH | 4 | 7 | 0.8498 | d036cb79 |
| GradNorm | 0 | 8 | 0.8280 | dbe5feb5 |
| GradNorm | 1 | 3 | 0.8427 | 02caa124 |
| GradNorm | 2 | 11 | 0.7910 | 46f2a86c |
| GradNorm | 3 | 17 | 0.8013 | 6b162955 |
| GradNorm | 4 | 17 | 0.8587 | 679443db |
| PCGrad | 0 | 9 | 0.8297 | 19409f5f |
| PCGrad | 1 | 3 | 0.8320 | 4c28fe99 |
| PCGrad | 2 | 11 | 0.8027 | eef4e882 |
| PCGrad | 3 | 18 | 0.8030 | 944574ad |
| PCGrad | 4 | 7 | 0.8612 | c7b74aaa |
| CAGrad | 0 | 9 | 0.8288 | 6b38a5a4 |
| CAGrad | 1 | 3 | 0.8354 | cea93f19 |
| CAGrad | 2 | 11 | 0.7916 | 875bb5d8 |
| CAGrad | 3 | 12 | 0.7823 | 7ffe96a9 |
| CAGrad | 4 | 7 | 0.8568 | f46f0efb |
| DB-MTL | 0 | 11 | 0.8165 | cf9487cf |
| DB-MTL | 1 | 12 | 0.8656 | e5db9469 |
| DB-MTL | 2 | 14 | 0.8125 | 0faaee34 |
| DB-MTL | 3 | 9 | 0.7886 | bba77f53 |
| DB-MTL | 4 | 28 | 0.8752 | eb9e0ae2 |

The full MLflow IDs, resolved configs, checkpoints, and per-epoch wide
`summary.txt` files are retained under
`logs/audio_representation_pilot/deep008_stageg_wsm/`. The five Equal control
summaries are under `logs/audio_representation_pilot/deep008_weighted_f/`.
Each summary contains all six required metrics for CARE TEST, WSM TEST,
CARE VAL, and WSM VAL, plus native corpus sample counts. Task-specific label
support counts were not emitted by the existing Chimera summary schema.

## Five-fold aggregate evidence

Values are mean ± sample standard deviation over folds. The metric order in
each population is Macro F1, Weighted F1, Macro Recall, Weighted Recall,
Patient Recall, Healthy Recall. TEST task average is the arithmetic mean of
the four corpus/task Macro Recall values; it was not used for selection.

| method | VAL monitor | TEST task-average Macro Recall |
|---|---:|---:|
| Equal | 0.8271 ± 0.0279 | 0.7643 ± 0.0231 |
| Static-STCH | 0.8266 ± 0.0267 | 0.7726 ± 0.0186 |
| Progress | 0.8249 ± 0.0311 | 0.7720 ± 0.0305 |
| RA-STCH | 0.8248 ± 0.0234 | 0.7668 ± 0.0227 |
| GradNorm | 0.8243 ± 0.0282 | 0.7668 ± 0.0198 |
| PCGrad | 0.8257 ± 0.0243 | 0.7650 ± 0.0261 |
| CAGrad | 0.8190 ± 0.0312 | 0.7686 ± 0.0223 |
| DB-MTL | 0.8317 ± 0.0371 | 0.7841 ± 0.0224 |

Paired TEST task-average differences versus Equal are: Static-STCH
`+0.0083 ± 0.0092` (4 wins/1 loss), Progress `+0.0077 ± 0.0164`
(3/2), RA-STCH `+0.0025 ± 0.0240` (4/1), GradNorm `+0.0025 ± 0.0138`
(3/2), PCGrad `+0.0007 ± 0.0129` (3/2), CAGrad `+0.0043 ± 0.0194`
(3/2), and DB-MTL `+0.0198 ± 0.0377` (2/3). These are descriptive
five-fold results, not a basis for post-hoc method elimination.

Selected Parkinson tradeoff: on CARE TEST, DB-MTL has Macro Recall
`0.7930 ± 0.0512`, Patient Recall `0.6823 ± 0.1165`, and Healthy Recall
`0.9035 ± 0.0588`; Progress has Macro Recall `0.7865 ± 0.0594`, Patient
Recall `0.6118 ± 0.1237`, and Healthy Recall `0.9613 ± 0.0139`. On WSM TEST,
DB-MTL has Macro Recall `0.7977 ± 0.0420`, Patient Recall `0.7474 ± 0.1057`,
and Healthy Recall `0.8480 ± 0.0564`. The raw summaries provide the complete
fold-by-fold six-metric tables for every population and method.

## Integrity and run notes

- All methods used fixed canonical folds and the same CARE+WSM data contract.
- Selection used validation only; TEST values above are evidence only.
- No feature artifact was re-extracted or modified.
- Initial STCH fold 0 and initial CAGrad fold 0 attempts exposed implementation
  issues before valid completion; both were corrected and rerun. The failed
  attempts are not counted in the 40 runs. A duplicate CAGrad launcher was
  interrupted before completion and is also not counted.
- The standard artifacts do not contain `mtl/*` controller/gradient
  diagnostics after completion. This is recorded as an evidence limitation;
  no diagnostic values are fabricated. The method invariants are covered by
  the targeted unit tests and the production runs completed without failures.
- Runtime and hardware details remain in each `train.log` and MLflow run.

