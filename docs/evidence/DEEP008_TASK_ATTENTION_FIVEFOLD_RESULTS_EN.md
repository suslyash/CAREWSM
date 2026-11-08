# DEEP-008 Task-Specific Attention: Five-Fold CUDA Results

## Outcome

The authorized five-fold `carewsm_gated_task_attention` experiment completed
successfully through the existing Chimera-ML launcher. The validation result is
negative against the pre-registered decision rules:

| Configuration | VAL Macro Recall mean | Sample SD | Difference vs weighted-F |
|---|---:|---:|---:|
| Weighted-F gated fusion | 0.82706 | 0.02788 | reference |
| Task-specific gated attention | 0.82298 | 0.03091 | -0.00408 |

Task-specific attention does not meet the required `+0.010` improvement and is
not promoted on this evidence. Automatic TEST outputs were excluded from the
report and were not used for this decision.

## Source, environment and fixed protocol

- Required source commit: `b2684bb04af2181b5b3eb7ec223c2f6b37c98386`.
- Experiment branch: `codex/deep008-task-attention-fivefold-results`.
- Python: 3.12.3.
- PyTorch: 2.11.0+cu128; CUDA runtime 12.8.
- GPU: NVIDIA GeForce RTX 4080, 16 GB.
- Model: `carewsm_gated_task_attention`.
- Features: existing native `emotionmsp_embeds` and `trillsson4_1` only.
- Protocol: fixed joint CARE+WSM folds, seed 5, hidden dimension 64, dropout
  0.1, batch size 64, AdamW with learning rate 0.0001, fold-specific
  TRAIN-derived Parkinson weights, weighted Parkinson BCE, 50 epoch maximum,
  patience 7, and validation-only
  `val_mean_corpora_tasks/macro_recall` selection.

The five configurations were:

```text
configs/audio_representation_pilot/deep008/task_attention/F_task_attention_fold0.yaml
configs/audio_representation_pilot/deep008/task_attention/F_task_attention_fold1.yaml
configs/audio_representation_pilot/deep008/task_attention/F_task_attention_fold2.yaml
configs/audio_representation_pilot/deep008/task_attention/F_task_attention_fold3.yaml
configs/audio_representation_pilot/deep008/task_attention/F_task_attention_fold4.yaml
```

## Run identity and selected checkpoints

Each fold has one completed runtime directory, one matching MLflow run, one
selected non-`last.pt` checkpoint, and a `Training complete` marker.

| Fold | Runtime directory | MLflow run ID | Selected epoch | Selected checkpoint | VAL monitor |
|---:|---|---|---:|---|---:|
| 0 | `F_task_attention_fold0_2026-11-08_14-09_carewsm_gated_task_attention_91c3a6f8` | `c2c4dae180bd4d21936a8182bb723c56` | 11 | `epoch=11_val_mean_corpora_tasks_macro_recall=0.8274.pt` | 0.8274 |
| 1 | `F_task_attention_fold1_2026-11-08_14-11_carewsm_gated_task_attention_f781b17b` | `d6053a4ab09342fab64a5a871e6bbebb` | 20 | `epoch=20_val_mean_corpora_tasks_macro_recall=0.8465.pt` | 0.8465 |
| 2 | `F_task_attention_fold2_2026-11-08_14-14_carewsm_gated_task_attention_92897ddc` | `0d8123e34f0745d58660f873ab0dccf2` | 9 | `epoch=9_val_mean_corpora_tasks_macro_recall=0.7827.pt` | 0.7827 |
| 3 | `F_task_attention_fold3_2026-11-08_14-16_carewsm_gated_task_attention_81098860` | `896e3ae6b0354c65a7094df5b59b07b1` | 13 | `epoch=13_val_mean_corpora_tasks_macro_recall=0.8013.pt` | 0.8013 |
| 4 | `F_task_attention_fold4_2026-11-08_14-19_carewsm_gated_task_attention_201c04de` | `3ef2e7d7eb1e476ba1aefa660677dfcf3` | 18 | `epoch=18_val_mean_corpora_tasks_macro_recall=0.8570.pt` | 0.8570 |

The runtime parent was
`logs/audio_representation_pilot/deep008_task_attention/audio_representation_pilot_deep008_task_attention/`.
The corresponding historical weighted-F runtime directories were the five
`F_weighted_parkinson_fold{0..4}_*` directories under
`logs/audio_representation_pilot/deep008_weighted_f/`.

## Foldwise validation comparison

The baseline column is the selected weighted-F VAL monitor from the matching
historical fold. Differences are task-attention minus weighted-F.

| Fold | Attention | Weighted-F | Difference | Result |
|---:|---:|---:|---:|---|
| 0 | 0.8274 | 0.8321 | -0.0047 | loss |
| 1 | 0.8465 | 0.8400 | +0.0065 | win |
| 2 | 0.7827 | 0.7979 | -0.0152 | loss |
| 3 | 0.8013 | 0.8010 | +0.0003 | win |
| 4 | 0.8570 | 0.8643 | -0.0073 | loss |

There were 2 fold wins and 3 fold losses. The paired mean difference was
`-0.00408`; the comparison is descriptive and does not establish statistical
significance.

## Corpus/task validation components

These are the four components of the canonical validation objective. Values are
five-fold means; differences are task-attention minus weighted-F.

| Corpus/task | Attention Macro Recall | Weighted-F Macro Recall | Difference |
|---|---:|---:|---:|
| CARE Depression | 0.83038 | 0.82642 | +0.00396 |
| CARE Parkinson | 0.83362 | 0.82270 | +0.01092 |
| WSM Depression | 0.72754 | 0.75624 | -0.02870 |
| WSM Parkinson | 0.90038 | 0.90276 | -0.00238 |

The component acceptance rule fails because WSM Depression decreases by more
than 0.020 absolute. CARE improves for both tasks, but that gain is outweighed
by the WSM Depression regression in the joint objective.

## Patient and Healthy Recall

The following table reports the selected validation checkpoint values for every
fold, corpus and task. Patient is the positive-class recall; Healthy is the
negative-class recall.

| Fold | Corpus/task | Macro Recall | Patient Recall | Healthy Recall |
|---:|---|---:|---:|---:|
| 0 | CARE Depression | 0.8145 | 0.6628 | 0.9662 |
| 0 | CARE Parkinson | 0.7774 | 0.6176 | 0.9371 |
| 0 | WSM Depression | 0.7903 | 0.7480 | 0.8327 |
| 0 | WSM Parkinson | 0.9274 | 0.9476 | 0.9071 |
| 1 | CARE Depression | 0.9081 | 0.8791 | 0.9370 |
| 1 | CARE Parkinson | 0.8909 | 0.7949 | 0.9869 |
| 1 | WSM Depression | 0.7057 | 0.6638 | 0.7475 |
| 1 | WSM Parkinson | 0.8815 | 0.8333 | 0.9297 |
| 2 | CARE Depression | 0.7160 | 0.5287 | 0.9032 |
| 2 | CARE Parkinson | 0.8447 | 0.7419 | 0.9474 |
| 2 | WSM Depression | 0.6801 | 0.5669 | 0.7932 |
| 2 | WSM Parkinson | 0.8899 | 0.9563 | 0.8235 |
| 3 | CARE Depression | 0.8605 | 0.8415 | 0.8795 |
| 3 | CARE Parkinson | 0.7193 | 0.4688 | 0.9699 |
| 3 | WSM Depression | 0.7013 | 0.7471 | 0.6555 |
| 3 | WSM Parkinson | 0.9239 | 0.9518 | 0.8960 |
| 4 | CARE Depression | 0.8528 | 0.7556 | 0.9500 |
| 4 | CARE Parkinson | 0.9358 | 0.9412 | 0.9304 |
| 4 | WSM Depression | 0.7603 | 0.6424 | 0.8782 |
| 4 | WSM Parkinson | 0.8792 | 0.8640 | 0.8944 |

Five-fold mean recall breakdowns were:

| Corpus/task | Patient attention | Patient weighted-F | Difference | Healthy attention | Healthy weighted-F | Difference |
|---|---:|---:|---:|---:|---:|---:|
| CARE Depression | 0.73354 | 0.74510 | -0.01156 | 0.92718 | 0.90770 | +0.01948 |
| CARE Parkinson | 0.71288 | 0.69884 | +0.01404 | 0.95434 | 0.94660 | +0.00774 |
| WSM Depression | 0.67364 | 0.73770 | -0.06406 | 0.78142 | 0.77480 | +0.00662 |
| WSM Parkinson | 0.91060 | 0.92026 | -0.00966 | 0.89014 | 0.88528 | +0.00486 |

The WSM Depression regression is primarily a Patient Recall regression, while
Healthy Recall increases slightly. This is consistent with the class-imbalance
trade-off visible in the component results.

## Acceptance decision and interpretation

The task-attention model does not satisfy the pre-registered acceptance rules:

1. Required mean improvement: `>= +0.010`; observed: `-0.00408` — **fail**.
2. Maximum allowed decrease in any component: `0.020`; WSM Depression change:
   `-0.02870` — **fail**.
3. Patient and Healthy Recall are reported above; WSM Depression Patient Recall
   decreases by `0.06406` on the five-fold mean.

CARE is the only corpus in which multiple turns are available for direct
attention-based selection. CARE Depression and Parkinson Macro Recall improve
by `+0.00396` and `+0.01092`, respectively. WSM records have one valid turn,
so task-specific attention cannot select among turns there; its WSM results are
effectively a learned one-turn transformation. The joint objective therefore
does not support promoting this architecture.

## Limitations and next research recommendation

- This is a five-fold validation comparison with the fixed seed and prescribed
  controls; no significance test or seed-variance study was authorized.
- The report excludes automatically generated TEST metrics and does not use
  them for selection or acceptance.
- Runtime logs, MLflow SQLite data, checkpoints, and private feature artifacts
  remain local and are not committed.
- No further experiment is recommended from this task. Any follow-up, such as
  diagnosing WSM Depression behavior or a CARE-focused analysis, requires a new
  manager-authored authorization and must preserve the validation-first policy.
