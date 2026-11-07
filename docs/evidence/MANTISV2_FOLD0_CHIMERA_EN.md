# MantisV2 fold-0 Chimera integration

## Outcome

The complete fold-0 experiment ran through the existing Chimera-ML pipeline on
`codex/mantisv2-chimera-fold0`. No standalone trainer, metrics, logger, or
checkpoint manager was used. The rejected
`scripts/run_mantisv2_fold0_scored_pilot.py` remains absent.

## Frozen cache

- Package: `mantis-tsfm==1.1.0`
- Checkpoint: `paris-noah/MantisV2`, revision
  `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`
- Checkpoint SHA-256: `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`
- Layer/token: transformer layer 2, combined token
- Source: 25 canonical eGeMAPS LLD channels
- Temporal contract: 1.0-second windows, 0.5-second hop, 95 consecutive
  valid frames, linear interpolation to 512 points without extrapolation,
  first and last eligible windows, mean pooling, flattened 12,800 features
- Cache: `/media/maxim/Databases/CARE/mantisv2_cache/fold0_validated3/`
- Artifacts: 10,448 finite NPZ representations; 496.4 MB
- Generation throughput: 157.7 seconds from first to last artifact, approximately 66.2 records/second
- Coverage: CARE train 1,361, CARE val 352, CARE test 318; WSM train 5,627,
  WSM val 1,427, WSM test 1,363
- Policy C exclusions: CARE 37 terminal frames across 35 recordings; WSM 0

The builder performs full fail-closed temporal validation before inference.
Only the authorized CARE terminal-frame exclusion is accepted. Cache reuse
requires the manifest contract, matching checkpoint/schema parameters, an
existing valid NPZ, shape `(1, 12800)`, and finite values. Validation/temporal
preparation uses eight spawned CPU workers; frozen Mantis inference remains in
the parent process.

A synthetic sentinel test confirms that changing the excluded terminal frame
to an extreme finite value does not alter the retained tensor. The indexed
selector was checked against the prior window interpolation behavior on a
representative sequence; the manager-authorized first/last policy is used.

## Chimera run

- YAML: `configs/audio_representation_pilot/mantisv2/fold0.yaml`
- Budget: 20 epochs; completed all 20
- MLflow run ID: `d5cb0760f3e74c0a969fbfc704e6ba3f`
- Runtime: 93 seconds (11:11:47–11:13:20)
- Native counts: TRAIN 6,988; CARE VAL 352; WSM VAL 1,427; CARE TEST 318;
  WSM TEST 1,363
- Selection monitor: `val_mean_corpora_tasks/macro_recall`
- Best epoch: 19 (1-based), selection score 0.639550
- Checkpoint: `logs/audio_representation_pilot/mantisv2/audio_representation_pilot_mantisv2/mantisv2_egemaps_mlp_fold0_2026-11-07_11-11_carewsm_pooled_mlp_9ff15dc3/checkpoints/epoch=19_val_mean_corpora_tasks_macro_recall=0.6395.pt`

### Best-checkpoint metrics

Values are `Macro F1 / Weighted F1 / Macro Recall / Weighted Recall / Patient Recall / Healthy Recall`. TEST is descriptive only.

| Split | Corpus | Depression | Parkinson |
|---|---|---|---|
| VAL | CARE | 0.7368 / 0.8034 / 0.7425 / 0.8011 / 0.6279 / 0.8571 | 0.5866 / 0.8652 / 0.5762 / 0.8750 / 0.2059 / 0.9465 |
| VAL | WSM | 0.6358 / 0.6400 / 0.6423 / 0.6377 / 0.6712 / 0.6134 | 0.5947 / 0.6152 / 0.5972 / 0.6304 / 0.4017 / 0.7926 |
| TEST | CARE | 0.6547 / 0.7225 / 0.6438 / 0.7421 / 0.3918 / 0.8959 | 0.6673 / 0.8688 / 0.6783 / 0.8648 / 0.4412 / 0.9155 |
| TEST | WSM | 0.5449 / 0.5531 / 0.5509 / 0.5490 / 0.5612 / 0.5407 | 0.5553 / 0.6878 / 0.5533 / 0.7108 / 0.2406 / 0.8660 |

## Verification

- Mantis YAML validation: PASS.
- Historical baseline YAML validation: PASS.
- Python compilation: PASS.
- Synthetic sentinel and temporal selector checks: PASS.
- `git diff --check`: PASS.
- Standard train log, summary, checkpoint, and MLflow artifacts: present.
