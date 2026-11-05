# CAREWSM Active Progress

## Current state

The first complete joint CARE+WSM single-feature audio matrix has been reviewed and merged.

Manager-reviewed evidence:

- 22 configurations;
- 110 canonical fold-runs;
- five folds per configuration;
- checkpoint selection from VAL `joint/average_macro_f1` only (historical contract; preserved, not current policy);
- CARE TEST and WSM TEST retained as descriptive statistics;
- all six required metrics logged for VAL and TEST;
- one historical duplicate excluded deterministically;
- reproducible analysis script and fold-level derived evidence committed.

Evidence:

- `docs/evidence/MANAGER_AUDIO_EXPERIMENT_REPORT_2026-11-04.md`
- `docs/evidence/MANAGER_AUDIO_EXPERIMENT_FOLD_RESULTS_2026-11-04.csv`
- `scripts/analyze_audio_experiments.py`

## Validation-first findings

Top configurations by mean five-fold VAL monitor:

1. `emotionmsp_embeds/mlp` — 0.7941
2. `emotionmsp_embeds/mdformer` — 0.7882
3. `trillsson4_1/mlp` — 0.7621
4. `trillsson4_1/mdformer` — 0.7559
5. `wav2vec_xlsr300m/mlp` — 0.7118
6. `wav2vec_xlsr300m/mdformer` — 0.6668

The strongest single-feature representations are therefore the static deep embeddings:

- `emotionmsp_embeds`;
- `trillsson4_1`;
- `wav2vec_xlsr300m`.

The strongest handcrafted static family is `disvoice_articulation_static/mlp` at 0.6186 VAL monitor.

Several weaker configurations exhibit majority-class collapse with near-zero Patient Recall. Weighted metrics must not be used alone to judge them.

TSCMamba was not evaluated because `mamba_ssm` is unavailable; it did not experimentally lose.

## Current methodology

Primary protocol remains joint CARE+WSM five-fold training.

For fold `k`:

- TRAIN = CARE fold-k TRAIN + WSM fold-k TRAIN;
- VAL = CARE fold-k VAL + WSM fold-k VAL;
- preprocessing and class weights use TRAIN only;
- checkpoint/configuration selection uses VAL only;
- CARE/WSM TEST are logged only for statistics.

Required metrics:

- Macro F1;
- Weighted F1;
- Macro Recall;
- Weighted Recall;
- Patient Recall;
- Healthy Recall.

## Current stage

REFACTOR-000 through REFACTOR-007 are complete and merged. LAUNCH-001 completed one successful real CUDA training epoch. FINAL-INTEGRATION-001 is complete and merged through PR #40 with two non-blocking cleanup findings and an approved PASS result. The active audio-only Chimera pipeline and canonical 110-config matrix are integration-ready. REFACTOR-007 preserves the validation-only `dev_mean_macro_recall` selection policy, adds equal-weight VAL/TEST averages, removes obsolete support/count outputs from new logs and summaries, and preserves the 110 canonical configurations under `configs/audio_feature_selection/`. No real training run after REFACTOR-007 has been performed.

No new experiments, cleanup implementation, or research tasks are authorized by the completed audit. Historical evidence and the original 110 runs remain preserved.

## Next task

No next implementation task is authorized. Further research work requires a new owner-approved task.

TASK-011 remains PAUSED.
