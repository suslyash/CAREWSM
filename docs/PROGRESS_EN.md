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

REFACTOR-002 corpus-specific masked metrics and validation monitor submitted for independent manager review. Legacy loader names, metrics, summary callback, and historical monitor remain compatible; YAML migration and final logging changes are deferred to later refactor tasks.

The next scientific question is whether the strongest static deep embeddings are complementary when combined.

The immediate task does **not** run the full combination study. It first makes multi-feature loading/fusion explicit, testable and coverage-aware.

## Next task

REFACTOR-002 — corpus-specific masked metrics and validation monitor submitted for independent manager review.

TASK-011 remains PAUSED until REFACTOR-000 through REFACTOR-005 and final integration review are independently reviewed and merged. The historical 110 runs remain preserved.
