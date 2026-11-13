# Interpretable Audio Feature Fusion

## Outcome

The production study completed all 25 planned real-GPU runs: five feature-family combinations (B, BE, BP, BA, BEPA) across folds 0--4. The final ledger contains 25 COMPLETED, 0 FAILED, and 0 RUNNING slots. The prior 145-run evidence set was not modified or rerun.

The validation-selected combination is **B** (the four established ABCD families), with validation overall Macro Recall **0.832530 +/- 0.038701** across five folds. Selection used validation only; TEST is descriptive.

## Cohort and protocol

The matched complete-case cohort requires all seven audited families: emotionmsp_embeds, emotionmsp_trajectories, trillsson4_1, wav2vec_xlsr300m, egemaps_func, disvoice_phonation_static, and disvoice_articulation_static. The audit passed with 10,273 complete-case rows from 10,464 manifest rows. Production training used the same cohort across all five combinations, independent per-family mean pooling, fixed seed 5, fixed folds, the existing gated hidden-64 architecture, Equal MTL loss, and validation-selected checkpoints.

CARE/WSM are mixed for training. Validation and TEST remain corpus-specific. The tracked aggregate CSV reports every required metric as fold mean and sample SD; no TEST value was used for selection.

## Feature-family study

| Combination | Families added to B | VAL overall Macro Recall mean +/- sample SD | TEST overall Macro Recall mean +/- sample SD |
|---|---|---:|---:|
| B | none | 0.832530 +/- 0.038701 | 0.788340 +/- 0.023783 |
| BE | eGeMAPS | 0.824495 +/- 0.040237 | 0.792500 +/- 0.010594 |
| BP | phonation | 0.827735 +/- 0.037216 | 0.804125 +/- 0.013541 |
| BA | articulation | 0.818490 +/- 0.032637 | 0.774710 +/- 0.011964 |
| BEPA | eGeMAPS + phonation + articulation | 0.825515 +/- 0.043365 | 0.789420 +/- 0.028891 |

Permutation analysis was run only on the five validation-selected B checkpoints, with ten within-corpus repeats per family. Results are in  through ; these are exploratory feature-importance evidence, not clinical attribution.

## Eight-method MTL comparison

The following table preserves the existing historical comparison: Equal, PCGrad, STCH, and DBMTL come from the prior ABCD matrix; Progress, RA-STCH, GradNorm, and CAGrad come from the completed MTL evidence set. Values are five-fold mean +/- sample SD. The current 25-run study is an Equal-loss feature-family extension and is reported separately above.

| Method | VAL Macro Recall | TEST Macro Recall |
|---|---:|---:|
| Equal | 0.830855 +/- 0.040364 | 0.782235 +/- 0.022355 |
| PCGrad | 0.827790 +/- 0.039442 | 0.782115 +/- 0.022468 |
| STCH | 0.827730 +/- 0.040654 | 0.784190 +/- 0.017373 |
| DBMTL | 0.819670 +/- 0.031847 | 0.787245 +/- 0.022920 |
| Progress | 0.829555 +/- 0.041119 | 0.782230 +/- 0.024072 |
| RA-STCH | 0.826630 +/- 0.039230 | 0.783740 +/- 0.014672 |
| GradNorm | 0.826665 +/- 0.043174 | 0.783310 +/- 0.020860 |
| CAGrad | 0.825060 +/- 0.040356 | 0.775135 +/- 0.019350 |

## Reproducibility and exclusions

The run ledger, resolved YAML configurations, source/config hashes, selected checkpoints, fold-level CSV, aggregate CSV, cohort audit, feature dictionary, and permutation CSVs provide traceability. Checkpoints and MLflow databases remain outside the commit. No raw identifiers are added to the report artifacts. A separate one-epoch CUDA smoke run was completed before production.

A dynamic/turn-aware extension is not implemented here. It should be proposed as a separate atomic task after review of this static complete-case evidence.
