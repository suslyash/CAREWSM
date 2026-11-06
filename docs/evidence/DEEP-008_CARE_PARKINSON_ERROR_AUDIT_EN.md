# DEEP-008 CARE Parkinson Validation Error Audit

## 1. Scope and reproducibility

This is a read-only validation audit. It used only the ten existing selected
DEEP-008 checkpoints for A and F, on CARE validation folds 0--4. No training,
TEST inference, threshold tuning, checkpoint replacement, MLflow modification,
or split modification was performed.

- Required base: `61782a766f49d01a38ad69493c68ffefaa41041d`
- Audit branch: `codex/deep-008-care-parkinson-error-audit`
- Audit source commit before this report: `61782a766f49d01a38ad69493c68ffefaa41041d`
- Diagnostic: `scripts/audit_deep008_care_parkinson.py`
- Aggregate output: `logs/audio_representation_pilot/deep008_error_audit/audit.json`
- Device: CPU inference; existing checkpoints were loaded unchanged

The script constructed CARE validation records from the canonical fold files and
CARE `dev.csv`/`train.csv` metadata only. It did not read CARE TEST metadata or
features. It used the active Dataset, native feature ordering, sequence masks,
model definitions, output column 1, `logit > 0`, and observed Parkinson labels.
The JSON output contains aggregate statistics only; no sample identifiers,
labels, predictions, or person-level records are persisted.

## 2. Confusion matrices

Each row is an independently recomputed CARE Parkinson validation result. Patient
means and Healthy means are the corresponding class recalls; Macro Recall is
their arithmetic mean. The selected checkpoint is named by its existing run
directory and epoch; no checkpoint was changed.

| Fold | Model/checkpoint epoch | Observed patients | Observed healthy | TP | FN | TN | FP | Patient Recall | Healthy Recall | Macro Recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A / 15 | 34 | 318 | 16 | 18 | 300 | 18 | 0.4706 | 0.9434 | 0.7070 |
| 0 | F / 14 | 34 | 318 | 13 | 21 | 307 | 11 | 0.3824 | 0.9654 | 0.6739 |
| 1 | A / 6 | 39 | 306 | 26 | 13 | 294 | 12 | 0.6667 | 0.9608 | 0.8137 |
| 1 | F / 2 | 39 | 306 | 18 | 21 | 304 | 2 | 0.4615 | 0.9935 | 0.7275 |
| 2 | A / 29 | 31 | 304 | 18 | 13 | 299 | 5 | 0.5806 | 0.9836 | 0.7821 |
| 2 | F / 23 | 31 | 304 | 19 | 12 | 292 | 12 | 0.6129 | 0.9605 | 0.7867 |
| 3 | A / 23 | 32 | 299 | 10 | 22 | 293 | 6 | 0.3125 | 0.9799 | 0.6462 |
| 3 | F / 17 | 32 | 299 | 11 | 21 | 289 | 10 | 0.3438 | 0.9666 | 0.6552 |
| 4 | A / 16 | 34 | 316 | 27 | 7 | 281 | 35 | 0.7941 | 0.8892 | 0.8417 |
| 4 | F / 23 | 34 | 316 | 32 | 2 | 298 | 18 | 0.9412 | 0.9430 | 0.9421 |

The recomputed Macro Recall values match the existing DEEP-008 five-fold
evidence to the logged four-decimal precision for all ten runs. No discrepancy
was found.

Across folds, A had 170 observed patients and 1,543 observed healthy samples,
with 97 TP, 73 FN, 1,467 TN and 76 FP. F had the same denominators, with 93 TP,
77 FN, 1,490 TN and 53 FP. The aggregate recalls were A Patient 0.5706,
Healthy 0.9507; F Patient 0.5471, Healthy 0.9657.

## 3. Paired A/F CARE validation errors

The paired comparison used identical CARE validation sample populations within
each fold and retained only aggregate counts:

| Paired outcome | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Patients correctly identified by both | 13 | 17 | 15 | 7 | 27 | 79 |
| Patients missed by both | 18 | 12 | 9 | 18 | 2 | 59 |
| Patients detected only by A | 3 | 9 | 3 | 3 | 0 | 18 |
| Patients detected only by F | 0 | 1 | 4 | 4 | 5 | 14 |
| Healthy false positives for both | 4 | 2 | 3 | 4 | 15 | 28 |
| Healthy false positives only by A | 14 | 10 | 2 | 2 | 20 | 48 |
| Healthy false positives only by F | 7 | 0 | 9 | 6 | 3 | 25 |
| Healthy false positive for either model | 25 | 12 | 14 | 12 | 38 | 101 |

The paired patient errors show substantial shared difficulty: 59 of 170
observed patients were missed by both models. A detected 18 patients that F
missed, while F detected 14 that A missed. F’s lower FP total is consistent with
its higher Healthy Recall, but it does not imply better CARE Parkinson patient
detection in aggregate.

The smallest CARE Parkinson patient denominators were 31 on fold 2 and 32 on
fold 3; the other folds had 34, 39 and 34 observed patients. Thus fold-level
recall is sensitive to a small number of cases. The lowest Patient Recall for
both models occurred on fold 3: A 0.3125 and F 0.3438.

## 4. Turn-sequence length diagnostic

Sequence length was summarized only in aggregate over observed CARE Parkinson
patients. Across all folds, A’s mean native sequence length was 1.61 turns for
true positives and 2.55 turns for false negatives. F’s corresponding means were
1.71 and 2.38 turns. These descriptive differences do not establish a length
effect: the false-negative group is small, sequence lengths are uneven across
folds, and the models use different input representations. No sample-level
length or prediction data were retained in the report or audit output.

## 5. Architectural and domain interpretation

The existing five-fold evidence shows that F’s improvement over A is
concentrated in WSM Parkinson: mean WSM Parkinson Macro Recall is F 0.8936
versus A 0.8471, a +0.0465 difference. CARE Parkinson is effectively unchanged:
F 0.7571 versus A 0.7581. The other component means are CARE Depression A/F
0.8428/0.8381 and WSM Depression 0.7594/0.7446.

The audit supports these cautious hypotheses:

- Native CARE Parkinson patient cases contain shared representation difficulty;
  most missed patients are missed by both models.
- F is more conservative on CARE: it produces fewer healthy false positives and
  therefore higher Healthy Recall, but this conservatism does not improve its
  aggregate CARE Parkinson Patient Recall.
- The strong F gain on WSM Parkinson but not CARE Parkinson suggests a
  corpus-dependent interaction between the fused representation and disease
  cues. Gated fusion may help when the two feature families provide compatible
  complementary evidence, while not resolving CARE-specific patient variation.
- CARE Parkinson class imbalance is material: the observed healthy denominator
  is about nine times the patient denominator in the combined audit. This makes
  Healthy Recall comparatively stable and patient errors disproportionately
  influential in Macro Recall.

These are diagnostic hypotheses, not causal conclusions. In particular, F also
changes capacity and architecture relative to historical fusion references.

## 6. Bounded future experiment recommendation

If authorized separately, run one pre-registered five-fold comparison of F with
the existing architecture and a TRAIN-only Parkinson `pos_weight` in the
existing masked weighted-BCE option, preserving seed, folds, validation monitor,
and all other settings. Report CARE and WSM Parkinson Patient/Healthy Recall
separately. This directly tests the class-imbalance hypothesis suggested by the
shared false negatives and high CARE Healthy Recall; it was **not executed** in
this audit.
