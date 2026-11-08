# DEEP-008 Next-Stage Diagnostics and Batch 02 Proposal

## 1. Scope and decision boundary

This diagnostic was performed from `c19b2c24441fd5268cbd6defcb7e509134e56aa2`
on TRAIN and fold-VALIDATION populations only. It uses the existing complete-
case `emotionmsp_embeds + trillsson4_1` population, the existing native
turn-level artifacts, the fixed five folds, and the existing Chimera-ML data
and model path.

No CUDA run, training, TEST inspection, TEST inference, TEST aggregate
recalculation, feature extraction, audio access, split change, or model
implementation was performed. Existing fixed weighted-F and DB-MTL
checkpoints were loaded on CPU only for the exploratory VAL ensemble audit.
No sample identifiers, embeddings, predictions, or private paths are stored in
this report.

The purpose is diagnosis and a concrete, reviewable Batch 02 proposal. It does
not authorize Batch 02 execution.

## 2. Verified TRAIN/VAL population audit

Counts below are complete-case records for both EmotionMSP and TRILLsson. Each
cell is `record count / unique profile count`; label cells are
`positive, negative, missing`. Profile overlap is zero for every corpus/fold
TRAIN–VAL comparison.

| corpus | fold | TRAIN records / profiles | VAL records / profiles | Depression TRAIN | Depression VAL | Parkinson TRAIN | Parkinson VAL | TRAIN/VAL profile overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CARE | 0 | 1361 / 191 | 352 / 49 | 350,1011,0 | 86,266,0 | 136,1225,0 | 34,318,0 | 0 |
| CARE | 1 | 1368 / 197 | 345 / 43 | 345,1023,0 | 91,254,0 | 131,1237,0 | 39,306,0 | 0 |
| CARE | 2 | 1378 / 190 | 335 / 50 | 349,1029,0 | 87,248,0 | 139,1239,0 | 31,304,0 | 0 |
| CARE | 3 | 1382 / 190 | 331 / 50 | 354,1028,0 | 82,249,0 | 138,1244,0 | 32,299,0 | 0 |
| CARE | 4 | 1363 / 192 | 350 / 48 | 346,1017,0 | 90,260,0 | 136,1227,0 | 34,316,0 | 0 |
| WSM | 0 | 5637 / 511 | 1429 / 127 | 1377,2027,2233 | 369,508,552 | 934,1299,3404 | 229,323,877 | 0 |
| WSM | 1 | 5659 / 511 | 1407 / 127 | 1401,2028,2230 | 345,507,555 | 935,1295,3429 | 228,327,852 | 0 |
| WSM | 2 | 5667 / 509 | 1399 / 129 | 1402,2032,2233 | 344,503,552 | 934,1299,3434 | 229,323,847 | 0 |
| WSM | 3 | 5638 / 512 | 1428 / 126 | 1402,2027,2209 | 344,508,576 | 914,1295,3429 | 249,327,852 | 0 |
| WSM | 4 | 5663 / 509 | 1403 / 129 | 1402,2026,2235 | 344,509,550 | 935,1300,3428 | 228,322,853 | 0 |

### Observed imbalance and availability

- CARE contributes 1,713 complete-case records per fold partition pair and has
  no missing target labels in this population. TRAIN Depression positive to
  negative ratios are 0.337–0.346; Parkinson ratios are 0.106–0.112.
- WSM contributes 7,066 complete-case records per fold partition pair. The
  three unavailable WSM records are outside the complete-case population.
  TRAIN Depression positive-to-negative ratios are 0.679–0.692, with roughly
  39% missing Depression labels. TRAIN Parkinson ratios are 0.706–0.722,
  with roughly 60% missing Parkinson labels.
- WSM therefore supplies about four times as many complete-case records as
  CARE and dominates joint record counts, while its observed-label mixture is
  substantially different. This is verified evidence consistent with a
  corpus/label-distribution shift; it does not by itself prove the cause of
  the validation plateau.
- All five folds preserve zero TRAIN–VAL profile overlap in both corpora.
  Fold counts vary modestly, but no fold has a missing target class in the
  reported complete-case populations.

## 3. Existing `wav2vec_xlsr300m` compatibility audit

The live registry declares `wav2vec_xlsr300m` as an audio NPZ
`turn_embedding` with `mean`, `std`, and ordered `turn_ids`. The selected
`mean` array is 1,024-dimensional. The same schema is used by the existing
EmotionMSP and TRILLsson native turn artifacts.

TRAIN/VAL artifact checks found:

- CARE: 1,713/1,713 complete-case records per fold pair; all artifacts were
  readable, finite, and 1,024-dimensional.
- WSM: 7,066/7,069 source records per fold pair were manifest-valid and
  readable; the same three records are unavailable in every fold pair.
- Every manifest-valid artifact checked had the expected NPZ keys and valid
  ordered turn IDs.
- Ordered turn IDs matched exactly across EmotionMSP, TRILLsson, and wav2vec
  for all checked TRAIN/VAL records. Adding wav2vec would therefore preserve
  native turn-level complete-case coverage rather than introduce additional
  filtering.

This is a valid native turn-aligned three-family fusion candidate. It is not a
frame-synchronous or timestamped temporal representation: the artifacts
contain ordered turn identifiers, not a defensible common 1-second/0.5-second
axis. A future experiment may concatenate aligned native turns through the
existing Dataset contract, but must not invent timestamps, repeat turns, or
claim frame-level synchronization. Mantis temporalization remains unrelated
and deferred.

The previous validation-only native fusion matrix already included wav2vec
pairwise and three-family combinations. It did not establish a promotion case
over the current EmotionMSP+TRILLsson gated-fusion reference; for example,
the reported three-family pooled-MLP and MD-Former means remained below the
current approximately 0.83 gated-fusion VAL reference. Accordingly, wav2vec
is compatible but is not prioritized as a primary Batch 02 hypothesis.

## 4. Existing-model complementarity on VAL

The existing selected checkpoints were used without changing epochs, weights,
thresholds, or populations. Weighted-F selected epochs were 14, 3, 11, 14,
and 7. DB-MTL selected epochs were 11, 12, 14, 9, and 28. The fixed ensemble
was computed as `0.5 * weighted_f_logits + 0.5 * dbmtl_logits`, with
`logit > 0` as the unchanged decision rule.

The four component order is CARE Depression, CARE Parkinson, WSM Depression,
WSM Parkinson. Each table entry is Macro Recall/UAR.

| fold | Weighted-F | DB-MTL | fixed ensemble |
|---:|---:|---:|---:|
| 0 | 0.8203 / 0.7921 / 0.7888 / 0.9271 | 0.8502 / 0.7849 / 0.8106 / 0.8202 | 0.8283 / 0.8105 / 0.8101 / 0.9060 |
| 1 | 0.8825 / 0.8363 / 0.7521 / 0.8890 | 0.9028 / 0.9177 / 0.7682 / 0.8738 | 0.9166 / 0.9084 / 0.7663 / 0.8880 |
| 2 | 0.7304 / 0.8558 / 0.7073 / 0.8979 | 0.8429 / 0.8111 / 0.7276 / 0.8685 | 0.7674 / 0.8190 / 0.7170 / 0.8840 |
| 3 | 0.8506 / 0.7177 / 0.7164 / 0.9193 | 0.8686 / 0.6993 / 0.7448 / 0.8410 | 0.8687 / 0.6953 / 0.7270 / 0.8974 |
| 4 | 0.8483 / 0.9116 / 0.8166 / 0.8805 | 0.8714 / 0.9184 / 0.8084 / 0.9029 | 0.8868 / 0.8890 / 0.8279 / 0.9066 |

| method | overall VAL UAR mean ± sample SD | paired fold differences vs Weighted-F |
|---|---:|---|
| Weighted-F | 0.8270 ± 0.0279 | reference |
| DB-MTL | 0.8317 ± 0.0371 | -0.0156, +0.0256, +0.0147, -0.0126, +0.0110 |
| fixed ensemble | 0.8360 ± 0.0385 | +0.0067, +0.0298, -0.0010, -0.0039, +0.0133 |

### Aggregate VAL components

Each component is shown as `Macro Recall (Patient Recall / Healthy Recall)`;
means are arithmetic means over the five folds.

| method | CARE Depression | CARE Parkinson | WSM Depression | WSM Parkinson |
|---|---:|---:|---:|---:|
| Weighted-F | 0.8264 (0.7451 / 0.9077) | 0.8227 (0.6988 / 0.9466) | 0.7563 (0.7377 / 0.7748) | 0.9028 (0.9203 / 0.8853) |
| DB-MTL | 0.8672 (0.8679 / 0.8665) | 0.8262 (0.7434 / 0.9090) | 0.7719 (0.8003 / 0.7435) | 0.8613 (0.9197 / 0.8028) |
| fixed ensemble | 0.8536 (0.8174 / 0.8897) | 0.8244 (0.7099 / 0.9390) | 0.7697 (0.7772 / 0.7621) | 0.8964 (0.9309 / 0.8620) |

Aggregate prediction disagreement between Weighted-F and DB-MTL, among
observed labels and reported by the same four components, was 11.30% CARE
Depression, 7.89% CARE Parkinson, 13.06% WSM Depression, and 9.29% WSM
Parkinson. This verifies non-identical decisions and makes complementarity
plausible, but it does not prove that the ensemble will generalize: both
component checkpoints were selected on these VAL folds and the ensemble was
constructed post hoc.

## 5. Diagnosis of the plateau

Verified factors are:

1. The joint population is strongly corpus-imbalanced: WSM supplies most
   records, while CARE has complete labels and a very different class mixture.
2. WSM has substantial task-label missingness, especially Parkinson, so the
   shared representation receives an uneven supervised signal across corpus
   and task.
3. DB-MTL improves the five-fold VAL mean over Weighted-F by 0.0046 and has
   complementary predictions. The fixed ensemble reaches 0.8360, but this is
   exploratory post-selection evidence, not independent confirmation.
4. The native deep features are turn-aligned but not frame-timestamped. More
   temporal complexity cannot be justified by the wav2vec artifact schema
   alone.

The following remain hypotheses, not established causes: negative task
transfer in the shared trunk, corpus-specific decision-boundary mismatch, and
whether the fixed ensemble gain would survive a pre-registered jointly trained
model. The available evidence does not support another dropout, hidden-size,
or isolated loss-weight sweep.

## 6. Proposed Batch 02

The recommended batch has two independent hypotheses and 15 new fold-runs in
total. Existing Weighted-F and DB-MTL five-fold controls are reused; they are
not retrained.

### H1 — independent task models: test negative task transfer

- **Change:** retain the existing EmotionMSP+TRILLsson input and canonical
  fold protocol, but train one Depression-only and one Parkinson-only model per
  fold. Remove cross-task shared-loss interaction; keep the same gated-fusion
  representation, optimizer, seed, class-weight rules, epoch budget, and
  VAL-only checkpoint monitor where applicable.
- **Mechanism:** each task can optimize its representation without gradients
  from the other task, directly testing whether shared multitask learning is
  limiting CARE performance.
- **Confounders:** two models have a larger aggregate parameter/update budget;
  missing-label availability differs by task; the comparison must not treat an
  unobserved task as a negative label.
- **Count:** 10 new runs, two task-specific runs × five folds.
- **Minimal scope:** add only explicit Chimera-ML single-task model/config
  support if the existing interfaces cannot express it; preserve Dataset,
  DataModule, metrics, logging, and checkpoint conventions. No custom training
  loop.
- **Acceptance:** task-specific five-fold VAL UAR must improve by at least
  0.010 for the target task over the corresponding DB-MTL task mean, with no
  CARE or WSM component deterioration greater than 0.020 and no overall
  four-component mean deterioration greater than 0.010. Report paired fold
  differences and patient/healthy recall explicitly.

### H2 — corpus-specialized output heads: test corpus shift

- **Change:** retain the shared gated-fusion trunk and both task branches, but
  use separate CARE and WSM output heads selected by corpus identity. The
  corpus identity is already available in the native DataModule metadata;
  there is no change to sample splits or labels.
- **Mechanism:** preserve cross-corpus representation learning while allowing
  corpus-specific calibration and decision boundaries for the verified class
  and missing-label differences.
- **Confounders:** additional head parameters, possible WSM-dominated shared
  gradients, and the fact that corpus identity may act as a shortcut. The
  implementation must keep profile isolation and must not leak validation
  corpus labels into preprocessing.
- **Count:** 5 new runs, one per fold.
- **Minimal scope:** one explicitly registered opt-in model variant using the
  existing Chimera Trainer/DataModule/loss/metrics. No new feature provider,
  temporal axis, or training loop.
- **Acceptance:** overall five-fold VAL UAR at least 0.010 above DB-MTL, CARE
  Depression and CARE Parkinson not each below DB-MTL by more than 0.020, WSM
  components not below DB-MTL by more than 0.020, and improvement in at least
  three paired folds. Patient and Healthy Recall must be reported by corpus
  and task.

### Deferred, not selected for Batch 02: wav2vec third-family fusion

The compatibility audit passes, so this is not technically blocked. It is not
selected because the existing validation matrix already contains native
wav2vec combinations without a convincing improvement over the current
reference, and the artifact has no stronger temporal semantics than the two
families already used. It can be reconsidered only after manager review of a
new rationale and a fixed feature-set comparison.

## 7. Required future evidence and controls

Both proposed hypotheses must use the same five fixed joint CARE+WSM folds,
participant/profile isolation, seed, optimizer, TRAIN-only class weights,
maximum epochs, patience, mixed precision, gradient clipping, and canonical
VAL monitor as the reused controls. Each future run must report the six
approved metrics for CARE and WSM VAL, native sample counts, four component
Macro Recalls, Patient Recall, Healthy Recall, paired fold differences, and
parameter/runtime footprint. No TEST value may enter selection, acceptance,
threshold choice, or continuation decisions.

The fixed ensemble result should be retained as exploratory context only. It
must not be promoted to a production model or used as evidence that the
plateau is solved without an approved, pre-registered experiment.

## 8. Verification and provenance

- Population audit read only canonical fold TRAIN/VAL CSVs, authoritative
  source metadata, and the existing availability manifest.
- Wav2vec audit read only TRAIN/VAL NPZ artifacts and compared their ordered
  turn IDs with EmotionMSP and TRILLsson; no TEST artifact was opened.
- CPU ensemble inference loaded only the existing validation-selected
  Weighted-F and DB-MTL checkpoints through the established Dataset,
  `MaskingCollate`, and gated-fusion model path.
- No repository model, loss, metric, DataModule, Dataset, configuration,
  launcher, or experimental file was changed.
- Temporary diagnostic code was kept outside the repository and is not part of
  the deliverable.

The next action requires manager review and explicit owner authorization for a
separate Batch 02 implementation/preparation task. This report does not
authorize CUDA execution.
