# CAREWSM Audio Method Contract

## 1. Source feature layer

The project starts from 11 owner-reported extracted audio feature families.

TASK-001 determines for every family:

- actual path/key/schema;
- temporal resolution;
- whether it is frame-level, turn-level or static/aggregated;
- timestamp availability;
- validity and coverage.

No downstream temporalization rule may invent timing information not present in the source.

## 2. Two representation paths

Each candidate feature family or feature block can participate through two controlled paths when technically meaningful.

### A. Native path

Use the source representation directly:

- temporal arrays remain temporal;
- turn sequences remain ordered turn sequences;
- static/aggregated vectors remain static.

This path answers whether the original representation is already sufficient.

### B. Mantis path

Construct 1.0-second windows with 0.5-second hop over a defensible time axis, then pass each window through one frozen Mantis checkpoint to obtain:

`[num_windows, mantis_dim]`.

For each window retain:

- start/end;
- order;
- valid mask;
- valid-observation fraction/confidence.

The Mantis backbone is a frozen feature extractor. It is not duplicated per video.

The exact Mantis-family checkpoint, layer and pooling strategy must be pinned before bulk derivation and recorded in every artifact.

For static/turn-level families that do not naturally support 1-second windows, do not repeat vectors over fake time. TASK-002 must define a scientifically defensible mapping or mark that family as native-only.

## 3. Joint fold construction and normalization / PCA

The primary fold contract is joint from the beginning.

For fold `k`:

- `TRAIN_k = CARE_train_k ∪ WSM_train_k`;
- `VAL_k = CARE_val_k ∪ WSM_val_k`.

Any scaler/PCA is a learned transformation and is fitted only on `TRAIN_k`.

The fitted preprocessing transforms:

- CARE validation;
- WSM validation;
- CARE TEST;
- WSM TEST.

No validation/TEST sample may influence preprocessing fit.

Class weights are also computed from known labels in `TRAIN_k` only.

Persist preprocessing state and fingerprint it.

The fixed DEV split is not automatically merged into TRAIN or VAL; a later manager task must define any final-use role explicitly.

## 4. Classifier

Shared representation:

`common feature encoder`

Task-specific branches:

`shared -> depression feature layer -> depression logit`

`shared -> parkinson feature layer -> parkinson logit`.

The two outputs are independent binary logits.

## 5. Missing labels

Let `m_d`, `m_p` be observed-label masks.

For each task, compute supervised loss only over rows with its mask = 1.

If a batch contains no observed labels for a task, that task contributes no supervised loss/gradient for that batch.

Missing labels are never mapped to zero class.

## 6. BCE variants

Unweighted:

`BCEWithLogitsLoss`.

Weighted:

`pos_weight_t = N_negative_t / N_positive_t`

computed separately for Depression and Parkinson using known TRAIN labels only.

Never calculate class weights from DEV, TEST, external target data or hidden labels.

## 7. Multitask optimization

Baseline:

`0.5 * L_depression + 0.5 * L_parkinson`.

Compare against:

- PCGrad;
- CAGrad;
- STCH;
- PAGB.

Also train single-task Depression-only and Parkinson-only models for reference.

These comparisons must use the same data/features/model budget whenever possible.

## 8. Validation and TEST metrics

For each task and fold compute the following on both validation and TEST:

- Macro F1;
- Weighted F1;
- Macro Recall;
- Weighted Recall;
- Patient Recall;
- Healthy Recall.

Compute them separately for CARE and WSM portions.

Then compute the `Average` block metric-wise as:

`0.5 * metric_CARE + 0.5 * metric_WSM`.

Do not pool corpora before computing the Average.

Patient Recall is positive-class recall. Healthy Recall is negative-class recall. Macro Recall is the unweighted mean of the two and corresponds to UAR in the binary setting.

Metrics use only examples with the relevant target observed.

Coverage, evaluated sample count and observed-label count are always logged beside metrics.

Validation metrics determine checkpoint/configuration quality. TEST metrics are recorded only as statistics.

## 9. Model families

Main temporal classifiers:

- TSCMamba-inspired;
- MPTSNet-inspired;
- MD-Former-inspired.

A simple pooled MLP/reference classifier is permitted as a sanity baseline.

The three research models consume standardized native or Mantis-derived representations through clean family adapters.

## 10. Fold-level TEST logging without TEST-driven iteration

Every CV-fold checkpoint/configuration is evaluated on both CARE TEST and WSM TEST and the complete six-metric table is logged.

These TEST numbers are evidence, not a selection signal.

TEST and external test sets must not determine:

- feature sets;
- architecture;
- loss balancing;
- preprocessing;
- thresholds;
- Optuna trials;
- retry/ablation continuation.

Development decisions must be justified from TRAIN CV and DEV according to the active experiment protocol.
