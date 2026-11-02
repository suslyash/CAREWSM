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

## 3. Normalization / PCA

Any scaler/PCA is a learned transformation and therefore follows split boundaries.

For fold k:

- fit on fold-k TRAIN only;
- transform fold-k VAL;
- transform DEV/TEST only with the fitted object when that fold/config is evaluated.

For final source-corpus training:

- fit preprocessing on authorized source TRAIN data only;
- target corpus data cannot refit preprocessing during direct transfer.

Joint training may fit preprocessing on the joint authorized training set.

Adaptation may refit only if the adaptation protocol explicitly permits it.

Persist preprocessing state and fingerprint it.

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

## 8. Metrics

Required:

- F1;
- weighted F1;
- UAR;
- weighted average recall.

Report per task.

For joint/transfer experiments also report per corpus/domain.

Coverage and number of observed labels are always reported beside metrics.

## 9. Model families

Main temporal classifiers:

- TSCMamba-inspired;
- MPTSNet-inspired;
- MD-Former-inspired.

A simple pooled MLP/reference classifier is permitted as a sanity baseline.

The three research models consume standardized native or Mantis-derived representations through clean family adapters.

## 10. No TEST-driven iteration

TEST and external test sets do not determine:

- feature sets;
- architecture;
- loss balancing;
- preprocessing;
- thresholds;
- Optuna trials;
- retry/ablation continuation.

Development decisions come from TRAIN CV and DEV according to the active experiment protocol.
