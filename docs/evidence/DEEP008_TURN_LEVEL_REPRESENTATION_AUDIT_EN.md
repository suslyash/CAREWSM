# DEEP008 — Native Turn-Level Representation Audit

Date: 2026-11-08  
Scope: existing native `emotionmsp_embeds` and `trillsson4_1` artifacts only

## 1. Outcome

The native artifacts contain usable temporal structure in CARE, but almost none
in WSM. Across the complete fixed TRAIN plus five fold-VALIDATION populations,
CARE has a median of 3 valid aligned turns per sample and a mean of 4.02; WSM
has exactly 1 turn for every valid complete-case sample. Therefore a future
order-aware model would be tested mainly on CARE. A small task-specific
attention pooling pilot is feasible, but a temporal convolution or recurrent
sequence model is not justified as the first controlled comparison because it
would have no temporal support in WSM and would add unnecessary capacity for
the dominant T=1 population.

This is an aggregate CPU audit. It did not read TEST populations, train or run
inference, inspect labels for selection, use a GPU, reconstruct audio, identify
speakers, extract features, use eGeMAPS/MantisV2, or persist sample-level
metadata.

## 2. Audit population and method

The audit used the authoritative fold CSVs and the existing
`CAREWSMAudioDataset` loading contract. The manifest was joined with source
split metadata for paths, as required by the active DataModule. For each
complete-case record, the two NPZ `turn_ids` arrays were compared; the Dataset
loader was also smoke-checked on an existing artifact. Only TRAIN and
`fold_{0..4}_val` records were included. The five fold TRAIN+VAL unions cover
the same fixed TRAIN population repeatedly, so the aggregate population is
reported once as unique TRAIN+fold-VAL records and not as five independent
datasets.

| Population | Fold-row count | Unique records | Complete valid pair | Excluded manifest records |
|---|---:|---:|---:|---:|
| CARE TRAIN + fold VAL | 8,565 | 1,713 | 1,713 | 0 |
| WSM TRAIN + fold VAL | 35,345 | 7,069 | 7,066 | 3 |
| Joint | 43,910 | 8,782 | 8,779 | 3 |

The three WSM exclusions were records not marked `VALID` for both selected
families. No sample was deleted from the repository; they are excluded only
from this pairwise turn-length summary.

## 3. Aggregate sequence-length statistics

Percentiles use linear interpolation. Proportions are proportions of complete
valid paired records.

| Corpus | N | Min | Median | Mean | P90 | Max | T=1 | T=2 | T=3–5 | T>5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CARE | 1,713 | 1 | 3 | 4.019 | 9 | 55 | 30.82% | 18.27% | 29.48% | 21.42% |
| WSM | 7,066 | 1 | 1 | 1.000 | 1 | 1 | 100.00% | 0.00% | 0.00% | 0.00% |
| Joint | 8,779 | 1 | 1 | 1.589 | 2 | 55 | 86.50% | 3.57% | 5.75% | 4.18% |

The joint distribution is dominated by WSM: only 13.50% of complete paired
records have T>1. This is not evidence that the CARE turn axis is useless;
rather, it means that any claimed joint-model gain must survive a large T=1
population where attention and temporal operators reduce to the baseline.

### Fold differences

CARE is the only corpus with meaningful fold variation in sequence length.
The table gives TRAIN and VALIDATION statistics separately; WSM is T=1 in
every fold and split.

| Fold | CARE TRAIN N / mean / median / P90 / max | CARE VAL N / mean / median / P90 / max | WSM TRAIN N | WSM VAL N |
|---:|---|---|---:|---:|
| 0 | 1,361 / 3.830 / 3 / 8 / 39 | 352 / 4.747 / 3 / 10 / 55 | 5,637 | 1,429 |
| 1 | 1,368 / 4.048 / 2 / 9 / 55 | 345 / 3.901 / 3 / 8 / 23 | 5,659 | 1,407 |
| 2 | 1,378 / 4.106 / 3 / 9 / 55 | 335 / 3.660 / 2 / 7 / 39 | 5,667 | 1,399 |
| 3 | 1,382 / 4.038 / 3 / 9 / 55 | 331 / 3.940 / 2 / 9 / 37 | 5,638 | 1,428 |
| 4 | 1,363 / 4.070 / 2 / 9 / 55 | 350 / 3.820 / 3 / 9 / 18 | 5,663 | 1,403 |

CARE TRAIN means range from 3.830 to 4.106 turns. CARE VAL means range from
3.660 to 4.747. The largest validation maximum is 55 turns in fold 0; no fold
has a uniformly long sequence population.

## 4. Ordering, alignment and masks

All 8,779 complete-case records had identical ordered `turn_ids` in
EmotionMSP and TRILLsson. The stored numeric turn IDs were non-decreasing in
all 8,779 records. The native Dataset does not sort, truncate or interpolate:
it loads each artifact in stored row order, concatenates the two families
along the feature dimension, verifies exact turn-ID equality, and emits an
all-ones `sequence_mask` over the retained rows. Thus the current pairwise
pipeline preserves the source row order and the same valid turns for both
families.

This is preservation of artifact order, not recovery of audio timestamps. The
turn-level NPZs expose turn IDs but do not provide a new timestamp axis in the
active Dataset contract. For WSM, the ordering claim is necessarily vacuous
for temporal modelling because every valid record has T=1.

## 5. What current models preserve and discard

`CAREWSMGatedFusionModel` applies a separate LayerNorm/linear/GELU projection
and learnable gate to each 1024-dimensional family at every turn. It sums the
gated family representations and then computes a single masked mean across
turns before the independent Depression and Parkinson heads. It therefore
preserves within-turn cross-family interaction and uses the number of valid
turns through the mean denominator, but discards turn position, order,
adjacent transitions, monotonic trends, and task-specific weighting of turns.

`CAREWSMAttentiveMLPModel` applies one shared projection and one scalar masked
attention score per turn, then uses the same pooled vector for both task heads.
It is order-invariant and attention is not task-specific. It preserves a
learned global importance weighting of turn content, but still discards order
and any explicit transition structure. Its masking is appropriate for padded
variable-length batches.

The pooled reference MLP first projects each turn and then uses masked mean;
the ordinary sequence models similarly reduce their encoded sequence with
masked mean after their temporal encoder. MPTSNet/MD-Former/TSCMamba paths can
transform valid positions before pooling, but their final reduction still
loses order-sensitive evidence at the sample representation boundary. The
TSCMamba path is dependency-blocked in the current environment and was not
used as evidence here.

Task-specific attention pooling is distinct from order-sensitive sequence
modelling: it lets Depression and Parkinson choose different weighted sums of
the same ordered per-turn features, but it cannot distinguish a sequence from
its reversal unless positional/order features or a sequence encoder are added.

## 6. Relationship to prior evidence

DEEP-008 fold-0 found the learned attentive MLP at 0.7741 validation Macro
Recall versus 0.7713 for the native EmotionMSP pooled MLP, while gated native
EmotionMSP+TRILLsson reached 0.8023. This is a useful negative/ambiguous result:
generic shared attention was not shown to produce a decisive gain, and it was
not applied after gated fusion or evaluated across five folds.

The controlled A/F five-fold follow-up found 0.8019 for native EmotionMSP MLP
and 0.8084 for gated EmotionMSP+TRILLsson. The gain was small and variable
(0.0065 mean difference); F's main positive component was WSM Parkinson, where
T=1 means the gain cannot be attributed to temporal ordering. The CARE
Parkinson error audit also found shared difficult patients and no established
sequence-length effect.

The TRAIN-only weighted-F experiment raised validation Macro Recall from 0.8084
to 0.8271 and CARE Parkinson Patient Recall from 0.5484 to 0.6988, at the cost
of lower Healthy Recall. This is evidence for class-imbalance handling, not
evidence for a temporal architecture. The DB-MTL five-fold evidence likewise
concerns task balancing and does not isolate turn order. These results require
the future architecture pilot to keep loss, fold construction, seed and
validation monitor fixed so that attention is not confounded with balancing.

## 7. Architecture feasibility

### A. Task-specific masked attention after gated fusion

Keep the existing per-turn gated fusion and add one scalar score head for each
task over the fused hidden representation. Apply masked softmax separately for
Depression and Parkinson, then feed the two pooled vectors to the existing
task heads.

- Difference: task-specific weighted pooling; still order-invariant.
- Extra parameters: approximately `2 * (hidden_dim + 1)` for hidden dimension
  64, about 130 parameters, excluding no new temporal block.
- Masking: score padded positions with `-inf`; ensure at least one valid turn;
  preserve the existing mask through both pools.
- Expected benefit: the two tasks can emphasize different turns and reduce
  dilution by irrelevant turns, especially in CARE samples with T=3–5 or
  longer sequences.
- Risk: low parameter overfitting risk, but the effective evidence is mostly
  short CARE sequences and nonexistent in WSM; the attention head may learn
  corpus or class shortcuts.
- Chimera scope: one new registered model factory/config variant, reusing the
  existing Dataset, DataModule, masked loss, metrics, callbacks and standard
  Chimera trainer. No custom training loop.

### B. Masked mean plus between-turn variability

Compute masked mean and masked standard deviation of the fused per-turn
representation, concatenate them, and update the task heads for the doubled
input. This captures dispersion and heterogeneity but not order.

- Extra parameters: the pooled representation doubles from H to 2H; the
  existing task-head input layers must be widened, adding roughly 2H² across
  the two heads at H=64, plus the corresponding bias terms.
- Masking: calculate variance only over valid turns and use a stable denominator;
  T=1 standard deviation must be defined explicitly (zero is the natural
  convention).
- Expected benefit: may detect inconsistent affect or turn heterogeneity.
- Risk: higher than A because variance is noisy for T=2 and adds no temporal
  order; it is redundant with a richer pooled feature and can amplify corpus
  differences.
- Chimera scope: a small model variant and one config; still compatible with
  the existing trainer.

### C. Lightweight temporal convolution with mask-safe aggregation

Apply one or two narrow 1-D temporal convolutions to the fused sequence, then
aggregate with a masked mean or task-specific pool.

- Difference: the kernel can encode local turn transitions and is order-sensitive.
- Extra parameters: a dense 64-channel kernel-3 layer adds about 12,352
  weights including bias; a second layer roughly doubles that block.
- Masking: padded values must be zeroed before convolution; valid-neighbour
  counts or an explicit mask-aware convolution are needed near padding so that
  zero padding is not learned as a fake turn. The output mask must be retained
  for aggregation.
- Expected benefit: local changes across adjacent CARE turns if those changes
  are clinically relevant and stable.
- Risk: high relative to observed support. It cannot learn temporal dynamics
  from WSM T=1, has little signal for T=2, and can overfit CARE-specific turn
  counts or padding boundaries.
- Chimera scope: materially larger than A, with a new registered model variant,
  mask-safe block tests, and an additional implementation review. It should not
  be the first pilot.

## 8. Recommendation and future pilot contract

Recommend **A: task-specific masked attention after existing gated fusion** as
one controlled pilot. It is the smallest change that tests a scientifically
distinct hypothesis—task-specific turn selection—without pretending that WSM
contains a usable sequence. It also directly extends the existing F arm, so
the comparison can be paired fold-by-fold and interpreted without changing
the feature representation, family gates, labels, loss or data layer.

### Baseline

Use the existing gated EmotionMSP+TRILLsson native F model, same seed, fixed
five joint folds, same native artifacts and masks, and the existing
TRAIN-derived Parkinson weighted-BCE setting used by the weighted-F evidence.
The baseline and attentive arm must run through the standard Chimera-ML YAML
configuration and trainer. No custom training loop, new preprocessing, TEST
inspection, or threshold tuning is permitted.

### VAL-only acceptance criteria

Pre-register the existing `val_mean_corpora_tasks/macro_recall` as primary:
the equal-weight mean of CARE/WSM × Depression/Parkinson validation Macro
Recall. Accept the attentive arm only if, over the five paired folds:

1. its mean primary monitor improves the F baseline by at least 0.010;
2. it does not reduce any one of the four component mean Macro Recalls by more
   than 0.020; and
3. the six approved metrics are reported separately for CARE and WSM and both
   tasks, with Patient and Healthy Recall shown together.

If these criteria are not met, retain gated masked mean pooling as the active
representation and record the result as negative evidence. TEST values may be
recorded only after the VAL decision and must not alter it.

### Minimal GPU budget

Two arms (fixed F baseline and task-specific attentive F) × five folds × one
seed = 10 sequential Chimera-ML runs. Use the existing batch size and native
data path, a maximum of 30 epochs with patience 5, and no sweep or retry
budget. This is the minimum useful paired evidence for the proposed question;
it is not authorized or launched by this audit. A later owner-approved task
may raise the epoch budget if the fixed pilot protocol requires parity with the
historical 50-epoch runs.

## 9. Limitations and conclusion

The audit establishes representation availability and alignment, not that turn
order predicts either disease. WSM's T=1 distribution prevents a joint claim
that temporal modelling helps across both corpora. CARE's sequence lengths are
long enough to support a low-capacity pooling comparison, but the prior
attention pilot and the small gated-fusion gain argue for a single paired
pilot, not a temporal-model search. No model, loss, metric, DataModule or
configuration was modified and no pilot was launched.
