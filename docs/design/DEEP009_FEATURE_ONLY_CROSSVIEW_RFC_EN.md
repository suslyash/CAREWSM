# DEEP-009 — CAREWSM-CrossView Feature-Only Architecture RFC

Date: 2026-11-08
Scope: existing native `emotionmsp_embeds` and `trillsson4_1` artifacts only

## 1. Decision summary

Recommend implementing one opt-in `carewsm_crossview` pilot. CrossView is a
small, testable extension of the weighted-F gated-fusion control: it adds a
low-rank multiplicative interaction between the two already aligned feature
families at each native turn, while keeping the projections, gates, masked
pooling, task heads, loss, data layer and validation protocol unchanged.

This is worth one controlled experiment because the current gated fusion is
additive after independent family projections. It can combine the two views,
but it does not explicitly test whether a feature in one view is useful only
in combination with a feature in the other. CrossView tests that hypothesis
with approximately 3,073 additional parameters. It is not expected to create
temporal evidence where none exists: WSM has `T=1` in every valid audited
record, and only 13.50% of the joint audit population has more than one turn.

No implementation or experiment is authorized by this RFC.

## 2. Evidence motivating the design

The historical weighted-F gated-fusion control achieved five-fold VAL UAR
`0.827060 +/- 0.027880`. Stage-G DB-MTL reached `0.831680 +/- 0.037077`, but
that comparison concerns task optimization rather than a cross-family
interaction. Task-specific gated attention reached VAL UAR `0.822980 +/-
0.030914` and failed its preregistered acceptance rule, including a WSM
Depression component decrease. It changed turn weighting, not family
interaction.

Batch 01 did not establish a useful alternative: its tested corpus balancing,
dropout and hidden-width variants did not meet the `+0.010` VAL criterion.
Batch 02 H1 independently optimized the two tasks and reached paired VAL UAR
`0.834070 +/- 0.032738`, an improvement of only `+0.007010`; H2 corpus heads
reached `0.823745 +/- 0.031097`. Neither met the frozen acceptance rules.
These results do not prove that the gated representation is inadequate. They
show that task weighting, turn selection, corpus routing and modest capacity
changes have not yet isolated or established a cross-family interaction gain.

The turn audit found 1,713 CARE records with median `T=3`, mean `4.019`, P90
`9`, and maximum `55`; 7,066 WSM records all had `T=1`. EmotionMSP and
TRILLsson had identical ordered turn IDs for all 8,779 complete paired
records. Thus the proposed interaction is valid for every complete record,
but its joint benefit must survive the dominant one-turn population.

The full historical TEST audit found a highest valid individual fold UAR of
`0.810075` and a highest complete five-fold mean of `0.784105 +/- 0.022360`.
Those TEST values are descriptive evidence only and do not motivate model,
feature, rank or threshold selection.

## 3. Proposed mathematics

Let a batch contain native values

\[
x \in \mathbb{R}^{B \times T \times 2048},
\qquad x = [x_e, x_t],
\]

where `x_e` and `x_t` each have 1,024 dimensions. Let `m` be the boolean
sequence mask. For each turn, retain the existing family-specific projection
and gate:

\[
e = P_e(x_e), \qquad t = P_t(x_t),
\]

\[
g_e = \sigma(G_e(x_e)) \odot e, \qquad
g_t = \sigma(G_t(x_t)) \odot t,
\]

\[
h_g = g_e + g_t,
\]

with hidden dimension `H=64`. `P` is the existing LayerNorm, linear and GELU
projection; `G` is the existing linear gate. No raw waveform or new temporal
axis is introduced.

For rank `r=16`, add bias-free low-rank projections:

\[
u = U_e(e) \in \mathbb{R}^{r}, \qquad
v = U_t(t) \in \mathbb{R}^{r},
\]

\[
c = W_o(u \odot v) \in \mathbb{R}^{H},
\]

and combine the branch residually:

\[
h = h_g + \alpha c.
\]

`U_e`, `U_t` and `W_o` are bias-free linear layers with shapes `64 -> 16`,
`64 -> 16` and `16 -> 64`. Use Xavier initialization for these matrices and a
learnable scalar `alpha` initialized to `0.1`. This makes the initial branch
contribution small while allowing gradients through the interaction from the
first update. The interaction is computed independently at each turn; it is
not attention, convolution, recurrence or positional encoding.

Finally, use the existing masked mean:

\[
z = \frac{\sum_{j=1}^{T} m_j h_j}{\max(1,\sum_{j=1}^{T}m_j)},
\]

then the unchanged independent Depression and Parkinson heads produce logits
in canonical `[Depression, Parkinson]` order. An opt-in implementation should
reject a sample with no valid turns rather than produce a meaningless pooled
vector. Padded positions must be excluded from the pooled sum.

## 4. Minimal Chimera-ML implementation

Add one readable model class and one explicit factory registration in the
existing representation-pilot model module. The implementation should:

1. reuse the existing `[1024, 1024]` family validation, `_unpack` helper and
   `ModelOutput` interface;
2. retain the existing `projections`, `gates` and `TaskHeads` behavior;
3. split the native tensor once, calculate `h_g`, calculate the three small
   CrossView layers, and add the residual branch;
4. apply the existing mask only for valid-turn pooling and reject all-invalid
   rows;
5. return the pooled features and logits in `aux` without changing the loss,
   metric, callback, DataModule or Trainer interfaces.

The YAML should declare only the new model registry name while preserving the
weighted-F input dimensions, hidden size, dropout, loss, seed, optimizer,
folds, callbacks and monitor. No generic attention framework, corpus-specific
heads, extra feature family, new loss, custom training loop or new data
preprocessing is needed.

## 5. Input, shape and mask contracts

| Item | Contract |
|---|---|
| Families | `emotionmsp_embeds` then `trillsson4_1` |
| Native feature shape | `[B, T, 2048]`, split as `[B,T,1024] + [B,T,1024]` |
| Hidden shape | Per-turn `[B,T,64]`; pooled `[B,64]` |
| Rank | `r=16`, fixed for the first pilot |
| Mask | boolean `[B,T]`, true for valid aligned turns |
| Alignment | Same native turn IDs and stored row order in both families |
| Pooling | Masked mean after residual CrossView fusion |
| Outputs | `[B,2]`, Depression then Parkinson logits |
| Labels | Existing observed-label masks; unavailable labels remain masked |
| Invalid rows | At least one valid turn required per sample |

Changing padded values must not change predictions. Because the interaction is
pointwise in `T` and only valid turns contribute to pooling, no padded value can
enter a valid turn's representation. WSM's one valid turn is still a valid
CrossView input: the model tests complementarity between two feature views
without claiming a temporal effect.

## 6. Parameter count and RTX 4080 budget

At `H=64`, the existing gated representation contributes 266,496 parameters.
The two unchanged task heads contribute 8,706, giving a 275,202-parameter
weighted-F gated model. The rank-16 CrossView branch adds:

| Component | Parameters |
|---|---:|
| `U_e: 64 -> 16`, no bias | 1,024 |
| `U_t: 64 -> 16`, no bias | 1,024 |
| `W_o: 16 -> 64`, no bias | 1,024 |
| Learnable `alpha` | 1 |
| **Additional total** | **3,073** |

The resulting estimate is 278,275 trainable parameters, a 1.12% increase.
The additional per-turn computation is approximately `3Hr = 3,072`
multiply-accumulates, excluding elementwise operations. At batch size 64 this
is about 0.20M MACs for `T=1`, 0.59M for `T=3`, and 0.98M for `T=5`.

The CrossView intermediate activations are small: storing `u`, `v` and the
rank product requires `3 * B * T * 16` values, plus the output branch. In
fp16 this is approximately 0.02 MB per turn at `B=64` for those rank tensors,
or 0.10 MB at `T=5`, before autograd bookkeeping. The model is therefore
parameter- and activation-light relative to the existing 1024-to-64 family
projections. A future RTX 4080 pilot should retain the established batch size,
mixed precision and gradient clipping; no memory-driven protocol change is
justified. These are analytical estimates, not a GPU benchmark.

## 7. Fair five-fold comparison

The future pilot should compare exactly one CrossView arm against the historical
weighted-F control, without retraining the control unless its existing run
artifacts are unavailable. Both arms must use:

- the same five fixed joint CARE+WSM folds and participant isolation;
- native EmotionMSP and TRILLsson features only;
- seed 5, hidden dimension 64, dropout 0.1 and batch size 64;
- the same AdamW learning rate `0.0001`, weight decay, epoch limit, patience,
  mixed precision and gradient clipping;
- each fold's TRAIN-derived Parkinson `pos_weight` and the same weighted-F
  loss/missing-label semantics;
- the same `val_mean_corpora_tasks/macro_recall` checkpoint monitor and
  validation-only early stopping;
- the same Chimera Dataset, DataModule, metrics, callbacks, logger and run
  provenance requirements.

For every fold report VAL UAR, the four corpus/task Macro Recalls, and Patient
and Healthy Recall for each corpus/task. Report the paired fold difference
from weighted-F, sample standard deviation, parameter count and the additional
CrossView compute estimate. TEST may be read only after the VAL decision is
frozen and is descriptive; it must not select a rank, initialization,
threshold, checkpoint or continuation.

## 8. Acceptance, risks and stop conditions

Pre-register the existing four-component arithmetic VAL UAR as the primary
decision quantity. Accept CrossView only if all of the following hold over the
five paired folds:

1. mean VAL UAR improves over weighted-F by at least `+0.010`;
2. no CARE/WSM by Depression/Parkinson component mean decreases by more than
   `0.020` absolute;
3. at least three of five paired folds improve;
4. all five folds have valid completion evidence and all required metrics,
   including Patient and Healthy Recall, are available.

Stop and retain weighted-F as the active representation if any fold is invalid,
if the acceptance rule fails, or if implementation tests reveal mask leakage,
non-finite logits/gradients or incorrect output ordering. Do not tune rank,
alpha, dropout, thresholds or features after seeing TEST. A failed pilot is
negative evidence and does not authorize a second architecture.

Primary risks are modest effective sample support for multi-turn behavior,
overfitting of a multiplicative shortcut, scale sensitivity between projected
families, and the fact that WSM cannot test temporal variation. LayerNorm in
the existing projections, small rank, residual scaling and the fixed protocol
limit these risks. If a later authorized implementation uses the proposed
`alpha=0.1` initialization, it should verify finite forward/backward behavior,
mask invariance, T=1 behavior and gradient flow through both family branches
before any real run.

### Final recommendation

Implement CrossView as one compact, opt-in, rank-16 pilot. It has a clearer
feature-only rationale than another temporal or corpus-routing variant because
it tests a mechanism still absent from the control while remaining meaningful
for WSM's `T=1` records. The recommendation is conditional: it is a
scientific test of complementarity, not a prediction of improvement. No
additional rank sweep, temporal convolution, attention mechanism or third
feature family should be added unless this single VAL-controlled comparison
first satisfies the preregistered criteria.
