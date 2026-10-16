# TASK-010A method specification and source-grounding gate

## Status

**SOURCE GATE RESOLVED.** This is a manager-authorized, DEV-only,
post-closure extension. It does not revise a frozen ledger, paper, model role,
or Final-Test result. The 12 production invocations remain forbidden until the
separate pre-production firewall commit is pushed.

## Source manifest

| Method | Primary source | Authorized implementation source | Pin and relevant file | Resolved point |
|---|---|---|---|---|
| PCGrad | Yu et al., *Gradient Surgery for Multi-Task Learning*, [arXiv:2001.06782](https://arxiv.org/abs/2001.06782), NeurIPS 2020 | [official PCGrad](https://github.com/tianheyu927/PCGrad) | `c5fbd7c856526373828074f06875230f7f3ee79e`, `PCGrad_tf.py` | shuffled task order and projection implementation |
| CAGrad | Liu et al., *Conflict-Averse Gradient Descent for Multi-task Learning*, [arXiv:2110.14048](https://arxiv.org/abs/2110.14048), NeurIPS 2021 | [official CAGrad](https://github.com/Cranial-XIX/CAGrad) | `dc3d48152b6196945cfd56144879b9d42353b095`, `toy.py`, `nyuv2/utils.py` | constrained simplex objective and combined direction |
| GradNorm | Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, [PMLR 80](https://proceedings.mlr.press/v80/chen18a.html), ICML 2018 | primary algorithm, plus [LibMTL](https://github.com/median-research-group/LibMTL) | LibMTL `4336804847eaa5e0b924b743d76beec7ac3fdc97`, `LibMTL/weighting/GradNorm.py`, `config.py` | weight update, alpha and normalization convention |
| DB-MTL | Lin et al., *Dual-Balancing for Multi-Task Learning*, [arXiv:2308.12029](https://arxiv.org/abs/2308.12029), Algorithm 1; *Neural Networks* 195 (2026) 108317 | [LibMTL](https://github.com/median-research-group/LibMTL) | `4336804847eaa5e0b924b743d76beec7ac3fdc97`, `LibMTL/weighting/DB_MTL.py`, `config.py` | Algorithm-1 EMA implementation defaults |

The pinned LibMTL revision is also the accepted reference for
`PCGrad.py` and `CAGrad.py`. The repository does not vendor these sources and
does not add a dependency. The existing `docs/2501.10945v3.pdf` is secondary
context only.

## Common WSM contract

Each batch first computes the Depression and Parkinson task objectives
`L_D, L_P` through the same `_components(...)` semantics as the frozen
`WSMR4RampsBalanceLoss`: observed BCE, reliability-weighted accepted-pseudo
BCE, mutable pseudo-scale warm-up, both auxiliary losses, agreement loss,
current masks, and detached pseudo targets/reliabilities. Observed truth
therefore continues to override pseudo supervision.

Per-task gradients are taken with respect to actual trainable model parameters,
not features or logits. A parameter is shared precisely when both gradients are
non-`None`; a D-only or P-only parameter receives its own task gradient, and a
neither parameter remains absent. A custom autograd scalar injects the resulting
per-parameter gradients before the existing trainer unscales, clips, and calls
unchanged AdamW. This is an equivalent parameter-gradient construction, not a
global PyTorch monkey patch. A single-active-task batch uses that original
objective gradient unchanged. The implementation logs D-only/P-only/both/neither
parameter counts.

The extension uses a DEV-only DataModule whose validation mapping is exactly
`{"dev": loader}`. No Test loader is returned or invoked.

## PCGrad

For a shared two-task gradient pair `g_D, g_P`, if `g_D^T g_P < 0`:

`g'_D = g_D - (g_D^T g_P / (||g_P||^2 + eps)) g_P`

`g'_P = g_P - (g_D^T g_P / (||g_D||^2 + eps)) g_D`.

Otherwise both gradients are unchanged. The shared final gradient is
`0.5 (g'_D + g'_P)`: the official/LibMTL implementation sums task gradients,
but the factor `0.5` preserves the fixed WSM Equal-loss global scale
`0.5 L_D + 0.5 L_P`; the hand-calculation test verifies it. PCGrad acts on
actual gradients, not loss weights. The official shuffle is represented by a
local deterministic RNG initialized from the config’s run seed. With exactly
two tasks, each task has only one opponent, so opponent order cannot change the
projection. D-only/P-only parameters are unchanged.

Logged train diagnostics are pre/post cosine, conflict fraction, and pre/post
norms.

## CAGrad

Let `G` stack the two shared parameter-gradient vectors, `A=GG^T`,
`g0_norm=sqrt(mean(A)+eps)`, `b=(1/2,1/2)`, and
`c=calpha*g0_norm+eps`. The pinned exact constrained solve is:

`min_x x^T A b + c sqrt(x^T A x + eps)` subject to `0<=x_i<=1` and
`sum_i x_i=1`, initialized at `b` and solved deterministically by SciPy SLSQP.

Then `g_w=sum_i x_i g_i`, `lambda=c/(||g_w||+eps)`, and with the frozen
`calpha=0.5`, `rescale=1`:

`g = (mean_i g_i + lambda g_w) / (1 + calpha^2)`.

CAGrad changes shared gradients only; D-only/P-only gradients are kept. There
is no hyperparameter search. Logged diagnostics are raw cosine, combined and
mean norm, adjustment magnitude, simplex weights, and solver convergence.

## GradNorm

GradNorm is dynamic **loss weighting**, not a projection method. Its manager
frozen settings are `alpha=1.5`, initial weights `[1,1]`, a separate Adam for
these two weights only with learning rate `0.025`, and renormalization after
each update to a positive total of two. The frozen model optimizer remains
AdamW (`lr=1e-4`, `weight_decay=0.01`).

`L_i(0)` is captured on the first both-active TRAIN update and then fixed.
Under the explicit WSM adaptation, the GradNorm measurement vector is the
concatenation of parameters whose current gradients from both tasks are
non-`None`. Let `G_i=w_i||grad_W L_i||`,
`r_i=(L_i/L_i(0))/mean_j(L_j/L_j(0))`, and
`target_i=detach(mean_j(G_j)*r_i^alpha)`. The separate weight optimizer
minimizes `sum_i |G_i-target_i|`; the target is detached. The model receives
`w_D grad L_D + w_P grad L_P` (and its corresponding task weight for a
single-task parameter). If one task is inactive, weights are not updated and
the active task uses its current weight. Logged diagnostics include weights,
losses, rates, norms, targets, and auxiliary objective.

## DB-MTL

DB-MTL is a gradient magnitude-balancing method. Algorithm 1 first uses
`h_i = grad log(L_i + 1e-8)` for shared task gradients. At step `k`, with
manager-frozen LibMTL defaults `DB_beta=0.9` and `DB_beta_sigma=0.0`:

`u_i <- h_i + (beta / k^beta_sigma) (u_i - h_i)`;

`a_i = max_j ||u_j|| / (||u_i|| + 1e-8)`;

`g_shared = sum_i a_i u_i`.

Thus every nonzero normalized EMA term has the maximum EMA norm before the
sum. D-only/P-only parameters receive their own transformed
`grad log(L_i+eps)` without cross-task normalization, per Algorithm 1. For a
single-active task, the extension’s common safety contract keeps its original
gradient unchanged. The transformed `.grad` values are constructed before the
unchanged WSM AdamW step. Logged diagnostics include raw/EMA cosine, conflict
fraction, EMA norms, and normalized magnitudes.

## Deterministic pre-production checks

CPU tests cover Equal component/gradient parity, PCGrad aligned/orthogonal/
conflicting hand cases and task-local gradients, CAGrad identical/opposite/
unequal-norm cases against an independent constrained reference, GradNorm
initialization/positivity/normalization/slower-task pressure, DB-MTL loss-log
EMA/max-norm behavior and near-zero protection, plus an actual R4 backward
smoke for every method. The firewall additionally checks cache identity,
parameter count, DEV-only loader keys, and forbidden diffs before production.
