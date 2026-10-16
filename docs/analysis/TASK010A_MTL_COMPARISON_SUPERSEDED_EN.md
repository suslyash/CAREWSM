SUPERSEDED — INVALID GRADIENT OWNERSHIP IMPLEMENTATION; DO NOT USE AS SCIENTIFIC EVIDENCE

This document preserves the prior TASK-010A values only as procedural history.
The old implementation classified all 51 trainable parameter tensors as shared
because it used autograd None/non-None behavior. Structural R4 ownership is
14 fully shared tensors, 18 Depression-only tensors, 18 Parkinson-only tensors,
and one row-partitioned task_queries tensor. The previous PCGrad, CAGrad,
GradNorm, and DB-MTL results are scientifically invalidated and must not be
used as comparator evidence.

# TASK-010A gradient-aware multi-task comparator evidence

## Scope and boundary

This is a manager-authorized, DEV-only post-closure extension. It does not
revise the frozen Stage-3, Stage-6, Stage-7, paper, Final-Test, or method-role
evidence. The experiment compares four actual-parameter gradient-handling
methods on the frozen R4 balancing composition:

- PCGrad: configs 01-03;
- CAGrad: configs 04-06;
- GradNorm: configs 07-09;
- DB-MTL: configs 10-12.

All twelve configurations were validated before production, remained frozen
after Commit A, and used the unchanged R4 model, loss contract, pseudo cache,
optimizer, warm-up, and three seeds. Checkpoint selection used only maximum
DEV `dev/mean_score`. The complete machine-readable selected-checkpoint
ledger is [task010a_mtl_runs_SUPERSEDED.csv](task010a_mtl_runs_SUPERSEDED.csv).

The prior CAGrad seed43 process was interrupted by the user's stop command
after TRAIN optimizer steps and before its accepted evidence handoff. The
replacement config05 run at 11:13 was explicitly authorized on continuation,
completed normally, and is the sole authoritative CAGrad seed43 result. The
11:05 directory is superseded and excluded from all metrics. Thus there are
twelve authoritative completed methods/config-seed results, with thirteen
physical invocations including one superseded interrupted attempt.

## Selected DEV evidence

| Method | Seed | Epoch | D UAR/MF1/Score | P UAR/MF1/Score | Mean | Checkpoint SHA256 |
|---|---:|---:|---|---|---:|---|
| PCGrad | 42 | 16 | 0.700654/0.700482/0.700568 | 0.826018/0.848247/0.837133 | 0.768850 | `08aa920ad8a1e78db53fdd7d2a83e459c761bdf8ee0be2c5bc229cad8106dc2d` |
| PCGrad | 43 | 7 | 0.704155/0.702898/0.703526 | 0.828157/0.842014/0.835086 | 0.769306 | `1fb350a6215756d838f071da16d608f0019033391f8f121872bac370bb97f1fa` |
| PCGrad | 44 | 15 | 0.711064/0.710783/0.710924 | 0.820773/0.827529/0.824151 | 0.767537 | `98a18a79dbdb2be85f33e83508d9324f541286f2109cb40dddcba5f3170b0d5b` |
| CAGrad | 42 | 3 | 0.704435/0.703827/0.704131 | 0.811387/0.823714/0.817551 | 0.760841 | `7c72417f0b2d9c46812b573b778941013d15fd95e14814958d43dc64eadb8298` |
| CAGrad | 43 | 11 | 0.714893/0.711924/0.713408 | 0.811318/0.821476/0.816397 | 0.764903 | `56ecc6b3bf614868b8fa9691509b1d8366df825209274e93a183d88dfbb29682` |
| CAGrad | 44 | 12 | 0.695191/0.695118/0.695155 | 0.830435/0.840512/0.835473 | 0.765314 | `681f3a77702dc6de36b0948c60bbdc28dbd14e51d271ffc0f2f7cbc649e9ecca` |
| GradNorm | 42 | 15 | 0.725117/0.724084/0.724600 | 0.830918/0.857549/0.844233 | 0.784417 | `e2c753c39e4fc7da6a03c842a1f3b339e080962fe0d394641ceb4d823b01e353` |
| GradNorm | 43 | 7 | 0.699580/0.698902/0.699241 | 0.837681/0.850396/0.844038 | 0.771640 | `3f1273bf926d87251c169b21784ddbd4fee59c739f2f2d0f1c70295fbe0778b9` |
| GradNorm | 44 | 15 | 0.703315/0.703297/0.703306 | 0.830435/0.840512/0.835473 | 0.769389 | `ff4631d1e14a9b23b5454e5f3d6304dc4461e8f1c0ed3cafe39663273e1ce6ea` |
| DB-MTL | 42 | 12 | 0.752894/0.747744/0.750319 | 0.840097/0.853720/0.846908 | 0.798614 | `2da6b770f5ffc100127e77fb7781609a2c8d233101267e174d93b2548fee28a3` |
| DB-MTL | 43 | 2 | 0.736648/0.730197/0.733422 | 0.711387/0.727104/0.719246 | 0.726334 | `f5da2dacbd72c29ff3180a7f2d89581836b2a8066b9ab33c1a533fffe986d59c` |
| DB-MTL | 44 | 4 | 0.663259/0.663275/0.663267 | 0.850794/0.819321/0.835057 | 0.749162 | `1406ec15ef23f4d3099cff0153ac6bdb4190bb4f0b5cabdaf422816c010e613b` |

Three-seed DEV aggregates (mean / sample standard deviation / range):

| Method | D Score | P Score | Mean Score |
|---|---|---|---|
| PCGrad | 0.7050060000 / 0.0053342745 / 0.010356 | 0.8321233333 / 0.0069796939 / 0.012982 | 0.7685643333 / 0.0009184467 / 0.001769 |
| CAGrad | 0.7042313333 / 0.0091269136 / 0.018253 | 0.8231403333 / 0.0106959773 / 0.019076 | 0.7636860000 / 0.0024723974 / 0.004473 |
| GradNorm | 0.7090490000 / 0.0136200682 / 0.025359 | 0.8412480000 / 0.0050022470 / 0.008760 | 0.7751486667 / 0.0081051374 / 0.015028 |
| DB-MTL | 0.7156693333 / 0.0461614618 / 0.087052 | 0.8004036667 / 0.0705339400 / 0.127662 | 0.7580366667 / 0.0369481986 / 0.072280 |

## Frozen comparator deltas

The frozen three-seed comparator means are Equal
`0.7791726667`, Static-STCH `0.7803973333`, corrected Progress
`0.7847510000`, and retained R4 `0.7869088179`. Relative to Equal, the
method Mean deltas are PCGrad `-0.0106083334`, CAGrad `-0.0154866667`,
GradNorm `-0.0040240000`, and DB-MTL `-0.0211360000`. Relative to
corrected Progress they are `-0.0161866667`, `-0.0210650000`,
`-0.0096023333`, and `-0.0267143333`, respectively. Every method is
below Equal on the three-seed Mean point estimate; no superiority, promotion,
or demotion claim follows.

The task-specific and per-seed values remain in the selected-run summaries
and CSV. These comparisons are descriptive DEV evidence only and do not
authorize tuning or reopening Stage 5.

## Controller and mechanism diagnostics

All selected diagnostics were finite. The parameter partition was
51 shared / 0 depression-only / 0 Parkinson-only / 0 neither for every
method and selected seed.

| Method | Selected diagnostic evidence |
|---|---|
| PCGrad | conflict fraction 0.393939-0.469697; pre-cosine 0.007886-0.022178; post-cosine 0.046454-0.115622; selected post-norms D/P 1.835862/1.581659, 4.850094/2.722590, and 2.740286/1.673206. |
| CAGrad | conflict fraction 0.353535-0.510101; solver converged on all seeds; simplex D/P weights 0.346132/0.653868, 0.523420/0.476580, 0.489194/0.510806; adjustment norms 0.924254, 0.886501, 0.699983. |
| GradNorm | final D/P weights 0.948037/1.051963, 0.831688/1.168312, 1.318834/0.681166; all positive and finite; selected norm pairs D/P 2.204847/1.634750, 4.784450/2.728454, 2.516688/1.757788. |
| DB-MTL | conflict fraction 0.338384-0.449495; pre/post cosine 0.025326/0.084242, 0.015280/0.075418, 0.029219/0.063154; normalized EMA norms were equal per selected seed at 1.321776, 5.721304, and 5.085740. |

These diagnostics show that the implementations exercised distinct
gradient-handling mechanisms under the frozen WSM contract. They do not
show that any mechanism improves the frozen method, nor do they establish
causal explanations for disease predictions.

## Audit boundary and interpretation

The DEV-only data module exposed only `{"dev": loader}`; no Test loader was
returned or iterated. No Test metrics were inspected or used. No Test strings
or protocol artifacts were present in any of the twelve authoritative run
directories. No recalibration, threshold search, metric-driven retry, sweep,
or extra seed was used. The superseded CAGrad seed43 directory is retained
only as procedural history and is excluded from this evidence.

The semantic cache, model parameterization, pseudo supervision, reliability,
optimizer, warm-up, architecture, and `src/audio` were unchanged. Unknown
labels remained masked and observed truth continued to override pseudo
supervision. This extension does not claim pseudo-label correctness,
missing-label recovery, comorbidity recovery, statistical significance,
corpus causality, or final method promotion.

TASK-010A is therefore recorded as a descriptive post-closure comparator
extension only. It does not revise the frozen WSM research evidence program
or authorize a follow-up experiment.
