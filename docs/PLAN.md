# WSM Development Plan

## 1. Execution Contract

Stages are normally sequential, but the owner explicitly overrides the research execution order after Stage 2: execute Stage 4 (Fusion), Stage 5 (RAMPS), and Stage 6 (core ablations/research) before returning to the deferred Stage 3 (Text/Description). Advance only after the active stage gate is met and evidence is recorded in PROGRESS_EN.md.

Authority: latest user instruction → PROJECT_REQUIREMENTS.md → this plan → BASELINES.md/SOTA_REVIEW_EN.md → existing code.

TEST_NONE/SOFT/HARD MUST be reported every epoch/validation cycle for comparative monitoring, using the same per-task UAR/MF1/Score/Mean_Score definitions as DEV. They MUST NOT be consumed by automatic checkpointing, early stopping, threshold search, or hyperparameter/model-selection logic. Sole automatic selection metric:

\[
\mathrm{DEV/Mean\_Score}
=\frac{1}{2}
\left[
\frac{\mathrm{UAR}_D+\mathrm{MF1}_D}{2}
+
\frac{\mathrm{UAR}_P+\mathrm{MF1}_P}{2}
\right].
\]

## 2. Research Hypothesis

Working method name: **RAMPS — Reliability-Aware Multimodal Partial Supervision**.

> Disjoint depression and Parkinson corpora can train one multilabel predictor if the missing disease head receives a soft cross-corpus target only when calibrated disease teachers, independent modality predictions, and semantic evidence agree; target reliability should control both pseudo-loss and inter-task gradient balance.

RAMPS jointly addresses:

1. multimodal fusion with variable stream availability/quality;
2. dataset-level partial labels;
3. negative transfer across tasks and corpora.

Proposed novelty:

- label-query fusion for independent disease labels;
- direct cross-corpus supervision for the missing head;
- reliability combining calibration, uncertainty, multimodal/semantic agreement, and OOD distance;
- reliability-conditioned smooth multi-objective optimization.

This is a working novelty hypothesis, not a guaranteed claim. Re-run a focused literature check before submission.

### 2.1. SOTA synthesis

| Direction | Useful mechanism | Remaining WSM gap |
|---|---|---|
| PASAL-like sparse MTL | Shared encoder learns from all observed tasks | Missing head gets no direct gradient |
| HiTTs-like label discovery | Direct pseudo-target for an unobserved task | Validated mainly for dense prediction |
| PPL/UNITI multi-dataset transfer | Teacher/student stability and distillation | Teacher bias and corpus shortcuts remain |
| SATL/CLS incomplete multilabel | Classwise thresholds and co-training | Usually not four-modality disjoint-disease MTL |
| HST semantic/prototype recovery | Label semantics recover unknowns | Semantic similarity is not clinical truth |
| Dual-branch pseudo-labeling | Independent views reduce confirmation bias | Must map views to A/V/T/D |
| URDF/CTRL uncertainty | Reject noisy supervision | Rarely coupled to task-gradient balance |
| V2L label-wise reliability | Per-label modality weighting | Does not solve cross-corpus missing tasks |

Required ladder: sparse MTL → calibrated pseudo-target → semantic/multimodal agreement → uncertainty/OOD reliability → reliability-aware optimization.

The unresolved intersection is multimodal + multitask + multilabel + dataset-level missing labels. If later work already matches this combination, revise the claim and architecture before expensive final runs.

## 3. Formulation

For sample \(x_i\):

\[
\mathbf y_i=(y_i^D,y_i^P),\qquad
\mathbf m_i=(m_i^D,m_i^P).
\]

\(m_i^t=1\) only for a truly observed label. The model returns:

\[
\mathbf z_i=(z_i^D,z_i^P),\qquad
\hat{\mathbf y}_i=\sigma(\mathbf z_i).
\]

Observed task loss:

\[
\mathcal L_{\mathrm{obs}}^t
=
\frac{\sum_i m_i^t\ell(z_i^t,y_i^t)}
{\sum_i m_i^t+\epsilon}.
\]

For \(m_i^t=0\), a teacher yields detached soft target \(\tilde y_i^t\), reliability \(r_i^t\in[0,1]\), and acceptance \(a_i^t\):

\[
\mathcal L_{\mathrm{pseudo}}^t
=
\frac{\sum_i(1-m_i^t)a_i^t r_i^t\ell(z_i^t,\tilde y_i^t)}
{\sum_i(1-m_i^t)a_i^t r_i^t+\epsilon}.
\]

Use separate positive/negative thresholds and warm-up:

\[
\mathcal L_t
=
\mathcal L_{\mathrm{obs}}^t
+
\mu(e)\mathcal L_{\mathrm{pseudo}}^t
+
\beta\mathcal L_{\mathrm{cons}}^t.
\]

Static STCH baseline:

\[
\mathcal L_{\mathrm{STCH}}
=
\tau\log\sum_t
\exp\left(
\frac{\lambda_t(\mathcal L_t-z_t^\star)}{\tau}
\right).
\]

RAMPS studies a detached, bounded, normalized, EMA-smoothed controller \(\alpha_t\) based on relative DEV progress, gradient norm/conflict, and accepted-pseudo reliability:

\[
\mathcal L_{\mathrm{RA-STCH}}
=
\tau\log\sum_t
\exp\left(
\frac{\alpha_t(\mathcal L_t-z_t^\star)}{\tau}
\right).
\]

Do not claim that dynamic RA-STCH inherits fixed-weight STCH theory without a new proof.

## 4. Architecture

### 4.1. Encoders

- Audio: frozen current WavLM-base-plus layer 9, pool 4, temporal Transformer; use checkpoint/cache through an external adapter.
- Video: frozen CLIP frame encoder, projection, temporal Transformer; one optional prototype-aware variant.
- Text: pretrained encoder with lightweight pooling/temporal head, selected after language/data audit.
- Description: cached observable descriptions/features from Qwen3-Omni-30B-A3B-Instruct or comparable model; no diagnosis or label/task input.

Each stream exposes token sequence, pooled representation, availability mask, quality metadata, and optional unimodal task logits.

### 4.2. Fusion ladder

1. masked concatenate/gated fusion;
2. TACME-like directed experts with task-specific gates;
3. label-query reliability fusion.

Final fusion uses learnable disease queries \(q_D,q_P\). Each query aggregates available modality tokens through cross-attention or a compact expert router. Gates depend on task, availability, and quality—not a corpus shortcut. Outputs remain independent logits.

TACME pairwise experts are a baseline. Prefer shared low-rank experts or query routing in the final model unless \(O(M^2)\) experts show clear benefit.

### 4.3. Teachers and reliability

Each disease teacher trains only on observed labels for that disease. Cross-corpus predictions never become truth automatically.

\[
r_i^t=g(c_i^t,u_i^t,a_{\mathrm{modal},i}^t,
a_{\mathrm{semantic},i}^t,d_{\mathrm{OOD},i}^t),
\]

where \(c\) is calibrated confidence, \(u\) uncertainty, modal/semantic terms measure agreement, and \(d_{\mathrm{OOD}}\) measures domain distance.

Start with a transparent normalized formula and class-specific thresholds. Learnable reliability requires a separate audit/calibration protocol.

### 4.4. Shortcut checks

Train an analysis-only corpus probe. If representations nearly identify corpus, test balanced sampling, train-only corpus normalization, domain-adversarial ablation, metadata restrictions, or stricter OOD filtering. Accept an intervention only if DEV/Mean_Score and negative transfer improve.

## 5. Stage Index

The active high-level roadmap and current status live in [PROGRESS_EN.md](PROGRESS_EN.md).
Detailed stage contracts are split so agents load only the stage relevant to the current task.

| Stage | Status | Detailed plan |
|---|---|---|
| 0. Reproducible base | COMPLETE | [plan/STAGE_0.md](plan/STAGE_0.md) |
| 1. Manifest and partial-label contract | COMPLETE | [plan/STAGE_1.md](plan/STAGE_1.md) |
| 2. Video | COMPLETE | [plan/STAGE_2.md](plan/STAGE_2.md) |
| 3. Text/description | DEFERRED | [plan/STAGE_3.md](plan/STAGE_3.md) |
| 4. Fusion baselines | COMPLETE | [plan/STAGE_4.md](plan/STAGE_4.md) |
| 5. RAMPS | CLOSED TO OPTIMIZATION — CANDIDATE RETAINED | [plan/STAGE_5.md](plan/STAGE_5.md) |
| 6. Ablations and claims audit | ACTIVE | [plan/STAGE_6.md](plan/STAGE_6.md) |
| 7. Final evaluation | LOCKED | [plan/STAGE_7.md](plan/STAGE_7.md) |

Closed-stage files and the historical progress archive are **optional context**. Do not load them unless the active task or manager explicitly requires historical evidence.

## 6. Experiment Matrix

| ID | Method | Question | Pre-final seeds |
|---|---|---|---:|
| A0 | Frozen audio | Is the original baseline reproducible? | 1 smoke + saved evidence |
| V1 | CLIP + Transformer | Is the base DEPART-like encoder sufficient? | 1–3 |
| V2 | V1 + prototypes | Do prototypes help? | 1–3 |
| T1 | Transcript encoder | Does text add independent signal? | 1–3 |
| T2 | Transcript + description | Does the semantic stream help? | 1–3 |
| F0 | Simple masked fusion | Basic multimodal gain | 3 |
| F1 | Sparse masked MTL | Shared two-head benefit | 3 |
| F2 | TACME-like fusion | Directed task-relation benefit | 3 |
| R1 | F2 + pseudo | Direct missing-head benefit | 3 |
| R2 | R1 + reliability | Negative-transfer reduction | 3 |
| R3 | Label-query RAMPS | Task/label-conditioned fusion benefit | 3 |
| R4 | R3 + RA-STCH | Reliability-aware balancing benefit | 3 |
| Final | Best baselines and RAMPS | Final comparison | 5+ |

This caps experiment families; it does not authorize a full grid search.

## 7. Promotion Rule

Compare DEV/Mean_Score first, then task-specific regressions, calibration/negative transfer, compute/stability, and diagnostic support.

Promote a component only when:

- the effect repeats across at least 3 seeds;
- mean DEV/Mean_Score beats the comparator;
- neither task shows a persistent unacceptable drop;
- diagnostics support the proposed mechanism.

Fix the unacceptable-drop threshold before final runs; initial recommendation is 1.0 absolute percentage point unless the confidence interval indicates noise.

## 8. Risks and Fallbacks

| Risk | Detection | Fallback |
|---|---|---|
| Corpus identity replaces disease signal | corpus probe/calibration | stricter OOD filter, balanced sampling, domain ablation |
| Pseudo-label majority collapse | classwise coverage/precision | separate thresholds, class cap, longer warm-up |
| Description leaks diagnosis/bias | prompt/manual audit | remove diagnostic tokens, fixed semantic features, drop stream |
| Video extraction fails | failure report/mask | mask stream; never synthesize fake features |
| Pairwise experts are too costly | memory/time profile | shared low-rank query router |
| Dynamic weighting is unstable | norms/cosines/NaN | static STCH or equal weights |
| Fusion underperforms audio | negative-transfer table | preserve strong audio path; improve gating/dropout |
| No dual-annotated subset | missing truth unevaluable | restrict claims to observed-task transfer |

## 9. Planned Layout

    configs/wsm_mm_pd_dep_v1/{audio,video,text,description,fusion,ablations}/
    scripts/{audio,video,text,description,fusion,common}/
    src/{audio,video,text,description,fusion,common}/

Universal manifest/masks, metrics, diagnostics, losses, and generic blocks belong in common. Teachers, routers, the multimodal datamodule, and the final model belong in fusion. All selectable components must be registered.

## 10. Manager Orchestration

Each cycle:

1. manager reads PROGRESS_EN and verifies the previous gate;
2. manager writes exactly one atomic task to NEXT_TASK_EN;
3. Codex reads AGENTS, REQUIREMENTS, PLAN, PROGRESS_EN, and NEXT_TASK_EN;
4. Codex executes only that scope, verifies it, updates PROGRESS_EN, and stops;
5. manager reviews diff/evidence and either corrects the task or advances the plan.
