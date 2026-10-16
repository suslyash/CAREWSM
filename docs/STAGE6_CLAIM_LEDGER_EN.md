# Stage-6 Claim Ledger and Completeness Matrix

Status: active pending manager synthesis review. This ledger is a documentation-only consolidation of accepted Stage-6 evidence on the manager branch at TASK-006K. It authorizes no experiment, method selection, promotion, demotion, Stage-7 work, or Final Test.

## Frozen interpretation rules

- All claims are DEV-selected, three-seed ablation claims under their stated isolation boundary. They are not significance claims and do not establish pseudo-label correctness, missing-label recovery, comorbidity recovery, causality, or final-method promotion.
- Test outputs were monitoring-only after checkpoint freeze and were not used for selection, calibration, comparison, or claims.
- Candidate implications below are descriptive. Full trial-012 remains a reference candidate, not a promoted final method.

## Authoritative Stage-6 claim ledger

| Plan item | Accepted evidence | Comparator/control and key DEV evidence | Frozen result | Status and boundary |
|---|---|---|---|---|
| 1. F1 sparse MTL | TASK-006J, configs58-63 | Full R4 versus independently selected D-only/P-only copies; paired Mean `0.810279/0.809722/0.779012`; aggregate full-minus-single D/P/paired Mean `+0.017410/-0.042935/-0.012762` | `SPARSE MTL JOINT-TRAINING CONTRIBUTION NOT SUPPORTED` | NOT SUPPORTED. The comparison concerns joint sparse-MTL training only. |
| 2. Task-aware fusion | TASK-006I, configs55-57 | Equal-parameter shared fusion (both 295239 parameters); shared D/P/Mean `0.7407873333/0.8467100000/0.7935663333`; full-minus-shared Mean `-0.0066575154`, full better on `0/3` seeds | `TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED` | NOT SUPPORTED. Sparse MTL and separate heads remained active. |
| 3. Direct pseudo-supervision | TASK-006D, configs40-42 | No-direct-loss D/P/Mean `0.7134376667/0.8263193333/0.7698780000`; full-minus-no-direct `+0.0218186667/+0.0122413333/+0.0170308179` | `DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED` | SUPPORTED. Concerns only the direct pseudo BCE term; pseudo masks, reliability, and RA paths remained present. |
| 4. Uncertainty/reliability | TASK-006E, reliability ablation | Uniform-reliability D/P/Mean `0.7352583333/0.8345853333/0.7849220000`; full-minus-uniform `-0.0000020000/+0.0039753333/+0.0019868179` | `UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED` | SUPPORTED under the frozen gate, with a small aggregate effect. Combined graded reliability in pseudo BCE and RA reliability is not separately identified. |
| 5. Semantic evidence | TASK-006F, configs46-48 | No-semantic depression D/P/Mean `0.714447/0.868369/0.791408`; full-minus-no-semantic `+0.020809/-0.029808/-0.004499` | `SEMANTIC-EVIDENCE CONTRIBUTION VIA DEPRESSION PSEUDO ACCEPTANCE NOT SUPPORTED` | NOT SUPPORTED. Boundary is semantic-enabled depression pseudo acceptance only. |
| 6. RA-STCH | TASK-006C, fixed-composition RA audit | RA D/P/Mean `0.6931503333/0.8519640000/0.7725573333`; Equal Mean `0.7791726667`; corrected Progress Mean `0.7847510000`; RA minus best simple comparator per seed `-0.004654/-0.010530/-0.024654` | `RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED`; `RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED` | NOT SUPPORTED. Fixed-composition balancing result does not automatically demote optimized trial-012. |
| 7. Remove each modality | TASK-006G/H, configs49-54 | No-video D/P/Mean `0.749645/0.783216/0.766431`; full-minus `-0.014389/+0.055344/+0.020478`. No-audio online branch D/P/Mean `0.653375/0.792030/0.722703`; full-minus `+0.081881/+0.046531/+0.064206` | `VIDEO MODALITY CONTRIBUTION NOT SUPPORTED`; `ONLINE AUDIO INPUT MODALITY CONTRIBUTION SUPPORTED` | Video claim is the exact no-video gate only. Audio claim concerns only the online student audio branch; historical audio-derived teacher evidence remains in the frozen pseudo cache. |
| 8. Shuffled/mismatched pseudo negative control | TASK-006A-FIX2 | Matched Equal Mean `0.7791726667`; shuffled Mean `0.7371180000`; matched-minus-shuffled `+0.0420546667`; matched higher on `3/3` seeds | `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL` | SUPPORTED narrowly for this negative control. No correctness, missing-label, comorbidity, significance, or promotion claim follows. |
| 9. Corpus probe | TASK-006B | True AUROC means: audio `0.791292`, video `0.705505`, projected A+V `0.750650`, task-fused `0.759832`, gates `0.551605`, logits `0.575197`; task-fused minus projected A+V mean `+0.009182` | `STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED`; `FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED` | DIAGNOSTIC ONLY. Corpus identity is structurally coupled to observed-task identity; no causal corpus-shortcut claim. |
| 10. Equal-parameter control if size grows materially | TASK-006I | Explicit equal-parameter shared-fusion control; full and shared models both have `295239` trainable parameters and shared aggregate Mean `0.7935663333` | Covered by TASK-006I | COMPLETE. The conditional equal-parameter requirement is satisfied; no additional experiment is required. |

## Required diagnostic coverage

| Diagnostic | Accepted evidence and scope |
|---|---|
| DEV UAR/MF1/Score | Every accepted production ablation selected checkpoints by DEV `mean_score` and recorded per-task UAR, MF1, Score, and Mean across the required seeds. |
| ECE/Brier | DEV-only calibration audits were recorded for the full reference and each accepted ablation; no Test rows entered post-hoc diagnostics. |
| Pseudo coverage/class balance | Frozen semantic cache SHA `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; accepted D/P counts `376/1801`; accepted classes D `376/0`, P `212/1589`. |
| Task gradient norms/cosines | TRAIN-only firewalls and accepted ablation evidence recorded finite active gradients, task gradient norms, and gradient-cosine/controller diagnostics under the relevant isolation boundary. |
| Negative-transfer deltas | Full-minus-ablation task and Mean deltas are recorded above for each component and modality audit; the TASK-006J paired task-isolated deltas are explicitly retained. |
| Gate distributions | DEV-only task-modality gate means/stds were recorded for the full reference and the accepted ablations, including modality-removal gate checks. |

CORE STAGE-6 A+V EVIDENCE MATRIX COMPLETE

## Frozen reference and descriptive candidate implications

Full R4 trial-012 reference: D `0.7352563333`, P `0.8385606667`, Mean `0.7869088179`, Mean sample std `0.0294384420`. Matched temporal audio reference: Mean `0.7733169588`, Mean sample std `0.0138512531`. Full R4 exceeds matched audio Mean on `3/3` seeds, while matched audio is more seed-stable. Shared-fusion equal-parameter Mean is `0.7935663333`; no-semantic-depression Mean is `0.791408` with a D/P trade-off; paired task-isolated Mean is `0.799671` but uses two models and is not an equal-total-deployment-size comparator; no-video Mean is `0.766431`; no-audio Mean is `0.722703`.

These facts do not justify claiming every full trial-012 component is necessary. Multiple complexity/mechanism claims are unsupported. Direct pseudo supervision, graded reliability, sample-specific pseudo alignment under its negative control, and the online audio branch have positive support only under their exact gates. No final method is promoted, demoted, or selected here.

## Stage boundary

Stage 6 remains active pending manager closure. Stage 7 remains LOCKED and Final Test remains unauthorized. PROJECT_REQUIREMENTS still requires the deferred Stage-3 Text/Description study before paper-ready/final evaluation freeze. The next programme-level phase after manager acceptance should be that deferred Stage-3 study unless the manager identifies a genuine Stage-6 documentation/evidence gap. This task does not activate Stage 3, edit its plan, start Stage 7, or propose new A+V tuning.

## TASK-006J procedural exception

TASK-006J had exactly 7 total training invocations: the first D42 invocation failed during TRAIN epoch 1 before DEV/Test evaluation or checkpoint selection; a corrective firewall preceded the accepted sequence D42, P42, D43, P43, D44, P44. The six completed runs are the accepted evidence. There was no metric-driven retry or tuning.
