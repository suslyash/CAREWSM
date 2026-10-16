# Baselines and Transferable Ideas

## 1. Purpose

This document reviews the four supplied papers and separates reproducible baselines, transferable mechanisms, incompatible assumptions, and hypotheses that still require evidence.

Sources:

1. [2402.19078v3.pdf](2402.19078v3.pdf) — Xi Lin et al., *Smooth Tchebycheff Scalarization for Multi-Objective Optimization*, ICML 2024.
2. [2501.10945v3.pdf](2501.10945v3.pdf) — Weiyu Chen et al., *Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond*, arXiv version dated 6 August 2025.
3. [BDCC-10-00089.pdf](BDCC-10-00089.pdf) — Elena Ryumina et al., *DEPART: Multi-Task Interpretable Depression and Parkinson’s Disease Detection from In-the-Wild Video Data*, BDCC 2026.
4. [Ryumin_EMNLP.pdf](Ryumin_EMNLP.pdf) — Elena Ryumina et al., *TACME: Task-Aware Cross-Modal Experts for Disjoint-Corpus Multimodal Behavioral Understanding*. Treated as a supplied manuscript; publication status was not independently verified.

## 2. Comparison

| Work | Problem | Core mechanism | Value for WSM | Missing piece |
|---|---|---|---|---|
| STCH | Conflicting objectives | Smooth log-sum-exp Tchebycheff scalarization | Cheap Pareto-aware aggregation of two task losses | No missing-label recovery or fusion |
| MOO survey | Multi-objective deep learning taxonomy | Loss/gradient balancing; finite/continuous Pareto sets | Defines the optimization family and comparators | Not a model for disjoint corpora |
| DEPART | Video depression/PD recognition on WSM | Body ROI, CLIP, temporal encoder, prototypes | Strong DEPART-like video baseline and failure analysis | Mutually exclusive three-class target |
| TACME | Multimodal MTL on disjoint corpora | Directed cross-modal experts, task gates, FM, PAGB | Closest fusion baseline and correct task masking | No direct target for the missing head; corpus/task confounding |

## 3. STCH

For differentiable objectives \(f_i(\theta)\), a weighted sum is cheap but may miss non-convex parts of the Pareto front. Classical Tchebycheff scalarization can reach them but is non-smooth:

\[
\max_i \lambda_i(f_i(\theta)-z_i^*).
\]

STCH replaces max with:

\[
\mathcal L_{\mathrm{STCH}}
=
\mu\log\sum_{i=1}^{m}
\exp\left(
\frac{\lambda_i(f_i(\theta)-z_i^*)}{\mu}
\right), \qquad \mu>0.
\]

As \(\mu\) decreases, STCH approaches Tchebycheff max. The paper gives an approximation gap of order \(\mu\log m\), Pareto guarantees under its stated assumptions, and convergence to a Pareto-stationary point in the non-convex case.

Unlike MGDA/CAGrad/PCGrad-like methods, STCH needs neither a separate backward pass per objective nor a per-step auxiliary quadratic program. Compared with equal weighting, it emphasizes the objective that is worse relative to its reference point. Loss normalization, \(\mu\), \(\lambda_i\), and \(z_i^*\) therefore belong to the experimental protocol.

Reported evidence:

- NYUv2: second-best aggregate improvement over the single-task baseline and much better than non-smooth TCH;
- Office-31: best aggregate result and the only compared method dominating the single-task baseline on every task;
- QM9: close to DB-MTL with much lower cost than gradient-balancing methods;
- Pareto-set experiments: best hypervolume difference on 10 of 11 tasks.

For WSM, use \(f_D=\mathcal L_D\) and \(f_P=\mathcal L_P\). STCH is a required optimization baseline, not a novelty claim: it creates no supervision for an unknown label and handles neither corpus confounding nor multimodal fusion.

## 4. Gradient-Based MOO Survey

The survey separates methods that produce one balanced model, a finite set of Pareto solutions, or a continuous Pareto set. WSM needs one balanced model.

Main families:

- loss balancing: equal weighting, DWA, uncertainty weighting, STCH, validation-aware methods;
- gradient balancing/manipulation: MGDA, GradNorm, CAGrad, Nash-MTL, FairGrad, PCGrad, GradDrop, Aligned-MTL, DB-MTL.

Loss balancing generally uses one backward pass. Gradient methods may control conflict more explicitly but cost more memory and computation.

Project decision:

- equal weighting is the simple baseline;
- STCH is the cheap Pareto-aware baseline;
- one strong progress/gradient-aware comparator is enough;
- optimization must not be confused with partial-label recovery;
- multi-seed stability and negative-transfer analysis matter more than a one-run gain.

## 5. DEPART

### 5.1. Data and architecture

DEPART processes 60-second WSM video segments. Segments without a detected face or speech are removed. It uniformly samples \(N=60f\) frames, detects the person with YOLOv8, crops the body ROI, and resizes it for the visual encoder.

Although each source corpus is binary, the paper combines depression and PD into three mutually exclusive classes: healthy, depression, and PD. It explicitly notes that comorbidity is not annotated and multilabel combinations are not modeled.

Architecture:

1. Frozen CLIP ViT-B/32 or ViT-B/16 frame encoder; CLS embedding size 768.
2. Linear projection, LayerNorm, and dropout.
3. Transformer or Mamba temporal encoder with masked mean pooling.
4. Fixed, time-wise, or feature-wise residual gating.
5. Several trainable prototypes per class; maximum cosine similarity divided by temperature gives the prototype logit.
6. Parallel two-layer MLP classifier.
7. Per-class learned gate mixing prototype and MLP logits.

Loss candidates are fused-head CE, MLP-head CE, prototype-head CE, and a prototype contrastive term. Although the generic formulation includes all components, the best ablation is fused CE plus contrastive loss.

### 5.2. Findings and transfer

- CLIP generally outperforms a standard ViT.
- The strongest base temporal model is CLIP plus Transformer with 60 frames.
- The prototype variant reaches three-class UAR 74.62% and MF1 65.40% in the cited ablation.
- Binary evaluation: multitask UAR 82.25% for depression and 66.82% for PD; best single-task UAR 85.55% and 78.58%.
- Manually cleaned Test: multitask UAR 83.67%/80.71%; single-task UAR 88.08%/77.96% for depression/PD.

Important failures:

1. annotation–modality mismatch when audio belongs to the speaker but video shows another person;
2. static visual content mistaken for motor impairment;
3. body-detector failure causing background/text features instead of human features.

Transfer to WSM:

- body-centric preprocessing and frozen CLIP features;
- CLIP plus temporal Transformer as video experiment V1;
- prototype-aware head as the second and final video family V2;
- task-specific evidence visualization and failure analysis.

Do not transfer the three-class softmax, reinterpret one corpus's healthy label as negative for the other disease, or select on cleaned Test. The final system must output two independent logits with an observed-label mask.

## 6. TACME

TACME learns emotion, personality, and ambivalence from three independently collected corpora. A sample need not carry labels for other tasks; an availability mask includes only the observed task loss.

Inputs are transcript (T), VLLM-generated observable-behavior description (D), visual behavior (V), and acoustic/prosodic behavior (A).

Architecture:

1. Reduce each variable-length stream to mean and standard-deviation tokens.
2. Project tokens to a common space.
3. Build one cross-attention expert \(E_{m\to n}\) per ordered modality pair: 12 experts for four modalities.
4. Pool each expert with a residual from its query modality.
5. Use a task-specific softmax gate over experts.
6. Feed the weighted mixture to a task-specific head.

A→V and V→A have different parameters. Feature-level and projected-prediction-level flow matching stabilize observed-task representations but do not recover an unknown disease label.

PAGB uses relative validation progress, degradation from the best validation score, and an EMA of task-gradient norms. Slow or degrading tasks receive more weight; normalization, clipping, and EMA smoothing limit instability.

Reported findings:

- task-aware TACME plus feature/prediction FM plus PAGB has average rank 1.00 among compared fusion models;
- single-task TACME beats multitask TACME on all six source-corpus metrics;
- multitask training reduces corpus specialization and improves zero-shot MELD UAR from 28.9% to 35.9%.

Limitations:

- task identity is confounded with corpus identity;
- the missing task head gets no direct pseudo-target;
- mean/std may erase micro-tremor and short facial events;
- experts scale as \(|M|(|M|-1)\);
- VLLM descriptions may carry interpretive/demographic bias;
- weakly related tasks may suffer negative transfer.

Use TACME as a strong fusion baseline: a common A/V/T/D interface, directed experts, task gates, observed-task masking, cheap mean/std aggregation, and PAGB. RAMPS must go further by giving the missing disease head a direct but reliability-filtered gradient.

## 7. Required Baseline Ladder

1. Frozen existing audio baseline.
2. DEPART-like CLIP plus Transformer video model.
3. One prototype-aware video variant.
4. At most two text/description configurations.
5. Sparse masked MTL with two sigmoid outputs.
6. TACME-like task-aware fusion.
7. Calibrated cross-corpus pseudo-supervision.
8. Semantic/LLM evidence as auxiliary evidence, not diagnosis.
9. Reliability-aware multi-objective balancing.

Central research question:

> Can a multimodal sample from the depression corpus safely train the Parkinson head, and vice versa, when pseudo-supervision is accepted only under independent modality/semantic agreement and its uncertainty controls its gradient contribution?

A positive answer requires more than higher DEV/Mean_Score: pseudo-label reliability, calibration, negative transfer, and multi-seed stability must also improve.
