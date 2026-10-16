# SOTA Review: Multimodal Multi-Task Multi-Label Learning with Dataset-Level Missing Tasks

Review cutoff: **22 September 2026**.

Problem convention: **?** is an unknown annotation, never a negative class. The central question is whether a sample without Task B annotation can provide a useful direct gradient to Task B instead of merely being ignored by that head.

The source review searched publisher/proceedings pages, IEEE, ACM, ScienceDirect, Springer, CVF, PMLR, AAAI, PubMed/DBLP/OpenAlex, arXiv, and code/project pages. Venue ranks used ICORE/CORE 2026; journal quartiles were stated only when supported by available SJR/SCImago checks. No unsupported citation counts are reported.

## 1. Executive Summary

1. The full intersection—multimodal + multi-task + multi-label + dataset-level missing tasks—remains weakly covered. No strong 2025–2026 work was found that directly solves independently collected Dataset A with only Task A labels and Dataset B with only Task B labels while both datasets train both heads.
2. **PASAL (Information Fusion 2025)** is the closest safe sparse-MTL baseline. Every sample updates the shared representation, but a sample without Task B annotation does not update head B.
3. **HiTTs (ACM MM 2025)** is the clearest direct missing-task mechanism: hierarchical task tokens discover feature- and prediction-level pseudo-labels for unannotated tasks. Its evidence is dense prediction, not multimodal multilabel classification.
4. **UNITI (ESWA 2025)** addresses heterogeneous inter-dataset MTL, feature interference, and forgetting through sequential learning and feature distillation, but does not reconstruct the missing label.
5. Incomplete multi-view/missing-multilabel learning is more mature: recent ICML, CVPR, and TPAMI work combines cross-view information, pseudo-labeling, uncertainty, semantic label embeddings, and label-aware fusion. Multi-view benchmarks are not automatically equivalent to raw multimodal data.
6. Strong anti-confirmation-bias mechanisms include independent/dual branches, class-specific thresholds, evidential uncertainty, and selecting only reliable pseudo-labels.
7. A strong VLM/CLIP line—HST, PositiveCoOp, VA³P, SCINet—supports semantic transfer into an unobserved label space.
8. The best practical design is layered: PASAL-like sparse supervised MTL; HiTTs/teacher-like direct pseudo-supervision; then uncertainty, semantic, and VLM evidence for acceptance.

## 2. Formal Problem

\[
y_{itk}\in\{0,1,?\},
\qquad
m_{itk}=\mathbf 1[y_{itk}\neq ?],
\]

where \(i\) is a sample, \(t\) a task/head, and \(k\) a label within that task.

\[
\mathcal L_{\rm sup} =
\frac{\sum_{i,t,k}m_{itk}\ell(\hat y_{itk},y_{itk})}
{\sum_{i,t,k}m_{itk}}.
\]

No negative BCE term is computed for unknown entries. Masking avoids false negatives but does not answer:

\[
m_{iBk}=0
\Longrightarrow
\text{can sample }i\text{ create a gradient for }h_B?
\]

| Transfer level | Effect of an A-only sample on Task B |
|---|---|
| Ignore | No B loss; no B-head gradient |
| Representation transfer | Shared encoder changes; B head does not |
| Relation/consistency transfer | B receives an indirect constraint |
| Direct recovered supervision | Pseudo/teacher/semantic B target directly updates B |

The final level is the target of this project.

## 3. Partial-Task Multi-Task Learning

### 3.1. HiTTs

J. Zhang et al., *Multi-Task Label Discovery via Hierarchical Task Tokens for Partially Annotated Dense Predictions*, ACM MM 2025, DOI 10.1145/3746027.3755727.

- Setting: partial task annotation for segmentation, depth, normals, and related dense tasks.
- Architecture: multitask backbone, global task tokens, task-specific fine-grained spatial tokens.
- Mechanism: token interactions discover pseudo task labels at feature and prediction levels.
- Transfer: direct cross-task pseudo-supervision, not merely shared-backbone transfer.

For WSM, spatial tokens can become task and label tokens. Dense-task relationships may be stronger than depression↔PD relationships, so semantic and uncertainty evidence is required.

### 3.2. PASAL

T. Gorges et al., *PASAL: Progress- and sparsity-aware loss balancing for heterogeneous dataset fusion*, Information Fusion 120, 103038, 2025, DOI 10.1016/j.inffus.2025.103038.

PASAL uses a shared network, dataset-specific paths/heads, observed-only losses, and progress/sparsity-aware balancing.

For \((x_A,y_A,?)\):

\[
E_{\rm shared}\ \checkmark,\qquad
h_A\ \checkmark,\qquad
h_B\ \times.
\]

It is the required control: a missing-label method must beat correct joint representation learning plus sparse loss balancing.

### 3.3. UNITI

S. Kim et al., *UNITI: Framework for multi-task learning across datasets to mitigate overfitting*, ESWA, 2025, DOI 10.1016/j.eswa.2025.127653.

UNITI uses task/dataset teachers, a shared student, sequential inter-dataset training, and feature-level distillation to reduce interference and forgetting. Reported gains reach 5.17 percentage points in some experiments. It stabilizes learning but does not recover the missing label.

### 3.4. Progressive Pseudo Labeling

K. Ye et al., *Progressive Pseudo Labeling for Multi-Dataset Detection Over Unified Label Space*, IEEE TMM, online 2024/volume 2025, DOI 10.1109/TMM.2024.3521841.

Different detection datasets omit different categories, so missing annotation does not imply absence. Teacher–student pseudo-labels, progressive filtering, confidence/entropy treatment, and a unified ontology create direct cross-dataset supervision. The structural analogy is strong despite the detection setting.

### 3.5. Predecessors

- DiffusionMTL, CVPR 2024: partially annotated dense MTL as conditional denoising.
- Region-aware Distribution Contrast, ECCV 2024: region-level distribution alignment across tasks.
- MTPSL, CVPR 2022: cross-task consistency under partial dense annotations.

## 4. Incomplete Multi-Label Learning

Relevant partial multilabel means \(y_k\in\{0,1,?\}\), not a candidate set polluted with false positives.

### 4.1. SATL

H. Ruan et al., *Learning Semantic-Aware Threshold for Multi-Label Image Recognition with Partial Labels*, ESWA, DOI 10.1016/j.eswa.2025.129216.

Pseudo-labels recover unknown entries. Category-aware thresholds replace one fixed threshold because score distributions differ by class. WSM should therefore use separate \(\tau_k^+\) and \(\tau_k^-\).

### 4.2. CLS

G. Lyu et al., *Addressing Multi-Label Learning with Partial Labels: From Sample Selection to Label Selection*, AAAI 2025, DOI 10.1609/aaai.v39i18.34119.

Two networks use Co-Label Selection to correct labels rather than discard whole uncertain samples. This reduces single-network confirmation bias. Its missing-positive assumption is narrower than WSM's positive/negative/unknown setting.

### 4.3. LogicMix

C. F. Chong et al., *LogicMix: Sample mixing data augmentation for multi-label image classification with partial labels*, Pattern Recognition 171, 112186, 2026, DOI 10.1016/j.patcog.2025.112186.

Logic-preserving mixing avoids turning missing labels into negatives. It is useful regularization but does not create a strong cross-task signal.

### 4.4. HST

T. Chen et al., *Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels*, IJCV 132:6091–6106, 2024, DOI 10.1007/s11263-024-02127-2.

HST reconstructs unknown labels through instance-specific label co-occurrence and category prototypes. Known and recovered labels jointly train the classifier. It establishes that unknown matrix entries can be recoverable supervision rather than only masked positions.

### 4.5. VLM/prompt methods

- **PositiveCoOp, WACV 2025:** CLIP prior under partial labels. Negative prompts may hurt because VLM pretraining models absence poorly. Unknown must not become textual “no X.”
- **VA³P, ACM TOMM 2026:** local visual–label alignment and attribute-aware prompts.
- **SCINet, IEEE TMM 2026:** correct/incorrect/unknown labels, text–image alignment, label/instance co-occurrence, and semantic augmentation.

## 5. Incomplete Multi-View Multi-Label Learning

Multi-view can include multimodal data, but many benchmarks use alternative feature views of one object. Adapt evidence carefully to raw A/V/T/D.

### 5.1. ICML 2025 compact semantics

J. Wen et al., *Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label Classification*, ICML 2025, PMLR 267:66467–66480.

The method preserves shared task-relevant information, suppresses redundant information, and uses dual-branch soft pseudo-label cross-imputation. An independent branch supplies pseudo-labels rather than self-confirming the same branch.

### 5.2. CVPR 2025 disentanglement

X. Yan et al., *Incomplete Multi-View Multi-label Learning via Disentangled Representation and Label Semantic Embedding*, CVPR 2025, DOI 10.1109/CVPR52734.2025.02861.

Uses a VAE, cross-view reconstruction, mutual-information separation of consistent/specific factors, label embeddings, and label-relevance topology. Label completion can be built into representation geometry, not only prediction thresholds.

### 5.3. URDF

J. Wen et al., *Partial Multiview Incomplete Multilabel Learning via Uncertainty-Driven Reliable Dynamic Fusion*, TPAMI 48(1):236–250, 2026, DOI 10.1109/TPAMI.2025.3603677.

Sample-level fusion weights depend on view uncertainty/reliability, while trustworthy pseudo-labels supervise unknown labels. Uncertainty should therefore govern both fusion and pseudo-target acceptance.

### 5.4. CTRL

Y. Liu et al., *Learning Compact Semantic Information and Reliable Pseudo-Labels for Incomplete Multi-View Multi-Label Classification*, TPAMI 48(7):7575–7589, 2026, DOI 10.1109/TPAMI.2026.3665813.

CTRL combines compact shared representation, a Beta Evidential Neural Network, Dempster–Shafer evidence, label-level uncertainty, and reliability-filtered pseudo-labels.

### 5.5. DCSI

*Disentangling Consistent and Specific Information for Double Incomplete Multi-View Multi-Label Classification*, TPAMI 48(7):7307–7320, 2026, DOI 10.1109/TPAMI.2026.3665097.

Preserves consistent and view-specific information. Excessive invariance can erase useful corpus/modality-specific evidence.

### 5.6. V2L

C. Liu et al., *When Semantically Consistent Encoding Meets View-Label Heterogeneity Modeling*, TPAMI ahead of print, 31 August 2026, DOI 10.1109/TPAMI.2026.3728832.

V2L combines semantically consistent variational encoding, view-specific evidence, and hybrid representation/decision fusion:

\[
w_{i,v,k}=\text{relevance of view }v
\text{ for label }k\text{ on instance }i.
\]

This motivates label- and instance-specific modality reliability. The paper is too recent for citation-influence claims.

### 5.7. TACVI-Net and CDSA

- TACVI-Net, Neural Networks 187, 107349, 2025: task-relevant representation and cross-view autoencoder imputation.
- CDSA, Knowledge-Based Systems 318, 113507, 2025: dynamic confidence weighting and label-informed semantic alignment.

Both focus more on missing views than dataset-level missing tasks.

## 6. Cross-Family Comparison

| Family | MTL | Multi-dataset | Multimodal/view | Multilabel | Direct missing-head signal |
|---|---:|---:|---:|---:|---|
| PASAL | yes | yes | architecture-agnostic | possible | no |
| UNITI | yes | yes | not central | not central | KD, no missing target |
| HiTTs | yes | partial-task data | vision | no | yes, pseudo |
| PPL | multi-dataset | yes | vision | unified categories | yes, pseudo |
| ICML 2025 MvML | no | no | multi-view | yes | yes, unknown labels |
| URDF/CTRL | no | no | multi-view | yes | yes, reliable pseudo |
| SCINet/VLM | no | no | VLM prior | yes | yes, semantic |

Required mechanisms exist but are distributed across literature lines. No mature general framework was found for multimodal multi-task multilabel heterogeneous dataset-level missing supervision.

## 7. Mechanism Taxonomy

    Heterogeneous partial supervision
    ├── Route observed supervision
    │   ├── masked/selective loss
    │   ├── dataset-specific heads
    │   └── sparse/progress balancing
    ├── Share representations
    │   ├── shared multimodal trunk
    │   ├── consistent/specific disentanglement
    │   ├── cross-view reconstruction
    │   └── task/label-conditioned fusion
    ├── Recover missing supervision
    │   ├── task-token discovery
    │   ├── teacher/EMA pseudo-labeling
    │   ├── dual-network pseudo-labeling
    │   ├── evidential uncertainty
    │   └── adaptive thresholds
    ├── Use task/label structure
    │   ├── cross-task consistency
    │   ├── co-occurrence graphs and prototypes
    │   ├── semantic embeddings
    │   └── VLM priors
    ├── Use modality/view structure
    │   ├── reconstruction and mutual information
    │   ├── confidence-weighted fusion
    │   ├── label-dependent view selection
    │   └── cross-modal alignment
    └── Stabilize optimization
        ├── sparse/progress balancing
        ├── knowledge distillation
        ├── sequential/balanced training
        └── reliability weighting

Missing-supervision recovery must be evaluated separately from shared-representation benefit.

## 8. Key Mechanisms

Masked loss:

\[
\mathcal L_{\rm obs}
=\sum_{i,t,k}m_{itk}\lambda_t
\operatorname{BCE}(p_{itk},y_{itk}).
\]

It is safe, but the missing head gets no target.

Direct pseudo-labeling:

\[
\tilde y_{iBk}=T_B(x_i),\qquad
q_{iBk}=\mathbf 1[\operatorname{confidence}(\tilde y_{iBk})>\tau_{Bk}],
\]

\[
\mathcal L_{\rm pseudo}
=\sum_{i,t,k}(1-m_{itk})q_{itk}
\operatorname{BCE}(p_{itk},\tilde y_{itk}).
\]

Then \(\nabla_{\theta_B}\mathcal L_{\rm pseudo}\neq0\) for an A sample.

Recommended confirmation-bias controls:

- separately trained or EMA teacher;
- independently initialized branches;
- class-specific positive/negative thresholds;
- evidential/ensemble uncertainty;
- confidence warm-up;
- consistency across augmentations/modalities;
- calibration only on observed labels.

Use text descriptions \(e_k=E_{\rm text}(\text{label description})\) to relate labels never observed together. VLM evidence

\[
s_k(x)=\operatorname{sim}(E_{\rm vision}(x),E_{\rm text}(\ell_k))
\]

is auxiliary evidence, not ground truth. Negative semantic prompts require extra caution.

## 9. Application to Dataset A + Dataset B

### Strategy 1 — sparse multimodal MTL

\[
z=F(E_I(x^I),E_T(x^T)),\qquad
p_A=h_A(z),\quad p_B=h_B(z).
\]

An A sample updates encoders/fusion/shared trunk/head A, but not head B.

### Strategy 2 — task/label-token discovery

Adapt HiTTs to task tokens, label tokens, cross-task attention, and multimodal keys/values. High-confidence B predictions for an A sample become pseudo-supervision. Add semantics because disease-task relations are weak.

### Strategy 3 — cross-dataset teacher–student

\[
T_A\leftarrow D_A,\quad T_B\leftarrow D_B,\qquad
\tilde y_A^B=T_B(x_A),\quad \tilde y_B^A=T_A(x_B).
\]

\[
\mathcal L=\mathcal L_{\rm observed}
+\mu(t)\mathcal L_{\rm pseudo}
+\gamma\mathcal L_{\rm consistency}
+\delta\mathcal L_{\rm KD}.
\]

Combine PPL teachers, ICML dual branches, CTRL/URDF uncertainty, SATL thresholds, and UNITI distillation.

### Strategy 4 — semantic/VLM bridge

\[
e_{Ak}=E_{\rm label}(d_{Ak}),\quad
e_{Bl}=E_{\rm label}(d_{Bl}),\quad
s_k=\operatorname{sim}(z,e_k).
\]

HST, label-topology methods, SCINet, PositiveCoOp, and VA³P support this direction. VLM evidence is better supported than free-form LLM medical annotation.

### Strategy 5 — stabilization

Add task heads/adapters, a controlled shared trunk, PASAL balancing, teacher feature distillation, and balanced/sequential corpus batches.

| Method | Shared encoder | Observed head | Missing head |
|---|---:|---:|---:|
| Masked/PASAL | yes | yes | no |
| UNITI KD | yes + KD | yes | usually no direct target |
| HiTTs-like | yes | yes | yes, pseudo |
| Teacher–student | yes | yes | yes, accepted pseudo |
| VLM teacher | yes | yes | yes, semantic/pseudo |

## 10. Research Gaps

1. Full intersection of multimodal, MTL, multilabel, and genuine dataset-level missing tasks.
2. Controlled direct supervision across fully disjoint corpora.
3. Methods beyond masking: task tokens, teachers, uncertainty, semantic topology, VLM priors.
4. Joint treatment of pseudo-label reliability and negative transfer.
5. Semantic bridges when empirical \(P(B|A)\) is unobservable.
6. Label-specific modality reliability \(w_{i,m,t,k}\), not one global weight.

## 11. Promising Contributions

### A. Uncertainty-calibrated cross-task pseudo-supervision

Combine direct label discovery, teacher–student transfer, evidential uncertainty, and class-specific thresholds:

\[
\text{direct pseudo transfer + label-specific uncertainty}
>
\text{masked shared MTL}.
\]

### B. Semantic bridge for disjoint ontologies

Combine separate heads, shared multimodal encoding, textual label descriptions, VLM embeddings, and semantic relation transfer.

### C. Modality-consistent acceptance

Require agreement among audio/video/text/description or unimodal/fusion predictions according to learned label-wise reliability, not one probability threshold.

### D. Joint sparse-gradient and reliability optimization

\[
\lambda_t=f(\text{task progress},\text{sparsity},
\text{pseudo uncertainty}).
\]

### E. Benchmark protocol

Use independently collected datasets with genuinely disjoint annotations, real modalities, and a small dual-annotated audit subset only for predeclared evaluation/calibration. Separately report observed-task performance, missing-task recovery, calibration, pseudo-label precision, negative transfer, and unseen cross-dataset combinations.

## 12. Key Bibliography

1. Zhang et al. *Multi-Task Label Discovery via Hierarchical Task Tokens for Partially Annotated Dense Predictions.* ACM MM 2025. DOI 10.1145/3746027.3755727.
2. Gorges et al. *PASAL: Progress- and sparsity-aware loss balancing for heterogeneous dataset fusion.* Information Fusion 120, 103038, 2025. DOI 10.1016/j.inffus.2025.103038.
3. Kim et al. *UNITI: Framework for multi-task learning across datasets to mitigate overfitting.* ESWA, 2025. DOI 10.1016/j.eswa.2025.127653.
4. Ye et al. *Progressive Pseudo Labeling for Multi-Dataset Detection Over Unified Label Space.* IEEE TMM. DOI 10.1109/TMM.2024.3521841.
5. Wen et al. *Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label Classification.* ICML 2025, PMLR 267:66467–66480.
6. Yan, Yin, Wen. *Incomplete Multi-View Multi-label Learning via Disentangled Representation and Label Semantic Embedding.* CVPR 2025. DOI 10.1109/CVPR52734.2025.02861.
7. Wen et al. *Partial Multiview Incomplete Multilabel Learning via Uncertainty-Driven Reliable Dynamic Fusion.* TPAMI 48(1):236–250, 2026. DOI 10.1109/TPAMI.2025.3603677.
8. Liu et al. *Learning Compact Semantic Information and Reliable Pseudo-Labels for Incomplete Multi-View Multi-Label Classification.* TPAMI 48(7):7575–7589, 2026. DOI 10.1109/TPAMI.2026.3665813.
9. Wen et al. *Disentangling Consistent and Specific Information for Double Incomplete Multi-View Multi-Label Classification.* TPAMI 48(7):7307–7320, 2026. DOI 10.1109/TPAMI.2026.3665097.
10. Wen et al. *Multi-Domain Feature Integration Based Trusted Partial Multi-View Incomplete Multi-Label Learning.* TPAMI online 2026. DOI 10.1109/TPAMI.2026.3692653.
11. Liu et al. *When Semantically Consistent Encoding Meets View-Label Heterogeneity Modeling.* TPAMI ahead of print, 2026. DOI 10.1109/TPAMI.2026.3728832.
12. Zhao et al. *Task-augmented cross-view imputation network for partial multi-view incomplete multi-label classification.* Neural Networks 187, 107349, 2025. DOI 10.1016/j.neunet.2025.107349.
13. Chen et al. *Confidence-Enhanced Dual-Space Semantic Alignment for partial multi-view incomplete multi-label classification.* KBS 318, 113507, 2025. DOI 10.1016/j.knosys.2025.113507.
14. Ruan et al. *Learning Semantic-Aware Threshold for Multi-Label Image Recognition with Partial Labels.* ESWA. DOI 10.1016/j.eswa.2025.129216.
15. Lyu et al. *Addressing Multi-Label Learning with Partial Labels: From Sample Selection to Label Selection.* AAAI 2025. DOI 10.1609/aaai.v39i18.34119.
16. Chong et al. *LogicMix: Sample mixing data augmentation for multi-label image classification with partial labels.* Pattern Recognition 171, 112186, 2026. DOI 10.1016/j.patcog.2025.112186.
17. Chen et al. *Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels.* IJCV 132:6091–6106, 2024. DOI 10.1007/s11263-024-02127-2.
18. Rawlekar et al. *PositiveCoOp: Rethinking Prompting Strategies for Multi-Label Recognition with Partial Annotations.* WACV 2025. DOI 10.1109/WACV61041.2025.00572.
19. Sun et al. *Visual–Label Alignment and Attribute-Aware Prompt for Multi-Label Image Recognition with Partial Labels.* ACM TOMM 22(8), 2026. DOI 10.1145/3828542.
20. Wu et al. *Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge (SCINet).* IEEE TMM, 2026. DOI 10.1109/TMM.2026.3705189.

## 13. Practical Conclusion

\[
\boxed{\text{1. masked sparse MTL}}
\rightarrow
\boxed{\text{2. calibrated cross-dataset pseudo-labels}}
\rightarrow
\boxed{\text{3. semantic/VLM label transfer}}
\rightarrow
\boxed{\text{4. uncertainty + interference-aware weighting}}.
\]

Level 1 measures shared-representation benefit. Level 2 asks whether \(x_A\) directly trains Task B. Level 3 supports disjoint annotations. Level 4 targets confirmation bias and negative transfer. The components exist, but not yet a mature general multimodal, multilabel, multitask, multi-dataset S2/S6 framework.
