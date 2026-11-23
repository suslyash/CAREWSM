# AV-003 Trimodal Models

## Status

Implemented as provisional code-only fusion models. Upstream FIT/SELECTION provenance and CARE observed-label semantics remain `UNKNOWN / BLOCKED`; this report makes no scientific-validation or leakage-free claims.

## Models

- `carewsm_text_guided_dual_gated`: independent modality projections, text-conditioned audio/video channel gates, learned modality gate, residual paths, and independent task heads.
- `carewsm_trimodal_cross_attention`: projected tokens with mask-safe cross-modal attention for genuine multi-token inputs and a named `static_fallback` for all-singleton inputs. Residual unimodal paths feed independent task heads.
- `carewsm_shared_private_trimodal`: shared and private projected components for each modality, learned six-component gating, residual aggregate, and independent task heads.

## Interface

Models consume `batch.inputs["audio"]`, `batch.inputs["video"]`, and `batch.inputs["text"]` with `[B,T,D]` shapes; dimensions are explicit config parameters (audited defaults 64/128/64). They use AV-002 modality lengths and `video_valid_mask` where available, return `ModelOutput(preds=[B,2], aux=...)`, and never read targets or metadata as predictive inputs.

## Configurations

- `configs/fusion/av003_carewsm_text_guided_dual_gated.yaml`
- `configs/fusion/av003_carewsm_trimodal_cross_attention.yaml`
- `configs/fusion/av003_carewsm_shared_private_trimodal.yaml`

## Verification

No-training tests cover registry construction, finite `[B,2]` output, gradients, modality sensitivity, padding masks, singleton/static fallback, genuine multi-token attention, independent task outputs, existing masked BCE/metrics compatibility, missing modalities, and invalid dimensions. Real artifact smoke is not run in AV-003 because upstream provenance is blocked and no training/evaluation is authorized.

## Limitations and next step

The implementations are adaptations for the AV-002 batch contract, not reproduction claims for cited papers. No contrastive or auxiliary supervised loss is added. AV-004 is the next atomic step; training remains out of scope until the owner approves a leakage-safe execution strategy.
