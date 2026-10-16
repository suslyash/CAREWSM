# WSM Stage 4 Detailed Plan

Status: **COMPLETE**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 4 — Honest multimodal baselines [COMPLETE — NO SAFE STRONG-AUDIO FUSION WINNER]

- F0: simple masked late/gated fusion.
- F1: sparse masked two-head MTL with no pseudo-labeling.
- F2: TACME-like task-aware relation bank with observed loss; add PAGB comparator only if budget permits.
- Owner-authorized pre-RAMPS strong-audio ablation: replace the pooled `audio_cls` fusion branch with the frozen DEV-selected historical temporal-audio checkpoint through a fusion-side adapter; do not edit `src/audio`.
- The first strong-audio residual run produced a valid negative result, but its randomly initialized residual path did not start from the exact frozen-audio function. The owner therefore authorizes one bounded audio-first temporal-fusion research sprint before RAMPS rather than continuing one hand-designed model at a time.
- The sprint may implement/train up to three seed-42 candidate fusion models in one manager task. All candidates MUST keep the exact historical temporal audio checkpoint frozen, MUST start with final logits exactly equal to the frozen audio base logits, and MUST use video only through a zero-initialized additive correction path.
- Candidate architectures are chosen by Codex from a manager-bounded design space emphasizing the stronger audio modality: zero-init residual fusion, task-specific audio-query temporal-video attention, audio-confidence/gated video correction, and/or one F2-like directed temporal relation variant. Exact internal details may vary within the fixed parameter/compute caps, but the candidate manifest MUST be frozen before the first training run.
- All candidates use the same canonical data, seed=42 screening protocol, trainable-only AdamW, DEV-only selector, and four evaluation streams every epoch. TEST_NONE/SOFT/HARD remain monitoring-only and MUST NOT influence candidate design/ranking.
- The screening goal is to exceed frozen audio DEV Mean_Score=0.7878268765 while avoiding a >1.0 absolute percentage-point DEV Score drop on either task. A single-seed screen may nominate a candidate only; promotion still requires later multi-seed confirmation.
- RAMPS Stage 5 is deferred until this bounded audio-first temporal-fusion sprint is completed or explicitly stopped by the manager.

Gate: same canonical splits/metric and DEV-only selector; exact frozen-audio reproduction at initialization for every candidate; no Test-driven search; at most three screened candidate models; preserve all negative results; nominate at most one DEV winner for later multi-seed confirmation.
