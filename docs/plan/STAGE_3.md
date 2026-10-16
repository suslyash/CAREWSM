# WSM Stage 3 Detailed Plan

Status: **COMPLETE**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 3 — At most two text/description families [COMPLETE]

Pre-T1 gate: audit transcript coverage, encoding, language, duplicate/leakage risk, and whether explicit segment-level alignment exists. Do not invent segment text from video-level transcripts.

T1: audited-language transcript encoder with mask-aware pooling and two heads.

T2: T1 plus cached observable description/semantic feature and simple T/D gating. Include prompt/model revision in the fingerprint and manually audit a stratified sample.

Gate: at most two model-family configs, diagnosis-free prompt, one DEV-selected representation. T1 training is forbidden until the transcript granularity/alignment contract is frozen.

Frozen completion: `STAGE-3 TEXT/DESCRIPTION EVIDENCE MATRIX COMPLETE`; `STAGE-3 TWO-FAMILY BUDGET EXHAUSTED`. Authoritative synthesis: [../STAGE3_CLAIM_LEDGER_EN.md](../STAGE3_CLAIM_LEDGER_EN.md). T1 is retained as standalone text evidence; T2 remains blocked; no further Stage-3 family search is authorized.
