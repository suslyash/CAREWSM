# WSM Stage 2 Detailed Plan

Status: **COMPLETE**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 2 — At most two video families

V1: DEPART-like uniform temporal sequence, YOLOv8 single-class human-body ROI detection/cropping per sampled frame, frozen CLIP frame encoding, projection, temporal Transformer, masked pooling, two sigmoid heads.

V2: V1 plus task-specific class prototypes, classwise prototype/MLP gating, and controlled contrastive ablation.

Gate: reproducible preprocessing/cache, extraction failure report, one DEV winner selected without Test.
