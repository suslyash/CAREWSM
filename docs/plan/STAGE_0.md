# WSM Stage 0 Detailed Plan

Status: **COMPLETE**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 0 — Reproducible base

Tasks:

1. restore the provider required by audio without editing src/audio;
2. create configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml from the best run;
3. verify plugin/registry;
4. verify required instrumentation and dev/mean_score monitor;
5. keep new helper scripts under scripts/<modality>;
6. prove src/audio is unchanged.

Gate: clean plugin import, audio model registered, canonical config valid/self-contained, shared experiment_name present, and empty audio diff.
