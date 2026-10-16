# WSM Stage 1 Detailed Plan

Status: **COMPLETE**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 1 — Manifest and partial-label contract

Create one manifest with:

    segment_id, video_id, speaker_id, corpus, split,
    y_depression, y_parkinson,
    observed_depression, observed_parkinson,
    audio_available, video_available,
    text_available, description_available

Add audit reports, cache fingerprints, unified split identity, and separate DEV/Test dataloaders.

Gate: no video split leakage; current train/dev/test speaker independence is accepted from the dataset-owner split contract despite unavailable speaker_id and must be recorded as an assumption rather than a measured identity audit; unknown never maps to zero; counts/missing streams saved machine-readably; Test absent from fit validation.
