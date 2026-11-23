# AV-002 Trimodal data layer

Status: implemented as a read-only data layer. No training, GPU job, ASR,
re-extraction, or upstream retraining was run.

## Files and schemas

The Dataset is in `src/fusion/data/carewsm_trimodal_dataset.py` and the
canonical fold DataModule is in `src/fusion/data/carewsm_trimodal_datamodule.py`.
The explicit default sources are `audio_id1`, `video_4`, and `text_id9`; the
other audited sources can be selected as `audio_id2`, `audio_id4`, or
`text_id10`. All sources load the `shared_features` key by default and never
load supervised `logits` or `probs`.

AV-001 confirmed audio `[1,64]`, video_4 `[1,128]`, and text `[1,64]` float32
static representations. The loader preserves dtype and shape, rejects
non-finite values, and accepts genuine `[T,D]` arrays without repeating static
tokens. Missing artifacts are excluded by complete-case selection; unreadable,
invalid, or schema-incompatible artifacts raise instead of being zero-filled.

## Splits and targets

The DataModule joins by `(corpus, sample_id)` and validates profile, video,
label, and subgroup identity. It creates joint `train`, separate `val_care`,
`val_wsm`, `test_care`, and `test_wsm` views from the unchanged five folds and
fixed test files. It checks profile isolation between the views.

AV-001 coverage for all three audited families was CARE 1,930/2,031 and WSM
8,243/8,343; per-fold complete-case counts remain those in the AV-001 report.
These are audit counts, not a new extraction or training result.

WSM masks use `splits/videos.csv` `y_depression` and `y_parkinson` columns;
empty values remain unobserved. CARE has only `label_id`, so both target masks
are explicitly `False` and targets are `NaN` with provenance
`UNKNOWN / BLOCKED`; no CARE disease mask semantics were guessed.

`trimodal_collate` uses Chimera `MaskingCollate` for batch padding and adds
video lengths, valid/padding masks, and audio/text lengths. Static inputs keep
singleton length one. Metadata retains corpus/sample/profile identity locally;
public logs are not produced by this layer.

## Verification and limits

Focused tests cover three-modality loading, missing/invalid artifacts, no
silent imputation, target masks, sequence padding, and Chimera registration.
The project Python environment provides NumPy, PyTorch, and Chimera-ML. A
representative real artifact schema was read locally during implementation;
no real training batch was run. Full real DataModule construction was not
claimed as a smoke result because it performs a large complete-case join.

Upstream FIT/SELECTION provenance for the supervised legacy representations
remains `UNKNOWN / BLOCKED`, exactly as recorded by AV-001. This Dataset is
not leakage-free and is not ready for scientifically valid model selection.
AV-003 may implement only separately authorized model/data-compatible paths;
it must preserve this provenance block until an approved leakage design exists.
