# AV-001 Feature and Provenance Audit

Date: 2026-11-23. Read-only audit; no training, ASR, diarization inference,
re-extraction, or embedding generation was run.

## Result

Audio, video, and text artifacts exist in both corpora, but unbiased trimodal
model selection is **BLOCKED**. Legacy fold-specific outputs are static
supervised representations with logits. WSM also contains genuine temporal
video tensors. Upstream FIT and checkpoint-SELECTION participant rosters are
not available in an auditable ledger.

## Confirmed schemas

| Artifact | Schema | Semantics |
| --- | --- | --- |
| `audio_id1/id2/id4_fold0..4.pickle`, CARE/WSM | shared, Depression, PD `[1,64]` float32; logits/probs `[2]` float32; finite in inspected fold/model variants | recording-level singleton, not a sequence |
| `video_4_fold0..4.pickle`, CARE/WSM | shared, Depression, PD `[1,128]` float32; logits/probs `[2]` float32; finite in inspected variants | static recording-level output |
| `text_id9/id10_fold0..4.pickle`, CARE/WSM | shared, Depression, PD `[1,64]` float32; logits/probs `[2]` float32; finite in inspected variants | static output; no timestamps/masks |
| WSM ViT feature store | `video_temporal [128,768]`, `video_cls [768]`, torch float32, metadata | real temporal/frame sequence; no explicit per-token timestamps/mask |
| WSM body/full-frame clip store | representative `video_temporal [532,768]`, pooled `[768]`, torch float32, extraction metadata | variable sequence; metadata includes FPS/frame counts and model/detector fields |

Frozen Mantis, TiViT, BioFormer, and Gated video are not accepted as present
without matching current files/checkpoints.

## Coverage and joins

Counts are unique samples with all five fold files for the legacy family,
joined by authoritative sample/profile/video metadata without imputation.

| Corpus | Metadata | Audio | Video | Text | A+V | A+T | V+T | A+V+T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CARE | 2,031 | 1,930 | 2,031 | 2,031 | 1,930 | 1,930 | 2,031 | 1,930 |
| WSM | 8,433 | 8,343 | 8,348 | 8,285 | 8,287 | 8,275 | 8,253 | 8,243 |

| Split | Rows | CARE A+V+T | WSM A+V+T |
| --- | ---: | ---: | ---: |
| train | 1,437 / 6,133 | 1,362 | 5,993 |
| dev | 276 / 936 | 264 | 909 |
| test | 318 / 1,364 | 304 | 1,341 |

Historical audio complete-case counts are reproduced: CARE 1,930 and WSM
8,343. CARE has no `splits/videos.csv`; its train/dev/test rows were used.
Both corpora have five fold train/val files. Profile/label-mask acceptance
checks remain required in AV-002.

## Text provenance and risks

CARE code documents transcript input aligned by WhisperX, DiariZen speaker
segmentation, longest-speaking-speaker participant assignment, participant
turn extraction, and MPNet `all-mpnet-base-v2` turn embeddings with long-turn
chunk averaging. The legacy text pickle has only static supervised outputs;
raw text, tokens, timestamps, masks, language, and role metadata are absent.
No verified WSM text source/role ledger was found separate from the legacy
outputs. Language mixture, interviewer exclusion, and target-revealing content
are therefore **UNKNOWN / BLOCKED**. Text must not be assumed participant-only.

## Upstream leakage matrix

| Artifact | FIT overlap | SELECTION overlap | Status for every outer fold |
| --- | --- | --- | --- |
| audio `id1/id2/id4` | likely same-fold in-sample; roster unavailable | own VAL selection reported; roster unavailable | **UNKNOWN / BLOCKED** |
| video `video_4` | unknown | unknown | **UNKNOWN / BLOCKED** |
| text `id9/id10` | unknown | unknown | **UNKNOWN / BLOCKED** |
| WSM temporal video stores | metadata documents extractor fields, not participant FIT/selection | unknown | **UNKNOWN / BLOCKED** |

Fold-specific names do not prove OOF behavior. A checkpoint selected on outer
VAL is not independent downstream validation evidence.

## Architecture readiness

| Model | Status | Basis |
| --- | --- | --- |
| Text-Guided Dual-Gated Fusion | CONDITIONAL | Static A/V/T channel/FiLM and inter-modality gates are possible, but text shortcuts and provenance must be resolved. |
| Tri-modal Cross-Attention Residual Fusion | CONDITIONAL | Real WSM video sequences exist; legacy audio/text are singleton static outputs and text sequences/masks are unconfirmed. No fake tokens or synchronization. |
| Shared–Private Tri-modal Fusion | CONDITIONAL | Six projected components are structurally possible, but semantics, missing-modality policy, and leakage remain unresolved. Reuse masked equal BCE; no contrastive loss here. |

None is `SUPPORTED` for valid training yet; no architecture was implemented.

## Nested cross-fitting proposal

Within each unchanged outer TRAIN, fit each supervised upstream encoder on
internal-train, select checkpoints on internal-validation, and generate OOF
representations for every outer-TRAIN sample. Keep outer VAL/TEST untouched by
upstream fit/selection. Frozen pretrained models still require pretraining and
finetuning provenance plus lexical leakage review. This costs at least one
upstream fit/selection cycle per internal split, per selected encoder, per
outer fold, followed by one downstream five-fold study; no retraining is
authorized in AV-001.

## Blockers and next step

Missing FIT/SELECTION participant ledgers, unresolved text role/origin and
label-shortcut audit, absent masks/timestamps for legacy static outputs, and
incomplete canonical fold/profile validation block production. Next atomic
step: independent review, then an owner-approved AV-002 data-layer design.
