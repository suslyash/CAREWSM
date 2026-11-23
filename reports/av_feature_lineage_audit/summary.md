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
Both corpora have five fold train/val files. Profile disjointness is checked
and zero for every TRAIN/VAL, TRAIN/TEST, and VAL/TEST comparison. CARE
observed-task mask semantics remain UNKNOWN because CARE provides only
`label_id`, not independent Depression/Parkinson observed-mask columns.

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

## Corrective Round 1: fold and split integrity

The authoritative fold files were read without modification. Each cell is population / audio / video / text / A+V+T; TEST is the fixed test population repeated for each outer fold.

### CARE

| Fold | Train | Val | Test |
| ---: | --- | --- | --- |
| 0 | 1361 / 1299 / 1361 / 1361 / 1299 | 352 / 327 / 352 / 352 / 327 | 318 / 304 / 318 / 318 / 304 |
| 1 | 1368 / 1296 / 1368 / 1368 / 1296 | 345 / 330 / 345 / 345 / 330 | 318 / 304 / 318 / 318 / 304 |
| 2 | 1378 / 1308 / 1378 / 1378 / 1308 | 335 / 318 / 335 / 335 / 318 | 318 / 304 / 318 / 318 / 304 |
| 3 | 1382 / 1312 / 1382 / 1382 / 1312 | 331 / 314 / 331 / 331 / 314 | 318 / 304 / 318 / 318 / 304 |
| 4 | 1363 / 1289 / 1363 / 1363 / 1289 | 350 / 337 / 350 / 350 / 337 | 318 / 304 / 318 / 318 / 304 |

### WSM

| Fold | Train | Val | Test |
| ---: | --- | --- | --- |
| 0 | 5640 / 5571 / 5580 / 5530 / 5499 | 1429 / 1416 / 1417 / 1410 / 1403 | 1364 / 1356 / 1351 / 1345 / 1341 |
| 1 | 5661 / 5599 / 5603 / 5560 / 5530 | 1408 / 1388 / 1394 / 1380 / 1372 | 1364 / 1356 / 1351 / 1345 / 1341 |
| 2 | 5670 / 5606 / 5617 / 5570 / 5544 | 1399 / 1381 / 1380 / 1370 / 1358 | 1364 / 1356 / 1351 / 1345 / 1341 |
| 3 | 5640 / 5576 / 5585 / 5539 / 5505 | 1429 / 1411 / 1412 / 1401 / 1397 | 1364 / 1356 / 1351 / 1345 / 1341 |
| 4 | 5665 / 5596 / 5603 / 5561 / 5530 | 1404 / 1391 / 1394 / 1379 / 1372 | 1364 / 1356 / 1351 / 1345 / 1341 |

After deduplication, source metadata has 2,031 CARE and 8,433 WSM unique sample IDs. Known duplicated source rows are train-file DEV repetitions: CARE 276 and WSM 936. Artifact-to-sample mappings had zero duplicates for every family and fold. Profile intersections were zero for TRAIN-VAL, TRAIN-TEST, and VAL-TEST in every outer fold and both corpora.

CARE contains only label_id (CONTROL, DEPRESSION, PARKINSON) and no independent Depression/Parkinson observed-mask columns; those masks are UNKNOWN. WSM videos.csv contains y_depression and y_parkinson; empty values are unobserved, not negative. WSM observed-label totals are Depression 5,111 and Parkinson 3,322.

## Corrective Round 1: exclusion reasons

Marginal missing counts overlap. Each exclusion tuple is missing-A / missing-V / missing-T / AV-without-T / AT-without-V / VT-without-A / missing-all.

| Corpus | Fold | Split | Exclusion tuple |
| --- | ---: | --- | --- |
| CARE | 0 | train | 62 / 0 / 0 / 0 / 0 / 62 / 0 |
| CARE | 0 | val | 25 / 0 / 0 / 0 / 0 / 25 / 0 |
| CARE | 0 | test | 14 / 0 / 0 / 0 / 0 / 14 / 0 |
| CARE | 1 | train | 72 / 0 / 0 / 0 / 0 / 72 / 0 |
| CARE | 1 | val | 15 / 0 / 0 / 0 / 0 / 15 / 0 |
| CARE | 1 | test | 14 / 0 / 0 / 0 / 0 / 14 / 0 |
| CARE | 2 | train | 70 / 0 / 0 / 0 / 0 / 70 / 0 |
| CARE | 2 | val | 17 / 0 / 0 / 0 / 0 / 17 / 0 |
| CARE | 2 | test | 14 / 0 / 0 / 0 / 0 / 14 / 0 |
| CARE | 3 | train | 70 / 0 / 0 / 0 / 0 / 70 / 0 |
| CARE | 3 | val | 17 / 0 / 0 / 0 / 0 / 17 / 0 |
| CARE | 3 | test | 14 / 0 / 0 / 0 / 0 / 14 / 0 |
| CARE | 4 | train | 74 / 0 / 0 / 0 / 0 / 74 / 0 |
| CARE | 4 | val | 13 / 0 / 0 / 0 / 0 / 13 / 0 |
| CARE | 4 | test | 14 / 0 / 0 / 0 / 0 / 14 / 0 |
| WSM | 0 | train | 69 / 60 / 110 / 32 / 23 / 8 / 20 |
| WSM | 0 | val | 13 / 12 / 19 / 4 / 5 / 2 / 3 |
| WSM | 0 | test | 8 / 13 / 19 / 8 / 4 / 0 / 6 |
| WSM | 1 | train | 62 / 58 / 101 / 29 / 23 / 7 / 18 |
| WSM | 1 | val | 20 / 14 / 28 / 7 / 5 / 3 / 5 |
| WSM | 1 | test | 8 / 13 / 19 / 8 / 4 / 0 / 6 |
| WSM | 2 | train | 64 / 53 / 100 / 26 / 18 / 8 / 17 |
| WSM | 2 | val | 18 / 19 / 29 / 10 / 10 / 2 / 6 |
| WSM | 2 | test | 8 / 13 / 19 / 8 / 4 / 0 / 6 |
| WSM | 3 | train | 64 / 55 / 101 / 32 / 26 / 8 / 16 |
| WSM | 3 | val | 18 / 17 / 28 / 4 / 2 / 2 / 7 |
| WSM | 3 | test | 8 / 13 / 19 / 8 / 4 / 0 / 6 |
| WSM | 4 | train | 69 / 62 / 104 / 25 / 22 / 9 / 21 |
| WSM | 4 | val | 13 / 10 / 25 / 11 / 6 / 1 / 2 |
| WSM | 4 | test | 8 / 13 / 19 / 8 / 4 / 0 / 6 |

The WSM reduction from 8,343 audio samples to 8,243 A+V+T samples is an intersection result, not a sum of marginal exclusions. CARE's 1,930/2,031 difference is entirely shared missing-audio coverage; video and text cover all 2,031 rows. No separate invalid-value or join-mismatch bucket is proven for unreadable/schema-invalid files outside inspected representatives, so that part remains UNKNOWN.

## Corrective Round 2: reproducibility

The committed read-only script is `scripts/fusion/audit_av001.py`. Run it with
local roots supplied at invocation; no private paths or IDs are stored in its
output or Git:

~~~sh
python3 scripts/fusion/audit_av001.py \
  --care-root "$CARE_ROOT" \
  --wsm-root "$WSM_ROOT"
~~~

The script explicitly intersects `audio_id1_fold*.pickle`,
`audio_id2_fold*.pickle`, and `audio_id4_fold*.pickle` for A; uses
`video_4_fold*.pickle` for V; and explicitly intersects both
`text_id9_fold*.pickle` and `text_id10_fold*.pickle` for T. It reports all
five folds, sample/profile joins, profile intersections, A+V+T coverage,
exclusion tuples, and duplicate artifact-to-sample keys. The old
`text_id[910]` glob is intentionally not used because it does not match
`text_id10_fold*.pickle`.

Exact invocation used for this round:

~~~sh
python3 scripts/fusion/audit_av001.py --self-test
python3 scripts/fusion/audit_av001.py --care-root "$CARE_ROOT" --wsm-root "$WSM_ROOT"
~~~

In the local run, `CARE_ROOT` and `WSM_ROOT` were set to the two configured
corpus roots; their absolute values are intentionally not recorded here.

Identity validation uses `(corpus, sample_id)` as the canonical record key.
Repeated train/dev references are accepted only when profile, video, label, and
subgroup agree. Reverse profile/video pairs are checked for ambiguity; fold
TRAIN/VAL and fixed TEST rows are checked against the source sample index; and
artifact paths are resolved through label/subgroup/profile/video context with
the CARE and WSM prefix rules. Invalid sample IDs, metadata mismatches,
ambiguous artifact mappings, duplicate artifact mappings, or profile
intersections return exit code 1 with `BLOCKED`.

The standard-library synthetic regression returned `synthetic_regression=PASS`
for: two sample IDs sharing one profile/video pair; an absent fold sample ID;
profile/video metadata mismatch; and TRAIN/VAL participant overlap. The real
audit exited 0. Its aggregate result was CARE canonical source/artifact
sample IDs 2,031/2,031 and WSM source/sample IDs 8,433/8,404, with zero
artifact IDs without source metadata and zero duplicate artifact-to-sample
keys. Profile intersections were zero in all 30 corpus/fold pair checks. The
script reproduced the report's fold counts and exclusion tuples, including
CARE 1,930/2,031 and WSM 8,243/8,343 complete A+V+T/audio coverage. The
representative pickle shape/dtype recheck is `UNKNOWN` in the available
runtime because NumPy is not installed; no value is inferred from that failed
read.

No repository data artifact, model, split, or temporary output was changed;
the captured aggregate output was outside Git.

The explicit per-row upstream FIT/SELECTION matrix is committed as upstream_leakage_matrix.csv. It separates derived membership from checkpoint-verified membership; absent checkpoint rosters remain UNKNOWN, including downstream TEST rows.


## Corrective Round 1: artifact provenance evidence

| Artifact pattern | Format and keys | Shape/dtype | Temporal semantics | Provenance source |
| --- | --- | --- | --- | --- |
| audio_id[1,2,4]_fold[0-4].pickle | pickle dict: shared_features, depression_features, pd_features, logits, probs | feature [1,64], logits/probs [2], NumPy float32 | singleton recording token | representative local read; Issue #116 for supervised fold/checkpoint statement |
| video_4_fold[0-4].pickle | pickle dict: shared_features, depression_features, pd_features, logits, probs | feature [1,128], logits/probs [2], NumPy float32 | static recording output | representative local read; legacy artifact pattern |
| text_id[9,10]_fold[0-4].pickle | pickle dict: shared_features, depression_features, pd_features, logits, probs | feature [1,64], logits/probs [2], NumPy float32 | static output; no token order, timestamp, or mask keys | representative local read; legacy artifact pattern |
| WSM ViT temporal pkl | pickle dict: video_temporal, video_cls, meta | [128,768] and [768], torch float32 | temporal frame/token sequence; no explicit timestamp or mask field | representative local read; artifact meta |
| WSM body/full-frame clip pkl | pickle dict: video_temporal, video_cls, meta | representative [532,768] and [768], torch float32 | variable sequence; FPS and frame-count metadata present | representative local read; artifact meta |

The CARE text pipeline consumes an external transcript file and then performs
WhisperX word alignment. The checked code does not establish whether that
transcript was manual or ASR-generated; origin is UNKNOWN. Language is not
stored in the legacy text pickle. DiariZen is used for speaker segmentation,
and longest-speaking-speaker is used as a participant heuristic, but the
legacy pickle does not preserve role labels. The code applies lowercase and
regex token cleaning for linguistic descriptors, participant-turn extraction,
and MPNet all-mpnet-base-v2 turn embeddings with long-turn chunk averaging.
The fold-specific text_id9/text_id10 outputs are supervised classifiers, not
pretrained text embeddings; their classifier FIT and checkpoint-selection
rosters are absent. No transcript content was published or locally inspected
for lexical diagnosis phrases, identifiers, or clinician/interviewer cues;
those shortcut checks are UNKNOWN, not PASS.
