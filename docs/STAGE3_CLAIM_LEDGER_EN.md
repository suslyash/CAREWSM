# Stage-3 Text/Description Claim Ledger

Status: synthesis dossier for TASK-003E; Stage 3 remains ACTIVE pending manager closure. This document is documentation-only and does not authorize training, inference, cache construction, model selection, Stage 7, or Final Test.

## 1. Transcript contract — TASK-003A

Accepted conclusions:

- `SEGMENT TEXT ALIGNMENT NOT ESTABLISHED`
- `T1 TEXT UNIT SHOULD BE VIDEO-LEVEL TRANSCRIPT`
- `T1 TRANSCRIPT DATA CONTRACT READY`

Authoritative report SHA256: `4254d81281522015d0633d6534e1616e925a2bf0130f67a37b541a2aacc2ed78`.

The canonical audit covered 8,622 segments and 755 unique `(corpus, split, video_id)` groups. There were 754 available plain-text transcript files and one missing TRAIN transcript, with no decode errors. There were no cross-split exact transcript-hash duplicates and three within-TRAIN exact duplicate hashes. Source metadata contained no explicit timing columns, and transcript timestamp parsing found no parseable timestamps. The corrected authoritative report disabled TEST lexical/language analysis; TEST was structural-only.

There is no scientifically established segment-level transcript alignment. T1 therefore uses video-level transcript units; segment broadcasting is an evaluation representation and does not create invented segment text or alignment.

## 2. T1 frozen family identity — TASK-003B

- Encoder: `FacebookAI/xlm-roberta-base`.
- Requested revision: `e73636d`.
- Resolved commit: `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
- Model weight SHA256: `6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb`.
- Cache index SHA256: `4f276b60e8e423a1eab2d69a02782daa274fb183ae028a525f16df3adcf83ea8`.
- TRAIN units: D287/P266/total553 unique video-level transcripts.
- Trainable downstream parameters: `447746`.

T1 is one frozen pretrained transcript family; no encoder search occurred.

## 3. T1 standalone evidence — TASK-003C

Exact completion: `T1 THREE-SEED TEXT BASELINE COMPLETE`.

Primary DEV D/P/Mean by seed:

| Seed | Depression | Parkinson | Mean |
|---:|---:|---:|---:|
| 42 | 0.721153 | 0.899381 | 0.810267 |
| 43 | 0.562911 | 0.825768 | 0.694339 |
| 44 | 0.596355 | 0.809025 | 0.702690 |

Three-seed mean: D `0.6268063333`, P `0.8447246667`, Mean `0.7357653333`.

Sample standard deviation: D `0.0834002123`, P `0.0480683689`, Mean `0.0646553057`.

Range: D `0.158242`, P `0.090356`, Mean `0.115928`.

Contextual differences:

- Versus matched temporal audio: D `-0.1035314632`, P `+0.0284285455`, Mean `-0.0375516255`.
- Versus full R4: D `-0.1084500000`, P `+0.0061640000`, Mean `-0.0511434846`.

T1 is a standalone text-stream baseline only. It does not show that text improves A+V. Parkinson performance is strong relative to the contextual audio/R4 references, while depression is materially weaker and seed-sensitive. Unique-video DEV diagnostics were secondary and lower than primary segment-weighted metrics on all seeds. Test monitoring did not drive selection. No three-seed significance claim is authorized.

## 4. T2 observable-description preflight — TASK-003D

Exact result: `T2 DESCRIPTION GENERATION CONTRACT BLOCKED`.

Frozen generator: `Qwen/Qwen3-VL-8B-Instruct`, revision `1dd1e02d981403da25ed73d43430e4ef598cb94b`.

- Prompt SHA256: `118b93ba090af2b4ce755d9da1476e073cab0624760e9aa4c347faad8750ee4d`.
- Deterministic video setting actually tested after the documented CUDA-memory correction: uniform `num_frames=8`.
- External report SHA256: `767d687e2f857938fa882555250d7a4be4159556e49537377967d82c36407930`.

Manual audit over 24 deterministic TRAIN/DEV samples:

| Audit field | Result |
|---|---:|
| Nonempty | 24/24 |
| Visually grounded/mostly observable | 24/24 |
| Major unsupported/hallucinated fact | 0/24 |
| Diagnosis/disease/health-state inference | 0/24 |
| Demographic/identity inference | 4/24 |
| Causal/medication inference | 0/24 |
| Dataset/task/label leakage | 0/24 |
| Concise enough for semantic encoding | 18/24 |

The failed frozen conditions were zero demographic/identity inference and at least 22/24 concise outputs. T2 stopped before description cache construction, T2 model implementation, training, DEV performance, or Test generation. No T2 performance claim exists. No prompt/model remediation is authorized inside this bounded Stage-3 family study.

## 5. Stage-3 completeness matrix

| Requirement | Accepted evidence | Status |
|---|---|---|
| Transcript coverage, encoding, language, and leakage audit | TASK-003A | Complete |
| No invented segment alignment | TASK-003A transcript contract | Complete |
| One pretrained transcript family implemented | TASK-003B | Complete |
| Three-seed standalone T1 evidence | TASK-003C | Complete |
| Second family attempted as transcript plus observable description | TASK-003D | Complete as bounded preflight |
| Description model/prompt revision fingerprinted | TASK-003D | Complete |
| Diagnosis-free prompt and manual audit | TASK-003D | Complete; gate failed on safety/concision counts |
| No broad text/description search | TASK-003A–D scope controls | Complete |
| No Test-driven text/description selection | TASK-003A–D scope controls | Complete |
| Description sanity check performed and stopped at frozen gate | TASK-003D | Complete |

`STAGE-3 TEXT/DESCRIPTION EVIDENCE MATRIX COMPLETE`

## 6. Family-budget conclusion

`STAGE-3 TWO-FAMILY BUDGET EXHAUSTED`

T1 was the sole implemented pretrained transcript family. T2 was the sole second-family transcript-plus-observable-description attempt and was correctly stopped at its blocked preflight. No third encoder, generator, prompt variant, or prompt-remediation family was attempted. A blocked T2 preflight is a valid bounded-study outcome and does not require forcing T2 training.

## 7. Descriptive pre-Stage-7 candidate inventory

This inventory is descriptive only. It is not a ranking, tiering, score, winner declaration, promotion, demotion, or Stage-7 selection.

### Frozen temporal audio

DEV D/P/Mean: `0.7303377965/0.8162961212/0.7733169588`; Mean sample std `0.0138512531`. This is the more seed-stable contextual reference by Mean variation.

### Full R4 trial012

DEV D/P/Mean: `0.7352563333/0.8385606667/0.7869088179`; Mean sample std `0.0294384420`. It exceeds matched temporal audio Mean on 3/3 seeds. Multiple Stage-6 component claims remain unsupported, so complexity necessity is not established.

### Equal-parameter shared-fusion comparator

DEV D/P/Mean: `0.7407873333/0.8467100000/0.7935663333`; full-minus-shared Mean `-0.0066575154`. Both models have `295239` trainable parameters. This is a relevant simplified single-model reference for a later manager decision, not a selection in TASK-003E.

### No-semantic-depression comparator

DEV D/P/Mean: `0.714447/0.868369/0.791408`; full-minus-ablation D/P/Mean `+0.020809/-0.029808/-0.004499`. It has a higher aggregate Mean than full R4 in the frozen comparison but a material D/P trade-off; it is contextual evidence only.

### Paired task-isolated comparator

DEV D/P/paired Mean: `0.717846/0.881495/0.799671`. It uses two separately trained models and is not an equal-total-deployment-size comparator. It is a diagnostic/upper-control reference, not a like-for-like single-model finalist.

### T1 text-only

DEV D/P/Mean: `0.6268063333/0.8447246667/0.7357653333`. This is a standalone contextual text baseline and is not evidence that text should be added to A+V.

### T2

T2 is blocked before implementation/training and is ineligible for Stage-7 performance comparison under the current evidence.

Stage-6 unsupported component claims remain unsupported. The candidate facts above do not select a final method. No significance claim is authorized from this three-seed pre-final evidence.

## 8. Stage boundary

Stage 3 remains ACTIVE until manager accepts TASK-003E. Stage 7 remains LOCKED and Final Test remains unauthorized. No further Stage-3 family is justified under the frozen two-family programme budget. After manager acceptance, the next programme-level action should be a separate Stage-7 candidate/config/seed-freeze task, not immediate training. This dossier does not activate Stage 7 or edit any stage plan.
