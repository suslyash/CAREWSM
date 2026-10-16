# Stage-3 Transcript Source Audit

Status: TASK-003A complete; audit-only. External report: `/media/maxim/Programs/Features/WSM/stage3_text_audit/transcript_source_audit_v1.json`.

Report SHA256:

`4254d81281522015d0633d6534e1616e925a2bf0130f67a37b541a2aacc2ed78`

## Scope and firewall

The audit used the canonical manifest implementation and derived only:

`data_root / corpus / f"{split}_labels" / video_id / f"{video_id}.txt"`

It did not modify the dataset, train a model, download a model, create a config, inspect Test predictions or metrics, inspect Test lexical content, or select an encoder. The report contains no transcript text or snippets. Test lexical inspection is explicitly false; Test was audited structurally only.

The pre-execution firewall passed on pushed commit `35a50c8`: help, synthetic plain/SRT/WebVTT/timestamped/JSON/invalid-UTF8/empty parser checks, overwrite refusal, raw-text schema guard, syntax, diff check, and empty source/config/dependency diff. The first full invocation exposed a script field-name defect before report creation; the narrowly corrected script was pushed as `f54649f`, after which the single successful audit completed. No dataset or conclusion was changed.

The first report with SHA256 `2dc2064aca131da0e80c103df1b8dd7a19d9735aa8c183e149392f7aafd1206b` was rejected during manager review because TEST records carried language-analysis fields. It is superseded. Corrective firewall commit `b9a2d85` pushed the fix before the one authorized replacement report. TEST records now contain no `coarse_language` or `mixed_language` outputs; `test_language_audit_performed = false`, `test_lexical_inspection_performed = false`, and `test_predictions_or_metrics_inspected = false`.

## Canonical coverage

The canonical manifest contained 8,622 segment rows and 755 unique `(corpus, split, video_id)` groups. Coverage is non-empty transcript files over canonical unique videos and segments:

| Corpus/split | Videos | Videos with transcript | Video coverage | Segments | Segments with transcript | Segment coverage |
|---|---:|---:|---:|---:|---:|---:|
| depression/train | 288 | 287 | 99.653% | 3,660 | 3,587 | 98.005% |
| depression/dev | 56 | 56 | 100.000% | 621 | 621 | 100.000% |
| depression/test | 55 | 55 | 100.000% | 827 | 827 | 100.000% |
| parkinson/train | 266 | 266 | 100.000% | 2,665 | 2,665 | 100.000% |
| parkinson/dev | 44 | 44 | 100.000% | 312 | 312 | 100.000% |
| parkinson/test | 46 | 46 | 100.000% | 537 | 537 | 100.000% |

Missing transcripts are counted as availability gaps; no substitute path or inferred text was used.

## File, format, and encoding audit

The 755 unique transcript paths classified as 754 `plain_text` files and 1 `missing` path. No decode-error file occurred. For each present file the report records existence, non-empty status, byte size, SHA256, UTF-8/UTF-8-SIG decoding, line/nonblank-line counts, Unicode character counts, whitespace-token counts, and deterministic format class. No WebVTT, SRT-like, timestamped-line, or JSON-like transcript format was found.

## Source metadata and alignment

All six source CSVs had `video_id`, `diagnosis`, and `segment_file`; the two Test CSVs additionally had `soft_filter`, `hard_filter`, and `diagnosis_2`. No source CSV contained explicit start, end, offset, timestamp, or equivalent segment timing columns. Transcript timestamp parsing found zero timestamp spans/lines.

The canonical segment identity was parsed exactly as `[corpus, video_id, segment_file]`. No segment ordinal, filename suffix, fixed duration, or video-duration arithmetic was used.

`SEGMENT TEXT ALIGNMENT NOT ESTABLISHED`

`T1 TEXT UNIT SHOULD BE VIDEO-LEVEL TRANSCRIPT`

The existing files are auditable video-level sources, but there is no explicit metadata mapping every canonical segment to a transcript span.

## Duplicate-content leakage audit

Duplicate analysis used transcript SHA256 only and lists no text. There were no cross-split exact duplicate hashes: TRAIN/DEV 0, TRAIN/TEST 0, and DEV/TEST 0. TRAIN contained three duplicate hashes, each represented by the same video ID in both corpus namespaces: `VJc6FUP7LTY`, `lpUYzrEUdQs`, and `yxNbbFOO7Jc`. These are within-split exact-duplicate risks and must remain visible to future data handling; no label or performance inference was made.

## TRAIN/DEV language audit

The Unicode-script audit found only Latin letters and digits in decoded TRAIN/DEV transcripts. TRAIN: Latin 3,214,969 (99.790949%), digits 6,735 (0.209051%), total classified characters 3,221,704. DEV: Latin 488,444 (99.807718%), digits 941 (0.192282%), total 489,385. Cyrillic, Greek, Arabic, Han, Hiragana/Katakana, Hangul, and other-letter counts were zero.

The deterministic manual-audit sample used stable SHA256 ordering of `corpus|split|video_id`, up to six videos per TRAIN/DEV corpus stratum, for 24 IDs total. All sampled files were readable and non-empty, had coarse label `English_or_other_Latin`, and had mixed-language `no`. IDs are recorded in the machine-readable report only; no phrases or snippets are committed.

## Readiness and T1 boundary

`T1 TRANSCRIPT DATA CONTRACT READY`

The path derivation is deterministic, existing files are structurally auditable without uncontrolled decode ambiguity, TRAIN/DEV language evidence is sufficient for an audited-language input contract, and duplicate risks are explicitly enumerated. This readiness result does not authorize T1 implementation or training.

Recommended granularity is video-level transcript. A future implementation may define an explicit policy for broadcasting a video-level representation to segment rows, but TASK-003A does not implement that policy and does not silently multiply transcript instances by segment count.

No encoder, model family, dependency, or Test-based language choice was selected or downloaded.
