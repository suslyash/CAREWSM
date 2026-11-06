# MantisV2 eGeMAPS Temporal Contract Acceptance

Date: 2026-11-06
Required main: `abf550385c2f33cab6b3d580bd722d44b2f7cad7`
Branch: `codex/mantisv2-egemaps-temporal-validator`
Validator correction base: `31538be6ed0d708e75226a579bee184f060b3a90`

## 1. Executive finding

**Recommendation: NO-GO for immediate MantisV2 temporalization; CONDITIONAL for a bounded data-contract repair.**

The corrected validator confirms a recording-relative eGeMAPS time axis, a 20 ms frame duration, and a 10 ms frame-start hop in both corpora. It also detects a real CARE integrity failure: frame intervals extend beyond their declared turn end in aggregate TRAIN/VAL population scans. The validator therefore exits nonzero and excludes those artifacts from fully-valid window counts. This report does not treat interval-union coverage as completeness: expected-frame density is reported separately. Mantis installation, checkpoint download, inference, resampling, cache generation, and training remain out of scope.

The prior report called the temporal contract GO because it checked interval bounds only through a weaker audit and did not fail on these boundary rows. The corrected run found no missing expected 10 ms starts, so the change is caused by enforcing frame-to-turn membership, not by a newly observed cadence gap.

## 2. Exact source schema and timestamp semantics

The authoritative CARE registry contains exactly these 25 eGeMAPS LLD channels. The registry set matches the artifact schema exactly; the validator checks the observed CSV serialization order explicitly:

`Loudness_sma3`, `alphaRatio_sma3`, `hammarbergIndex_sma3`, `slope0-500_sma3`, `slope500-1500_sma3`, `spectralFlux_sma3`, `mfcc1_sma3`, `mfcc2_sma3`, `mfcc3_sma3`, `mfcc4_sma3`, `F0semitoneFrom27.5Hz_sma3nz`, `jitterLocal_sma3nz`, `shimmerLocaldB_sma3nz`, `HNRdBACF_sma3nz`, `logRelF0-H1-H2_sma3nz`, `logRelF0-H1-A3_sma3nz`, `F1frequency_sma3nz`, `F1bandwidth_sma3nz`, `F1amplitudeLogRelF0_sma3nz`, `F2frequency_sma3nz`, `F2bandwidth_sma3nz`, `F2amplitudeLogRelF0_sma3nz`, `F3frequency_sma3nz`, `F3bandwidth_sma3nz`, `F3amplitudeLogRelF0_sma3nz`.

Required metadata columns are `turn_id`, `turn_start_sec`, `turn_end_sec`, `frame_start_sec`, and `frame_end_sec`. The validator requires these columns followed by the 25 channels in the observed artifact order.

The measured nominal frame-start hop is derived from the first three rows of each manifest-valid TRAIN/VAL artifact and requires at least 99% modal agreement. CARE: `0.010000` seconds in 3,426 transition observations; WSM: `0.010000` seconds in 14,130 transition observations. Across all inspected valid rows, frame duration is `0.020000` seconds: CARE 95,590,725 observations and WSM 206,967,265 observations. These are overlapping 20 ms observations at a 10 ms start cadence.

Positive recording-relative turn starts and retained nonzero positions in later turns show that timestamps are not reset to zero per turn. The validator rejects ambiguous or nonfinite coordinates rather than correcting them.

## 3. Deterministic temporalization and fail-closed rules

1. Candidate windows are generated independently within each turn at `turn_start_sec + n * 0.5` and require a complete one-second interval within the declared turn.
2. Endpoints are half-open: `[start_sec, end_sec)`. Frame intervals contribute to interval-union observation coverage only when they intersect the window.
3. Expected-frame density is separate: with the measured 10 ms hop, a one-second window expects 100 frame starts; density is observed in-window starts divided by 100.
4. A missing 10 ms frame is invalid even when overlapping frame intervals still produce continuous union coverage. Duplicate starts, nonfinite timestamps, out-of-order rows, irregular cadence, inconsistent turn metadata, and frames outside turn boundaries are invalid.
5. Windows shorter than one second, windows crossing turns, unexplained gaps, extrapolation, interpolation, and fabricated observations are rejected. The 80% and 95% thresholds are descriptive quality thresholds, not model-selection choices.
6. Readable files, fully valid artifacts, manifest-invalid coverage exclusions, interval coverage, and expected-frame density are reported separately.
7. The synthetic suite uses temporary CSV fixtures and the real inspection logic. It covers correct 10 ms cadence, one missing frame, duplicate timestamps, nonfinite timestamps, out-of-order rows, inconsistent turn bounds, a short turn, boundary-aligned windows, wrong channel order, and a valid cache field structure. It passes, including nonzero failure exit checks.

## 4. Corrected TRAIN/VAL coverage

Only canonical `fold_{0..4}_{train,val}.csv` files and their eGeMAPS LLD artifacts were read. TEST files and TEST artifacts were not read. `Source` is the fold population; `Manifest` is manifest-valid; `Readable` is an actually read manifest-valid file; `Full` is fully valid under the corrected integrity rules. Window counts are candidate / accepted at 80% / accepted at 95%. Density is mean / minimum.

| Corpus | Fold | Split | Source | Manifest | Readable | Full | Candidate / 80% / 95% | Density mean / min |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CARE | 0 | TRAIN | 1361 | 1361 | 1361 | 1340 | 293509 / 293505 / 293488 | 0.999949 / 0.60 |
| CARE | 0 | VAL | 352 | 352 | 352 | 342 | 72632 / 72631 / 72627 | 0.999941 / 0.73 |
| CARE | 1 | TRAIN | 1368 | 1368 | 1368 | 1342 | 286881 / 286876 / 286856 | 0.999943 / 0.60 |
| CARE | 1 | VAL | 345 | 345 | 345 | 340 | 79260 / 79260 / 79259 | 0.999965 / 0.76 |
| CARE | 2 | TRAIN | 1378 | 1378 | 1378 | 1350 | 294558 / 294553 / 294536 | 0.999946 / 0.60 |
| CARE | 2 | VAL | 335 | 335 | 335 | 332 | 71583 / 71583 / 71579 | 0.999954 / 0.77 |
| CARE | 3 | TRAIN | 1382 | 1382 | 1382 | 1355 | 294542 / 294538 / 294522 | 0.999947 / 0.67 |
| CARE | 3 | VAL | 331 | 331 | 331 | 327 | 71599 / 71598 / 71593 | 0.999948 / 0.60 |
| CARE | 4 | TRAIN | 1363 | 1363 | 1363 | 1341 | 295074 / 295072 / 295058 | 0.999952 / 0.60 |
| CARE | 4 | VAL | 350 | 350 | 350 | 341 | 71067 / 71064 / 71057 | 0.999927 / 0.67 |
| WSM | 0 | TRAIN | 5640 | 5637 | 5637 | 5637 | 652246 / 652246 / 652246 | 0.999985 / 0.96 |
| WSM | 0 | VAL | 1429 | 1428 | 1428 | 1428 | 165692 / 165692 / 165692 | 0.999985 / 0.96 |
| WSM | 1 | TRAIN | 5661 | 5658 | 5658 | 5658 | 654500 / 654500 / 654500 | 0.999985 / 0.96 |
| WSM | 1 | VAL | 1408 | 1407 | 1407 | 1407 | 163438 / 163438 / 163438 | 0.999984 / 0.96 |
| WSM | 2 | TRAIN | 5670 | 5666 | 5666 | 5666 | 656033 / 656033 / 656033 | 0.999985 / 0.96 |
| WSM | 2 | VAL | 1399 | 1399 | 1399 | 1399 | 161905 / 161905 / 161905 | 0.999983 / 0.96 |
| WSM | 3 | TRAIN | 5640 | 5637 | 5637 | 5637 | 652950 / 652950 / 652950 | 0.999984 / 0.96 |
| WSM | 3 | VAL | 1429 | 1428 | 1428 | 1428 | 164988 / 164988 / 164988 | 0.999987 / 0.96 |
| WSM | 4 | TRAIN | 5665 | 5662 | 5662 | 5662 | 656023 / 656023 / 656023 | 0.999985 / 0.96 |
| WSM | 4 | VAL | 1404 | 1403 | 1403 | 1403 | 161915 / 161915 / 161915 | 0.999985 / 0.96 |

The 80%/95% accepted-window counts above are calculated only from fully valid artifacts. CARE has 155 invalid artifact occurrences across the 10 fold populations and 165 out-of-turn frame rows in those population scans; the same artifact can occur in more than one fold population, so these are population-level aggregate counts, not person-level counts. WSM has no out-of-turn rows. Manifest-invalid WSM artifacts remain visible as exclusions rather than replacements.

Interval-union coverage remains high for fully valid artifacts, but it is not a completeness guarantee. The separate density minima are 0.60–0.77 in CARE and 0.96 in WSM; these low minima occur in individual candidate windows that remain visible through the 80%/95% acceptance counts. No missing expected frame starts or irregular cadence events were detected in either corpus.

## 5. Integrity diagnostics and fold isolation

| Check | CARE | WSM |
|---|---:|---:|
| Invalid/nonfinite timestamp rows | 0 | 0 |
| Nonfinite feature rows | 0 | 0 |
| Duplicate starts | 0 | 0 |
| Out-of-order rows | 0 | 0 |
| Inconsistent turn boundaries | 0 | 0 |
| Frames outside declared turns | 165 rows in population scans | 0 |
| Missing expected 10 ms starts | 0 | 0 |
| Irregular cadence events | 0 | 0 |
| Interval gaps | 0 | 0 |
| Interval overlaps | present, expected at 10/20 ms | present, expected at 10/20 ms |
| TRAIN/VAL profile overlap | 0 in every fold | 0 in every fold |

The CARE boundary failure is exemplified by frame intervals whose end is slightly later than the declared turn end. The validator uses a documented `1e-6` second numerical tolerance; the observed violations exceed that tolerance and therefore remain failures. No individual sample identifiers are included in this report.

## 6. Scientific limitations

This is a timestamp and artifact-integrity audit, not evidence that eGeMAPS is an effective Mantis input or that MantisV2 improves disease recognition. The current NO-GO is specifically about using the present artifacts without resolving the CARE turn-boundary contract.

A later repair must establish whether the turn-end metadata or the final frame interval is authoritative, document that decision, and rerun this fail-closed audit. It must not silently clip frames, change timestamps, fabricate observations, or alter canonical splits. The proposed 1.0 s / 0.5 s windows remain unvalidated for the affected artifacts until that decision is approved.

No real data were resampled to 64 or 512 points. No Mantis package or checkpoint was installed or downloaded. No model inference, cache creation, or training was performed.

## 7. Final recommendation

**NO-GO** for immediate MantisV2 temporalization on the current unqualified CARE/WSM artifact set. **CONDITIONAL** for the next step only after a manager-authorized correction of the CARE frame/turn boundary contract and a clean rerun with zero integrity failures. The cadence result itself is positive, but it does not override the boundary blocker.

## 8. Reproduction

From the repository root:

```bash
python3 scripts/validate_mantis_egemaps_temporalization.py --check
```

The command runs synthetic checks first, reads the authoritative registry and existing availability manifest, then reads only canonical TRAIN/VAL folds 0–4 and their eGeMAPS LLD artifacts. It prints aggregate JSON and returns nonzero when any required integrity condition fails. It does not read TEST and does not modify source data.
