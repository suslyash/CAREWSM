# MantisV2 eGeMAPS Temporal Contract Acceptance

Date: 2026-11-06  
Required main: `abf550385c2f33cab6b3d580bd722d44b2f7cad7`  
Branch: `codex/mantisv2-egemaps-temporal-validator`

## 1. Executive finding

Recommendation: **GO for the timestamp/window contract, CONDITIONAL for MantisV2 integration**.

The existing canonical CARE/WSM TRAIN and fold-VAL eGeMAPS LLD artifacts provide a consistent, recording-relative timestamp axis suitable for deterministic within-turn 1.0-second windows with a 0.5-second hop. The validator found no invalid timestamp rows, nonfinite feature rows, duplicate timestamps, out-of-order rows, or temporal gaps in the inspected artifacts. All observed frame transitions were 20 ms frame duration with a 10 ms start-time hop, producing the expected 50% frame overlap.

This acceptance is limited to timestamp semantics and window coverage. It does not authorize Mantis installation, checkpoint download, inference, resampling to Mantis input length, cache generation, or training. MantisV2 integration remains conditional on a later isolated implementation gate.

## 2. Exact source schema and timestamp semantics

The authoritative CARE registry defines exactly 25 eGeMAPS LLD channels, in this order:

`Loudness_sma3`, `F1frequency_sma3nz`, `F1bandwidth_sma3nz`, `F2frequency_sma3nz`, `F2bandwidth_sma3nz`, `F3frequency_sma3nz`, `F3bandwidth_sma3nz`, `jitterLocal_sma3nz`, `shimmerLocaldB_sma3nz`, `F0semitoneFrom27.5Hz_sma3nz`, `logRelF0-H1-H2_sma3nz`, `logRelF0-H1-A3_sma3nz`, `F1amplitudeLogRelF0_sma3nz`, `F2amplitudeLogRelF0_sma3nz`, `F3amplitudeLogRelF0_sma3nz`, `alphaRatio_sma3`, `hammarbergIndex_sma3`, `slope0-500_sma3`, `slope500-1500_sma3`, `spectralFlux_sma3`, `mfcc1_sma3`, `mfcc2_sma3`, `mfcc3_sma3`, `mfcc4_sma3`, `HNRdBACF_sma3nz`.

Every accepted artifact was required to contain these metadata columns:

- `turn_id`
- `turn_start_sec`
- `turn_end_sec`
- `frame_start_sec`
- `frame_end_sec`

Across 8,779 unique canonical TRAIN/VAL artifact paths, the first accepted row was inside its turn bounds. 1,713 CARE artifacts had positive turn starts and 7,066 artifacts had zero starts; this is consistent with recording-relative coordinates, not a turn-relative reset rule. The first frame begins at the turn start in the representative artifacts, while later turns retain their nonzero recording position.

A first-three-row transition audit across 8,778 artifacts produced 17,556 transitions, all with a 0.020-second frame interval and a 0.010-second frame-start hop. Thus the data must not be described as non-overlapping 20-ms frames: consecutive 20-ms observations overlap by 10 ms.

## 3. Deterministic temporalization contract

The validator implements the following descriptive contract:

1. Candidate windows are generated independently inside each turn.
2. A candidate begins at `turn_start_sec + n * 0.5` and is retained only when its full one-second interval ends no later than `turn_end_sec`.
3. Windows use the half-open endpoint rule `[start_sec, end_sec)`.
4. A frame contributes its actual timestamped interval when it intersects the window; no timestamp or feature value is fabricated.
5. Valid-observation fraction is the union duration of observed frame intervals inside the one-second window divided by 1.0 second, capped at 1.0.
6. Windows shorter than one second are rejected. Windows do not cross turn boundaries, bridge gaps, or extrapolate observations.
7. The 80% and 95% thresholds are predeclared descriptive quality gates. They are not model-selection hyperparameters.
8. The validator does not resample real observations to 64 or 512 points and does not touch turn-level deep embeddings.

The synthetic checks cover short turns, missing timestamp values, duplicate timestamps, frame gaps, exact turn-boundary windows, and the proposed cache fields. They pass.

## 4. TRAIN/VAL coverage

Only canonical fold TRAIN and VAL files for folds 0–4 were read. TEST files and TEST artifacts were not read.

`Source` is the canonical fold population. `Artifact` is the manifest-valid eGeMAPS artifact count. `Usable` is the number of samples with at least one window meeting the threshold. Window counts are candidate / accepted.

| Corpus | Fold | Split | Source | Artifact | Usable 80% | Usable 95% | Windows candidate / accepted-80 / accepted-95 |
|---|---:|---|---:|---:|---:|---:|---:|
| CARE | 0 | TRAIN | 1361 | 1361 | 1361 (100.00%) | 1361 (100.00%) | 298440 / 298436 / 298419 |
| CARE | 0 | VAL | 352 | 352 | 352 (100.00%) | 352 (100.00%) | 74631 / 74630 / 74625 |
| CARE | 1 | TRAIN | 1368 | 1368 | 1368 (100.00%) | 1368 (100.00%) | 292532 / 292527 / 292506 |
| CARE | 1 | VAL | 345 | 345 | 345 (100.00%) | 345 (100.00%) | 80539 / 80539 / 80538 |
| CARE | 2 | TRAIN | 1378 | 1378 | 1378 (100.00%) | 1378 (100.00%) | 300814 / 300809 / 300791 |
| CARE | 2 | VAL | 335 | 335 | 335 (100.00%) | 335 (100.00%) | 72257 / 72257 / 72253 |
| CARE | 3 | TRAIN | 1382 | 1382 | 1382 (100.00%) | 1382 (100.00%) | 300605 / 300601 / 300584 |
| CARE | 3 | VAL | 331 | 331 | 331 (100.00%) | 331 (100.00%) | 72466 / 72465 / 72460 |
| CARE | 4 | TRAIN | 1363 | 1363 | 1363 (100.00%) | 1363 (100.00%) | 299893 / 299891 / 299876 |
| CARE | 4 | VAL | 350 | 350 | 350 (100.00%) | 350 (100.00%) | 73178 / 73175 / 73168 |
| WSM | 0 | TRAIN | 5640 | 5637 | 5627 (99.77%) | 5627 (99.77%) | 652246 / 652246 / 652246 |
| WSM | 0 | VAL | 1429 | 1428 | 1427 (99.86%) | 1427 (99.86%) | 165692 / 165692 / 165692 |
| WSM | 1 | TRAIN | 5661 | 5658 | 5649 (99.79%) | 5649 (99.79%) | 654500 / 654500 / 654500 |
| WSM | 1 | VAL | 1408 | 1407 | 1405 (99.79%) | 1405 (99.79%) | 163438 / 163438 / 163438 |
| WSM | 2 | TRAIN | 5670 | 5666 | 5657 (99.77%) | 5657 (99.77%) | 656033 / 656033 / 656033 |
| WSM | 2 | VAL | 1399 | 1399 | 1397 (99.86%) | 1397 (99.86%) | 161905 / 161905 / 161905 |
| WSM | 3 | TRAIN | 5640 | 5637 | 5628 (99.79%) | 5628 (99.79%) | 652950 / 652950 / 652950 |
| WSM | 3 | VAL | 1429 | 1428 | 1426 (99.79%) | 1426 (99.79%) | 164988 / 164988 / 164988 |
| WSM | 4 | TRAIN | 5665 | 5662 | 5655 (99.82%) | 5655 (99.82%) | 656023 / 656023 / 656023 |
| WSM | 4 | VAL | 1404 | 1403 | 1399 (99.64%) | 1399 (99.64%) | 161915 / 161915 / 161915 |

CARE's few rejected windows occur at the 80%/95% quality gates, not from
invalid timestamp rows. WSM's missing artifacts are reflected in the Artifact
column; all candidate WSM windows met both thresholds. The WSM samples without
usable windows are the missing-artifact cases, not silently repaired records.

## 5. Integrity diagnostics and fold isolation

Across all canonical TRAIN/VAL populations:

- Invalid timestamp rows: 0.
- Nonfinite feature rows: 0.
- Duplicate timestamp rows: 0.
- Out-of-order timestamp rows: 0.
- Out-of-turn frame rows: 0.
- Detected timestamp gaps: 0.
- Detected interval overlaps: expected and pervasive because the measured hop is 10 ms while each frame interval is 20 ms; these are overlapping observations, not missing-data gaps.
- TRAIN/VAL profile overlap: 0 for CARE and WSM in every fold.

The validator reads each unique artifact at most once during a run, but reports
coverage separately for every canonical fold population. It writes no cache,
manifest, split, feature, or model output.

## 6. Scientific limitations

The acceptance does not establish that eGeMAPS LLD is the best Mantis input or
that MantisV2 will improve disease recognition. It establishes only that the
existing frame timestamps support a reproducible within-turn window contract.

The timestamped LLD values are already extracted acoustic descriptors, not raw
waveform samples. Their 10-ms hop and 20-ms overlapping frame intervals must
remain part of the provenance. A later Mantis input-length conversion to a
multiple of 32 or 512 points is a separate preprocessing decision and has not
been validated here.

The 1.0-second windows are anchored to each turn start. This is deterministic,
but it is not evidence that turn starts are physiologically meaningful event
boundaries. Windows crossing turn gaps were intentionally excluded. A later
contract may compare a documented alternative only under a separate task.

The report contains aggregate statistics only. It does not contain sample IDs,
patient metadata, individual feature values, raw audio, or predictions.

## 7. Final recommendation

**GO** for the eGeMAPS timestamp/window contract and **CONDITIONAL** for the
next MantisV2 integration step.

The next step may be a separate manager-authorized isolated setup task to
validate `mantis-tsfm==1.1.0` and the pinned `paris-noah/MantisV2` revision in a
separate environment, followed by a synthetic tensor smoke test. It must still
not download weights or run real inference until that setup is reviewed.

## 8. Reproduction

Run from the repository root:

```bash
python3 scripts/validate_mantis_egemaps_temporalization.py --check
```

The command reads the authoritative registry, the existing availability
manifest, and only `fold_{0..4}_{train,val}.csv` plus their eGeMAPS LLD artifacts.
It runs the synthetic checks first and prints aggregate JSON. It does not read
TEST and does not modify source data.
