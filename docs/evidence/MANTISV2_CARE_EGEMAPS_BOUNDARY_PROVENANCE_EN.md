# CARE eGeMAPS Frame/Turn Boundary Provenance Investigation

Date: 2026-11-06
Required main: `3c48ae3a8df3a2010f7df4b218cf27755dc0ce56`
Branch: `codex/mantisv2-care-egemaps-boundary-provenance`

## 1. Executive finding

**Recommendation: NO-GO for changing the current strict boundary rule.**

The CARE anomaly is highly predictable and concentrated at turn ends, but the available evidence does not establish whether the terminal openSMILE interval is a valid tail-frame convention or an interval label that should be excluded. All 33 affected turns have the same one-sample excess pattern, and all affected frame starts and midpoints remain inside their declared turns. However, the exact openSMILE package/version and original WAV headers are not available in the current environment, so a scientifically defensible policy cannot yet be authorized.

The evidence supports a probable terminal-frame convention, not a completed causal proof. The existing fail-closed validator was not modified.

## 2. Extraction and turn provenance

The inspected extraction implementation uses the following sequence:

- `librosa.load(..., sr=16000, mono=True)` loads a mono waveform at 16 kHz.
- `get_turn_times()` uses the whole recording unless `--use-diarization` is supplied. When enabled and an RTTM exists, it calls `extract_participant_turns()`.
- RTTM participant segments are merged across gaps up to 3 seconds, interviewer segments create hard boundaries, and turns shorter than 0.3 seconds are removed.
- Each float turn endpoint is converted with `int(start_sec * 16000)` and `int(end_sec * 16000)` and the waveform is sliced with the half-open interval `[start_sample:end_sample]`.
- openSMILE eGeMAPSv02 LLD output is generated on each sliced turn. The code writes openSMILE index `start` and `end` timestamps plus `start_sec`; it does not clip, round, or validate the resulting frame endpoints against the declared turn.

The affected artifacts contain multiple participant turns and retain RTTM files, which is consistent with extraction using `--use-diarization`. All 33 affected turn records had a corresponding participant RTTM endpoint matching the declared turn end under the available metadata comparison. Thus the declared endpoint is a diarization-derived boundary, not an independently verified openSMILE effective analysis endpoint.

The repository does not pin an openSMILE version, the current environment cannot import `opensmile`, and no archived extraction-environment metadata was found. Original WAV files or WAV headers are not present under the inspected CARE artifact tree. No audio was decoded.

## 3. Aggregate anomaly quantification

The audit read only canonical CARE TRAIN/VAL fold populations for folds 0–4 and only manifest-valid eGeMAPS artifacts. It found:

| Quantity | Aggregate |
|---|---:|
| Unique CARE artifacts | 1,713 |
| Total fold-population artifact occurrences | 8,565 |
| Unique affected artifacts | 31 |
| Unique affected turns | 33 |
| Repeated affected artifact occurrences across folds | 155 |
| Repeated affected turn occurrences across folds | 165 |
| Unaffected artifacts | 1,682 |
| Total turns | 6,884 |
| Unaffected turns | 6,851 |

Excess is `frame_end_sec - turn_end_sec` for the affected terminal frame:

| Statistic | Excess (ms) |
|---|---:|
| Minimum | 0.062499 |
| Median | 0.062500 |
| P90 | 0.062500 |
| P99 | 0.062500 |
| Maximum | 0.062500 |

The excess distribution is 32 rows at 0.062500 ms and one row at 0.062499 ms. This is one sample at 16 kHz, within floating-point representation noise.

Every affected anomaly was on the final frame of its turn. All affected frame starts remained inside the declared turn, and all affected frame midpoints remained inside the declared turn. The affected final frame began 50.000–59.000 ms before the declared endpoint, with midpoint 24.969–29.469 ms before it.

Declared turn ends were not aligned to the measured 10 ms or 20 ms frame grids: 13 affected endpoints were 2 ms from the nearest 10 ms boundary and 20 were 3 ms from it. Their distances from the nearest 20 ms boundary were 8 ms and 7 ms respectively. This argues against a simple “turn end rounded to a frame grid” explanation.

As an unaffected reference, terminal-frame slack (`turn_end_sec - final frame_end_sec`) across all turns had median 0 ms, P90 0.0625 ms, P99 98.6875 ms and maximum 839.6875 ms. The small negative minimum was within floating-point tolerance. Unaffected terminal frames are therefore not uniformly one fixed duration either.

## 4. Frame-edge behavior and supported cause

The complete frame-duration audit is important because adjacent-frame cadence alone does not characterize the terminal row:

| Frame class | Count | Duration summary |
|---|---:|---|
| Interior frames | 19,118,145 | Approximately 20 ms throughout |
| Terminal frames | 6,884 | 50 ms median; P90 59 ms; maximum 59.9375 ms |
| All frames | 19,125,029 | Maximum 59.9375 ms |

The 33 affected frames are all terminal frames, and their excess is exactly one 16 kHz sample. Together with the extraction code’s direct use of openSMILE `start`/`end` indices and absence of clipping, this is consistent with a predictable terminal-frame/index convention at the effective signal boundary. It is not consistent with random dropped frames: the 10 ms start cadence has no missing expected starts, and interior frames remain regular.

Verified facts:

- CARE turn endpoints are derived from RTTM participant timing for the affected artifacts.
- The extraction slices waveform samples using integer endpoints.
- openSMILE timestamps are copied into the feature artifact without boundary clipping.
- Every observed violation is terminal and approximately one sample long.
- The affected starts and midpoints remain inside the declared turns.

Not verified:

- The exact openSMILE release/configuration used to create the artifacts.
- Whether openSMILE’s terminal `end` field denotes true acoustic support, a padded analysis region, or a boundary timestamp convention.
- Original WAV sample counts and headers for the affected recordings.
- Whether integer sample truncation contributes to every affected case.

Integer sample truncation may contribute generally, but it is not sufficient as a complete explanation: at least the inspected boundary values are sample-aligned while the excess is still one sample. The unresolved openSMILE terminal-frame semantics therefore remain material.

## 5. Hypothetical policy comparison

These calculations were performed in memory for analysis only. No artifact, timestamp, validator, split, or dataset was changed.

### Policy A — strict frame containment

This is the current fail-closed rule. It is scientifically conservative because every retained frame interval is guaranteed to lie within the declared turn. It excludes 31 unique artifacts / 33 turns, repeated as 155 artifact and 165 turn occurrences across the five fold populations. Across the ten CARE fold populations, the corrected validator reports 1,830,705 candidate windows, 1,830,680 accepted at 80%, and 1,830,575 accepted at 95%.

Its cost is small but concentrated: affected terminal acoustic information is discarded, and the affected artifact is excluded rather than merely losing one boundary frame.

### Policy B — frame-start or frame-center membership

All 33 affected frames satisfy both start and midpoint membership. Under an explicitly documented in-memory interpretation that clips coverage to the declared turn and records a boundary-uncertainty mask, all 1,713 artifacts are retained. Aggregate CARE counts become 1,865,355 candidate windows, 1,865,330 accepted at 80%, and 1,865,220 accepted at 95%.

This preserves the terminal representation and restores coverage, but it treats an interval extending outside the declared turn as usable without proving what openSMILE’s terminal endpoint means. It is not acceptable as an implicit validator relaxation.

### Policy C — exclude affected boundary frames

This policy keeps interior frames and removes only frames whose interval crosses a declared turn boundary. In this audit it also retains all 1,713 artifacts. The aggregate window counts are the same as Policy B: 1,865,355 candidate, 1,865,330 accepted at 80%, and 1,865,220 accepted at 95%.

The equality occurs because the affected terminal frames do not change the predeclared window acceptance counts after boundary handling. Policy C loses the uncertain tail-frame values but avoids including audio beyond the declared turn. It requires an explicit frame-level exclusion mask and provenance field; it must not be implemented by silently editing the CSV artifacts.

Policy C is the most conservative candidate if a later provenance review confirms that the terminal frame’s acoustic support cannot be established. Policy B has greater information retention but higher semantic risk. Neither policy is approved by this investigation.

## 6. Scientific limitations

The artifact evidence strongly suggests a stable one-sample terminal edge convention, but the missing openSMILE version/configuration and unavailable source WAV headers prevent a complete causal attribution. The extraction script also does not persist effective sample endpoints, package versions, or frame-validity provenance, so those facts cannot be reconstructed from the current CSVs alone.

The analysis used only canonical CARE TRAIN/VAL folds 0–4. It did not read TEST data, decode audio, install Mantis, download checkpoints, run inference, extract features, or train models. No individual identifiers, records, audio, or predictions are included in this report.

The policy window totals are descriptive counterfactuals. They do not establish downstream model performance, clinical value, or superiority of one policy.

## 7. Recommendation

**NO-GO** for changing the current strict validator or starting MantisV2 temporalization.

Before any policy change, the manager should require a provenance repair that pins the openSMILE package/configuration and provides read-only access to representative source WAV headers or equivalent sample-count metadata. The terminal-frame semantics should then be verified with a synthetic 16 kHz signal and the recorded extraction configuration. If that review confirms that the terminal interval is only a boundary convention, Policy C (explicit exclusion with a validity mask) is the safer bounded policy; Policy B should require stronger justification for retaining the uncertain tail.

## 8. Reproduction and scope

The aggregate audit was performed with a temporary read-only analysis script outside the repository. No reproduction script was added because the existing validator must remain unchanged and the investigation logic was intentionally kept out of the active pipeline.

The extraction sources reviewed were:

- `care_scripts/audio_descriptors/extract_egemaps.py`
- `care_scripts/audio_descriptors/audio_utils.py`
- `care_scripts/turn_utils.py`
- `scripts/validate_mantis_egemaps_temporalization.py`
- `docs/evidence/MANTISV2_EGEMAPS_TEMPORAL_CONTRACT_EN.md`

No files outside this evidence report were modified.
