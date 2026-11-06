# CARE eGeMAPS Source Provenance Gate

Date: 2026-11-06
Required main: `be782c0ce9c06748ef308b5a8f177eb0f6a0186e`
Branch: `codex/mantisv2-egemaps-terminal-provenance-gate`

## 1. Gate decision

**PROVENANCE INCOMPLETE.**

Existing source evidence resolves the extraction algorithm and shows that the CARE anomaly is a deterministic terminal-frame boundary pattern. It does not resolve the historical openSMILE/librosa environment or provide source WAV sample-count metadata. No policy change is approved in this task.

A strictly bounded future correction using **Policy C**—explicitly exclude only terminal frames whose intervals cross declared turn boundaries while retaining interior frames—remains scientifically defensible as a conservative implementation candidate. It is conditional on manager approval and on preserving an explicit frame-validity mask. The current strict validator remains unchanged and correctly reports NO-GO.

## 2. Confirmed extraction facts

The repository source at `care_scripts/audio_descriptors/extract_egemaps.py` confirms:

- `librosa.load` is called with `sr=16000` and `mono=True`.
- openSMILE is constructed with `FeatureSet.eGeMAPSv02`.
- LLD timestamps come directly from the openSMILE DataFrame index levels `start` and `end`, shifted by the float `start_sec`.
- Turn slices use `int(start_sec * 16000)` and `int(end_sec * 16000)` with Python’s half-open waveform slice.
- The code does not clip, round, or validate `frame_end_sec` against `turn_end_sec` before writing CSV.
- The CLI supports `--audio-dir`, `--output-dir`, `--metadata-csv`, `--use-diarization`, and `--is-functional`, but the historical invocation is not recorded.

`audio_utils.py` confirms that diarization is used only when `--use-diarization` is supplied and an output-directory RTTM exists. `turn_utils.py` merges participant RTTM segments across gaps up to 3 seconds, treats interviewer segments as hard boundaries, and removes turns shorter than 0.3 seconds.

The observed affected artifacts contain multiple participant turns and RTTM metadata. The prior aggregate audit found RTTM participant endpoints corresponding to all 33 affected declared turn ends. Thus the declared turn endpoint is diarization-derived, while the effective openSMILE analysis endpoint is not separately persisted.

## 3. Historical provenance search

| Evidence item | Result | Status |
|---|---|---|
| openSMILE version | No version in repository, artifact metadata, or lock file | UNKNOWN |
| openSMILE configuration | Source selects eGeMAPSv02 and LLD level; other historical config unavailable | PARTIAL |
| Historical extraction arguments | CLI exists; actual command/invocation record unavailable | UNKNOWN |
| librosa version | No pinned version or historical environment record | UNKNOWN |
| Audio loading contract | 16 kHz mono requested in source code | CONFIRMED |
| Frame timestamp source | openSMILE index `start`/`end` plus turn start | CONFIRMED |
| Environment/package manifest | No applicable project lock or extraction manifest found | UNAVAILABLE |
| Current openSMILE import | Import unavailable in current environment | CONFIRMED UNAVAILABLE |

The current environment must not be treated as historical evidence. Because openSMILE is unavailable and its historical version is not positively identified, the synthetic openSMILE test was not run. No newer package was substituted.

## 4. Source recording metadata

The existing CARE artifact tree contains extracted CSVs and RTTM files but no original WAV files or WAV headers for the affected recordings. The repository split metadata does not provide authoritative source sample counts for those recordings. Therefore the following cannot be verified from current source evidence:

- Original WAV sampling rate and sample count for each affected recording.
- Whether integer endpoint conversion shortened any source slice by a fractional sample.
- The exact effective signal length presented to historical openSMILE.
- Whether the terminal `end` index represents acoustic support, a padded analysis region, or an endpoint convention.

No audio was decoded. No TEST data were accessed.

## 5. What the existing artifact evidence establishes

The prior CARE TRAIN/VAL audit established:

- 1,713 unique manifest-valid CARE artifacts were inspected.
- 31 unique artifacts and 33 unique turns were affected.
- Repeated five-fold population accounting produced 155 artifact occurrences and 165 turn occurrences.
- Every violation was on the terminal frame.
- Excess was one 16 kHz sample: median, P90, P99 and maximum approximately 0.0625 ms.
- All affected frame starts and midpoints remained inside the declared turns.
- Interior frames were approximately 20 ms with a 10 ms start cadence.
- No missing expected 10 ms frame starts were detected.
- Terminal frame durations were materially longer and variable, with a 50 ms median and maximum approximately 59.9375 ms.

These facts support the hypothesis of a stable terminal-frame/index convention rather than dropped frames. They do not prove that the terminal interval beyond the declared turn is valid acoustic content.

## 6. Boundary-policy assessment

### Policy A — strict frame containment

Keep the current validator rule and exclude affected artifacts. This is the only currently implemented and fully fail-closed policy. It guarantees that every retained interval lies within the declared turn, but removes the entire affected artifact rather than only its uncertain terminal frame.

### Policy B — frame-start or frame-center membership

All affected frames satisfy start and midpoint membership. This would preserve more data, but it would authorize an interval whose declared end exceeds the turn boundary without establishing the historical openSMILE semantics. It requires a documented uncertainty mask and is not defensible as an implicit relaxation.

### Policy C — explicit terminal-frame exclusion

Remove only rows whose frame interval crosses a declared turn boundary in the in-memory representation, retain valid interior rows, and persist a frame/window validity mask in any future derived cache. Do not rewrite source CSVs, clip timestamps, or alter turn endpoints.

This is conservative because it never includes unverified audio outside the declared turn and loses only the boundary frame, while preserving the overwhelming majority of interior observations. The previous counterfactual audit showed that Policy C retained all 1,713 artifacts and produced the same predeclared window acceptance totals as start/center handling for the observed data. That coverage result is descriptive and does not authorize implementation.

## 7. Reproducibility limitations and next gate

The provenance search is exhausted for the existing repository and artifact sources. Repeating the same search is unlikely to resolve the missing historical environment. The next gate should require, before implementation:

1. A pinned historical openSMILE package/version and relevant configuration, or an explicit acknowledgement that they cannot be recovered.
2. Read-only WAV header or equivalent sample-count metadata for representative affected and unaffected TRAIN/VAL recordings.
3. A synthetic 16 kHz frame-index test using the positively identified historical implementation, if recoverable.
4. Manager approval of Policy C, including its frame-level validity-mask and cache-provenance fields.
5. A rerun of the unchanged fail-closed audit plus a separate policy-C audit showing no retained out-of-turn frames.

If the historical package remains unrecoverable, Policy C can still be approved as a conservative data-handling rule only because it discards, rather than interprets, the unverified terminal interval. Policy B should not be approved on the present evidence.

## 8. Scope and recommendation

No Mantis package or checkpoint was installed or downloaded. No inference, resampling, cache generation, feature extraction, audio decoding, training, or TEST access occurred. The current validator, datasets, extracted features, timestamps, and historical experiments were not modified.

**Final recommendation: PROVENANCE INCOMPLETE. Do not weaken the strict validator now. Hold MantisV2 temporalization until the manager either recovers the missing provenance or explicitly approves Policy C as a conservative frame-exclusion correction with masks and a reproducibility record.**
