# CAREWSM MantisV2 Feasibility and Temporalization Audit

Date: 2026-11-06  
Required base: `a9bbdf3032c965a88120ccb18a19b2579860fdaa`  
Branch: `codex/mantisv2-feasibility`

## 1. Executive finding

Decision: **CONDITIONAL GO, limited to existing timestamped frame/segment
features**.

MantisV2 is technically usable as a frozen representation extractor, but the
current 1,024-dimensional deep NPZ embeddings are not valid Mantis time-series
inputs. Their first dimension is an ordered speaker-turn index, not a sampled
time axis, and WSM contains only one turn per artifact. Repeating or
interpolating these vectors to 512 points would manufacture temporal evidence
and is rejected.

The defensible first route is a small pilot using one existing genuinely
time-indexed family, preferably `egemaps_lld`, with its frame timestamps and a
frozen, explicitly defined 1.0-second/0.5-second windowing rule. The existing
dynamic DisVoice families are also candidates, subject to a separate channel
and validity contract. This audit does not implement extraction, install
`mantis-tsfm`, download weights, run inference, or train a classifier.

## 2. Actual CARE/WSM data inventory

The inspection used the existing availability manifest, authoritative feature
registry, source split metadata, active Dataset/DataModule, and direct reads of
live extracted artifacts. No sample identifiers or person-level records are
included here.

The existing audit covers 10,464 manifest rows: 2,034 CARE and 8,430 WSM.
The four DEEP-001 families have 10,461 valid rows; the three missing rows are
in WSM. The complete-case count across all 11 registered families is 7,774
(74.29%). The fixed five-fold joint populations and profile-grouped splits are
unchanged. The active loader preserves native ordered turns, masks labels, and
filters only the selected feature families.

| Family | Native artifact | Level | Channels | Native sequence evidence |
|---|---|---|---:|---|
| `emotionmsp_embeds` | NPZ `embed`, `turn_ids` | turn embedding | 1,024 | CARE T=1–55; WSM T=1 |
| `trillsson4_1` | NPZ `mean`, `std`, `turn_ids` | turn embedding | 1,024 selected `mean` | CARE T=1–55; WSM T=1 |
| `wav2vec_xlsr300m` | NPZ `mean`, `std`, `turn_ids` | turn embedding | 1,024 selected `mean` | CARE T=1–55; WSM T=1 |
| `emotionmsp_trajectories` | CSV with turn timestamps | turn | 3 | CARE T=1–55; WSM T=1 |
| `egemaps_lld` | CSV with frame timestamps | frame in turn | 25 | real frame timestamps |
| `egemaps_func` | CSV with turn timestamps | turn aggregate | 88 | one aggregate per turn |
| `disvoice_prosody_dynamic` | CSV with frame/segment timestamps | segment in turn | 13 | real segment timestamps |
| `disvoice_phonation_dynamic` | CSV with frame timestamps | frame in turn | 7 | real frame timestamps |

Representative live eGeMAPS LLD rows expose `turn_start_sec`, `turn_end_sec`,
`frame_start_sec`, and `frame_end_sec`; the inspected frame interval is 20 ms.
The live trajectory artifacts expose turn timestamps, but not a dense sample
clock. The inspected CARE trajectory turn durations range from approximately
0.3 s to 345.18 s, with a median of approximately 12.10 s. These are turn
boundaries, not evidence that turn-level embeddings are sampled at a fixed
rate.

All inspected valid NPZ arrays were finite, readable, non-empty, and shaped as
`[T,1024]`. All inspected trajectory arrays were finite, readable, and shaped
as `[T,3]`. The existing audit records the complete per-family status counts,
including missing, empty, invalid-dtype, and nonfinite categories; this report
does not replace that manifest.

The available native metadata does not establish a single common sampling rate
for all families. CARE has variable turn counts; WSM's extracted deep and
trajectory artifacts are all single-turn. No valid global time axis may be
inferred from turn IDs.

## 3. Verified MantisV2 technical contract

The authoritative sources consulted were the official [Mantis repository](https://github.com/vfeofanov/mantis),
its [MantisV2 implementation](https://github.com/vfeofanov/mantis/blob/main/src/mantis/architecture/version2.py),
the [official MantisV2 checkpoint page](https://huggingface.co/paris-noah/MantisV2),
and the [MantisV2 paper](https://arxiv.org/abs/2602.17868).

### Architecture and API

- Official package: `mantis-tsfm`, repository `main` currently declares
  version `1.1.0`, Python `>=3.10,<4.0`, and dependencies including PyTorch,
  NumPy, Hugging Face Hub, safetensors, and scikit-learn.
- MantisV2 has 4,188,672 parameters in the full architecture. The official
  intermediate representation path returns a 2,214,144-parameter truncated
  computation according to the implementation documentation.
- The class is `mantis.architecture.MantisV2`; the repository also provides
  `MantisTrainer.transform(X)` for frozen feature extraction.
- The official recommended frozen extraction setting is
  `return_transf_layer=2` and `output_token='combined'`. The combined token is
  the concatenation of the classification token and the mean of non-class
  tokens, so the default hidden dimension 256 yields a 512-dimensional output.

### Input contract

- Inputs are multivariate time series, with channels treated as variables and
  the sequence axis treated as time. The repository states that sequence
  length must be a multiple of 32.
- The official example resizes to 512 points with linear interpolation because
  the model was pretrained at that length. This is an input-length operation,
  not permission to invent a time axis for turn embeddings.
- The V2 token generator includes per-variable scaling and the architecture
  accepts multichannel data. The repository documents `MultichannelProjector`
  for PCA-based channel reduction and `LinearChannelCombiner` for a learned
  adapter. Any PCA/scaler must be fitted on the current TRAIN population only;
  a learned adapter would be a future trainable component, not frozen
  preprocessing.
- The public frozen `transform` API is suitable after a valid fixed-shape
  tensor has been constructed. The API documentation does not provide a
  CAREWSM-specific padding mask or irregular-time representation. Variable
  lengths therefore require explicit window construction, valid-window masks,
  and batch padding outside Mantis; padding must never be passed as real audio
  observations without an audited rule.

### Checkpoint, revision, software, and license

The checkpoint identifier is `paris-noah/MantisV2`. The Hugging Face page
currently exposes immutable commit `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`
(the page shows the short form `99fe0f5`), which is the revision to pin for a
future pilot. This audit did not download the checkpoint. The upstream package
source declares version `1.1.0` and Apache-2.0 licensing; the Hugging Face
model card also declares Apache-2.0. The V2 source class metadata contains an
inconsistent MIT string, so legal review should use the repository/checkpoint
license files rather than the class metadata alone.

The current CAREWSM environment was not altered and does not establish that
the Mantis dependencies are installed or compatible with Chimera-ML. A future
implementation step must verify this in an isolated environment before any
weights are downloaded.

## 4. Comparison of integration routes

| Route | Scientific validity | Coverage/shape | Main requirements and risks | Decision |
|---|---|---|---|---|
| A. Existing turn embeddings | Invalid as Mantis time series. `[T,1024]` is turn order, not time; WSM is T=1. | Very high artifact coverage; no defensible 1-s clock. | Would require fake repetition, stretching, or interpolation across turns. Loses physical meaning and can create leakage-like artifacts. | **NO-GO** |
| B. Existing timestamped features | Defensible for frame/segment families with real timestamps. eGeMAPS LLD is the cleanest first candidate; dynamic DisVoice families need their own semantic audit. | High coverage per existing audit; frame/segment lengths are variable and gaps/turn boundaries must be handled. | Choose channels, preserve timestamps, define turn-boundary and missing-observation rules, then create 1-s windows/0.5-s hops. No cross-family alignment may be assumed without timestamps. | **CONDITIONAL** |
| C. New dense temporal audio features | Scientifically defensible if extracted from original 16-kHz mono recordings with a fixed feature definition and auditable time base. | Depends on original recording availability and extraction success; cannot be claimed from current artifacts. | New extraction, quality/validity manifest, fixed frame hop, window construction, and fold-safe preprocessing. Cost is a new extraction pass plus Mantis inference/cache storage. | **CONDITIONAL** |

Route B is the recommended feasibility path because it reuses already audited
timestamped artifacts and does not require re-extracting audio. It must remain
separate from the existing static/turn embedding experiments.

## 5. Dataset coverage and temporal-resolution assessment

The protocol's 1.0-second windows and 0.5-second hop are feasible only for a
source with a real time coordinate. For eGeMAPS LLD, the inspected rows carry
20-ms frame intervals and turn start/end metadata, so a 1.0-second window can
be defined over the recorded frame timestamps. The future implementation must
state whether windows may cross turn boundaries. The conservative first pilot
should keep windows within a turn, emit a validity/confidence fraction, and
discard or mask windows without sufficient observed frames. It must not bridge
silence or turn gaps by copying the last frame.

For segment features, the same rule is possible only after specifying how
segments map to a window and what valid-observation fraction means. Turn-level
functionals, trajectories without dense timestamps, and all three deep NPZ
families cannot satisfy this contract by themselves.

The current artifacts do not provide a universal original recording sample
rate alongside every extracted feature. Project documentation states that
source audio preprocessing was 16 kHz mono, but the feature audit—not a
reconstructed audio clock—must remain authoritative for frame timestamps.
No global preprocessing is required for the frozen encoder; any scaler/PCA
used before Mantis or after Mantis must be fitted separately within each joint
TRAIN fold.

## 6. Reproducibility and leakage controls

A future pilot must:

1. Pin CAREWSM source SHA, the Mantis package version, checkpoint revision
   `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`, selected layer 2, and combined
   output token.
2. Pin the exact source family, channel list/order, frame/segment acceptance
   rule, 1.0-s window, 0.5-s hop, boundary policy, and interpolation policy.
3. Keep native timestamps, window start/end, source row indices only in local
   cache metadata, validity masks, and valid-observation fractions. The
   committed evidence must contain aggregates only.
4. Build caches from fixed CARE/WSM source splits without using labels to alter
   feature extraction or window validity.
5. Fit any scaler, PCA, or trainable channel adapter using only joint
   `CARE fold_k TRAIN + WSM fold_k TRAIN`; apply the frozen state to VAL and
   TEST without refitting.
6. Preserve missing Depression/Parkinson labels and the canonical joint
   validation protocol. Select checkpoints from validation only; TEST remains
   descriptive.
7. Fingerprint source artifacts, registry, split files, preprocessing state,
   Mantis package, checkpoint revision, and cache schema.
8. Never duplicate checkpoint weights per video; store only cache metadata and
   derived embeddings.

## 7. Recommended minimal pilot

The pilot is **proposed only and was not executed**.

- **Source:** `egemaps_lld`, using an explicitly frozen subset or all 25
  audited acoustic channels in their registry order.
- **Time construction:** retain the source 20-ms frame timestamps; form
  within-turn 1.0-second windows at 0.5-second hop; require a documented
  minimum valid-frame fraction; do not cross turn gaps in the first pilot.
- **Mantis:** `mantis-tsfm==1.1.0`, `MantisV2`, checkpoint revision
  `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`, `return_transf_layer=2`,
  `output_token='combined'`, frozen weights.
- **Length handling:** construct the 1-second source window first, then apply a
  deterministic, recorded resampling rule to the Mantis-required multiple-of-
  32 input length. The pilot must test that rule on timestamps and document
  the resulting point count; it must not resize turn embeddings.
- **Normalization:** begin with Mantis' documented internal scaling and no
  learned global transform. If external standardization is required, fit it
  on each joint TRAIN fold only and record its fingerprint.
- **Cache:** one per-video ordered array of window embeddings, window
  boundaries, validity mask, confidence/fraction, source-family metadata,
  split/fold metadata, and checkpoint/preprocessing fingerprints.
- **Protocol:** compare a frozen-Mantis-derived pooled reference against the
  strongest existing native EmotionMSP baseline under the same joint
  CARE+WSM five-fold validation protocol. Do not alter historical runs.
- **Resources:** this should be a small isolated cache/inference pilot before
  any classifier matrix. Storage and runtime must be measured from a small
  TRAIN-only pilot subset before full derivation; no estimate is claimed here.
- **Success:** every retained window has auditable timestamps and finite
  values; coverage is reported by corpus/fold; no split/profile leakage is
  found; extraction is reproducible from fingerprints; validation comparison
  is complete without TEST-driven selection.
- **Failure:** missing/ambiguous time semantics, inadequate valid-window
  coverage, package/checkpoint incompatibility, or any leakage invalidates the
  route and returns it to NO-GO without model training.

## 8. Exact next atomic implementation step

Create a read-only, TRAIN/VAL-only `egemaps_lld` temporalization contract
validator. It should inspect timestamps and channel schemas, report aggregate
window coverage under the proposed within-turn 1.0-s/0.5-s rule, and validate
the cache schema on synthetic tensors. It must not install Mantis, download
weights, run Mantis inference, access TEST for selection, extract new audio
features, or train a model. Only after that validator is manager-reviewed
should a separate task authorize isolated Mantis package/checkpoint setup.

## Limitations

This is a feasibility audit, not a performance experiment. It does not prove
that MantisV2 transfers to CARE/WSM, that the proposed windows maximize signal,
or that a future pilot will improve the native baseline. Existing split
metadata contains previously documented train/dev inconsistencies; this audit
does not alter or reinterpret those historical findings. No raw audio, person-
level prediction, sample identifier, checkpoint, or Mantis-derived feature was
committed.
