# TASK-001 Feature Audit Evidence

## Executive result

The corrected audit inspected 10,464 manifest rows in 428.59s. WSM artifacts are resolved through `extracted/<label>/<subgroup>/PROFILE_<id>/VIDEO_<id>`; the legacy CSV `video_dir` is retained only as evidence. CARE and WSM NPZ files pass dependency-light NPY header and actual numeric-value validation; NaN/Inf values are classified as NONFINITE.

## Registry and temporal semantics

| family | artifact | level | provisional Mantis classification |
|---|---|---|---|
| egemaps_func | egemaps_func.csv | turn | MANTIS_WINDOWABLE_FROM_TIMESTAMPS |
| egemaps_lld | egemaps_lld.csv | frame_in_turn | MANTIS_WINDOWABLE_NATIVE |
| disvoice_prosody_static | disvoice_prosody_static.csv | turn | NATIVE_ONLY_NO_DEFENSIBLE_1S_AXIS |
| disvoice_prosody_dynamic | disvoice_prosody_dynamic.csv | segment_in_turn | MANTIS_WINDOWABLE_FROM_TIMESTAMPS |
| disvoice_phonation_static | disvoice_phonation_static.csv | turn | NATIVE_ONLY_NO_DEFENSIBLE_1S_AXIS |
| disvoice_phonation_dynamic | disvoice_phonation_dynamic.csv | frame_in_turn | MANTIS_WINDOWABLE_FROM_TIMESTAMPS |
| disvoice_articulation_static | disvoice_articulation_static.csv | turn | NATIVE_ONLY_NO_DEFENSIBLE_1S_AXIS |
| wav2vec_xlsr300m | wav2vec_xlsr300m.npz | turn_embedding | NATIVE_ONLY_NO_DEFENSIBLE_1S_AXIS |
| trillsson4_1 | trillsson4_1.npz | turn_embedding | NATIVE_ONLY_NO_DEFENSIBLE_1S_AXIS |
| emotionmsp_trajectories | emotionmsp_trajectories.csv | turn | MANTIS_WINDOWABLE_FROM_TIMESTAMPS |
| emotionmsp_embeds | emotionmsp_embeds.npz | turn_embedding | NATIVE_ONLY_NO_DEFENSIBLE_1S_AXIS |

## Split/fold/profile contract

CARE has no `splits/videos.csv`. The split checks are computed independently per corpus. CARE has 276 train/dev sample overlaps and 43 train/dev profile overlaps; WSM has 936 train/dev sample overlaps and 100 train/dev profile overlaps. Neither corpus has train/test or dev/test sample/profile overlap. Five fold train/val files exist for both corpora and have zero train/val sample/profile overlap; pairwise fold validation overlaps are recorded in JSON.

| corpus | split | rows | profiles | declared mismatch |
|---|---|---:|---:|---:|
| CARE | train | 1713 | 240 | 276 |
| CARE | dev | 276 | 43 | 0 |
| CARE | test | 318 | 41 | 0 |
| WSM | train | 7069 | 638 | 936 |
| WSM | dev | 936 | 100 | 0 |
| WSM | test | 1364 | 101 | 0 |

Joint fold totals:

| fold | train rows | val rows |
|---:|---:|---:|
| 0 | 7001 | 1781 |
| 1 | 7029 | 1753 |
| 2 | 7048 | 1734 |
| 3 | 7022 | 1760 |
| 4 | 7028 | 1754 |

## Status and coverage

The manifest contains one status and boolean validity column per family. Selected status counts:

| corpus | family | VALID | MISSING | EMPTY | INVALID_DTYPE |
|---|---|---:|---:|---:|---:|
| CARE | egemaps_func | 2031 | 0 | 0 | 0 |
| CARE | egemaps_lld | 2031 | 0 | 0 | 0 |
| CARE | disvoice_prosody_static | 935 | 0 | 0 | 1096 |
| CARE | disvoice_prosody_dynamic | 2004 | 0 | 0 | 27 |
| CARE | disvoice_phonation_static | 1938 | 0 | 0 | 93 |
| CARE | disvoice_phonation_dynamic | 2031 | 0 | 0 | 0 |
| CARE | disvoice_articulation_static | 2020 | 0 | 0 | 11 |
| CARE | wav2vec_xlsr300m | 2031 | 0 | 0 | 0 |
| CARE | trillsson4_1 | 2031 | 0 | 0 | 0 |
| CARE | emotionmsp_trajectories | 2031 | 0 | 0 | 0 |
| CARE | emotionmsp_embeds | 2031 | 0 | 0 | 0 |
| WSM | egemaps_func | 8429 | 3 | 0 | 1 |
| WSM | egemaps_lld | 8429 | 3 | 0 | 1 |
| WSM | disvoice_prosody_static | 6922 | 3 | 0 | 1508 |
| WSM | disvoice_prosody_dynamic | 8353 | 3 | 0 | 77 |
| WSM | disvoice_phonation_static | 8344 | 3 | 0 | 86 |
| WSM | disvoice_phonation_dynamic | 8345 | 3 | 85 | 0 |
| WSM | disvoice_articulation_static | 8385 | 3 | 0 | 45 |
| WSM | wav2vec_xlsr300m | 8430 | 3 | 0 | 0 |
| WSM | trillsson4_1 | 8430 | 3 | 0 | 0 |
| WSM | emotionmsp_trajectories | 8430 | 3 | 0 | 0 |
| WSM | emotionmsp_embeds | 8430 | 3 | 0 | 0 |

Global complete-case across all 11 families: **7,774/10,464 (74.29%)**.

Pairwise joint-valid counts are recorded in `audit_summary.json` under `pairwise_joint_valid`; diagonal entries are single-family valid counts. This includes the required pairwise matrix without pooling or deleting samples.

## Observed label coverage

```json
{
  "CARE": {
    "depression": {
      "negative": 1498,
      "observed": 2031,
      "positive": 533
    },
    "parkinson": {
      "negative": 1827,
      "observed": 2031,
      "positive": 204
    }
  },
  "WSM": {
    "depression": {
      "negative": 3028,
      "observed": 5111,
      "positive": 2083
    },
    "parkinson": {
      "negative": 2026,
      "observed": 3322,
      "positive": 1296
    }
  }
}
```

## Manifest and reproducibility

- path: `/media/maxim/Programs/Projects/CAREWSM/reports/feature_audit/sample_feature_availability.csv`
- rows: 10,464; bytes: 2,592,264
- SHA256: `bacf3b9715d0ff38f39596934bfcfe13da8dddbc5177a22ff362a56fdf389533`
- command: `python3 -m src.audio.data.feature_audit --out reports/feature_audit`

## Tests, limitations and recommendation

Focused synthetic tests cover valid, missing, corrupt/unreadable, empty, wrong schema/shape, invalid dtype, NaN/Inf and timestamp order. The project venv runs pytest. WSM has six missing rows per family under the corrected mapping; this is real artifact absence, not a path-mapping failure. The split inconsistency and absent CARE videos registry remain explicit manager-review findings.

TASK-002 should consume the corrected extracted-path mapping and preserve the split/fold contract findings. Only timestamp-bearing families are candidates for defensible 1.0-second/0.5-second windowization; turn/static embeddings remain native-only pending an approved mapping.
