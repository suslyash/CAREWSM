# TASK-001 Feature Audit Evidence

## Executive result

The live CARE registry contains 19 sources: 11 audio, 5 video and 3 text. The operational audio registry is the 11 names below. The deterministic audit inspected 11,400 split rows (CARE 2,031; WSM 9,369) and wrote one row per sample/family status. No source artifact or canonical split was modified.

## Operational audio registry and temporal evidence

| family | artifact | level | temporal/Mantis classification |
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

CSV timestamp metadata are `turn_start_sec`/`turn_end_sec`; frame/segment sources additionally expose `frame_start_sec`/`frame_end_sec`. NPZ sources expose turn IDs but no timing fields. Valid-observation/confidence fields are not present in the 11 audio schemas.

## Split/fold/label sanity

CARE has train/dev/test CSVs but no `splits/videos.csv`; its split rows expose `sample_id`, `profile_id`, `video_dir`, `split`, `cv_fold`, and `label_id`. WSM has `videos.csv` and train/dev/test CSVs, but `train.csv` contains rows whose embedded `split` value is `dev`; this is recorded, not normalized. WSM `videos.csv` exposes `y_depression` and `y_parkinson`, with empty values representing unavailable labels. Five fold train/val files exist for both corpora. The audit does not regenerate or pair manifests.

## Coverage and failures

| corpus | rows | notable statuses |
|---|---:|---|
| CARE | 2,031 | eGeMAPS and emotion trajectories valid for all; DisVoice static/dynamic have INVALID_DTYPE failures; three NPZ families are UNREADABLE under the required NPZ schema |
| WSM | 9,369 | all 11 families MISSING under the split `video_dir` mapping |

Exact counts, per-family status counts, and deterministic manifest metadata are in `reports/feature_audit/audit_summary.json`. Pairwise and complete-case coverage is represented by the per-family boolean columns and can be calculated without re-opening artifacts; no complete-case promotion is made by this audit because WSM has no valid mapped artifact rows.

## Manifest and reproducibility

- path: `/media/maxim/Programs/Projects/CAREWSM/reports/feature_audit/sample_feature_availability.csv`
- rows: 11,400; bytes: 2,994,550
- SHA256: `505d5f61ed917de081f8cec111c0bb2704ea5eaef6570e98a9c3d291cc8c7d9f`
- runtime: 137.99 s
- command: `python3 -m src.audio.data.feature_audit --out reports/feature_audit`

## Tests and limitations

Reusable validation functions and focused synthetic tests cover missing, empty, valid, schema-mismatch, non-finite and invalid-dtype CSV artifacts. The environment has no `pytest` executable/module, so the focused test command could not run. NPZ artifacts were opened with `allow_pickle=False`; unreadable NPZ statuses are preserved. WSM path mapping and the split anomaly require manager review before TASK-002.

## Recommendation for TASK-002

First reconcile WSM split `video_dir`/feature-directory semantics and the CARE missing `videos.csv` contract. Then freeze a data-layer identity mapping. Only timestamp-bearing families should proceed to native 1-second/0.5-second windowization; turn/static and turn-embedding families remain native-only until a later approved mapping.
