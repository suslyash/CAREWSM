# CAREWSM — 19-family CARE→WSM feature parity target

**Status: source-derived engineering contract** from accepted `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md` and `/media/maxim/Databases/CARE/splits/features.json`. The 19 families were observed in one bounded CARE example (`extracted/CONTROL/ASTHMA/PROFILE_053/VIDEO_000`); exact values, all alternate file variants, native pandas dtypes and CARE loader semantics remain unverified. Cross-corpus numerical *and semantic* parity requires executable producer comparison, not only the same column count.

## Input/output contract

For WSM, use manifest `splits/{train,dev,test}.csv` columns `video_path` (raw media) and `feature_dir` (feature destination), relative to `/media/maxim/Databases/WSM_NEW/`, with careful path validation. The observed output structure is `extracted/<label>/<subgroup>/PROFILE_<profile_id>/VIDEO_<segment_id>/`; **use the manifest field rather than constructing from sample-ID tokens**. Do not alter existing `transcript.txt` (present for 1,776 of 8,433 samples at audit). Do not touch existing fixed split/fold/label manifests. CARE producer reference scripts are read-only. All WSM audio feature production lives in `scripts/audio/`; video production in `scripts/video/` and linguistic in `scripts/text/` (new directories only when their own tasks authorize); generic cross-modal validation can live at `scripts/validate_care_feature_parity.py`. None of these import ChimeraML.

## Exact family checklist

| Stage | Family | Filename | Observed CARE representation | Observed numeric width / NPZ keys |
|---|---|---|---|---|
| 003A audio descriptors | egemaps_func | `egemaps_func.csv` | turn CSV, 3 metadata columns | 88 |
| 003A audio descriptors | egemaps_lld | `egemaps_lld.csv` | frame-in-turn CSV, 5 metadata columns | 25 |
| 003A audio descriptors | disvoice_prosody_static | `disvoice_prosody_static.csv` | turn CSV, 3 metadata columns | 103 |
| 003A audio descriptors | disvoice_prosody_dynamic | `disvoice_prosody_dynamic.csv` | segment-in-turn CSV, 5 metadata columns | 13 |
| 003A audio descriptors | disvoice_phonation_static | `disvoice_phonation_static.csv` | turn CSV, 3 metadata columns | 28 |
| 003A audio descriptors | disvoice_phonation_dynamic | `disvoice_phonation_dynamic.csv` | frame-in-turn CSV, 5 metadata columns | 7 |
| 003A audio descriptors | disvoice_articulation_static | `disvoice_articulation_static.csv` | turn CSV, 3 metadata columns | 488 |
| 003B deep audio | wav2vec_xlsr300m | `wav2vec_xlsr300m.npz` | turn embedding | `mean,std` float32 (turns,1024), `turn_ids` int64 |
| 003B deep audio | trillsson4_1 | `trillsson4_1.npz` | turn embedding | `mean,std` float32 (turns,1024), `turn_ids` int64 |
| 003B deep audio | emotionmsp_trajectories | `emotionmsp_trajectories.csv` | turn CSV, 3 metadata columns | `arousal,dominance,valence` (3) |
| 003B deep audio | emotionmsp_embeds | `emotionmsp_embeds.npz` | turn embedding | `embed` float32 (turns,1024), `turn_ids` int64 |
| 003C video | openface | `openface.csv` | frame CSV, 4 metadata columns | 710 numeric-like columns |
| 003C video | emonet_trajectories | `emonet_trajectories.csv` | frame CSV, 3 metadata columns | 10 |
| 003C video | emonet_embeds | `emonet_embeds.npz` | turn embedding | `mean,std` float32 (turns,256), `valid_ratio` float64, `turn_ids` int64 |
| 003C video | ear_blinking | `ear_blinking.csv` | frame CSV, 3 metadata columns | 4 |
| 003C video | mediapipe_bodypose | `mediapipe_bodypose.csv` | frame CSV, 3 metadata columns | 165 |
| 003D linguistic | linguistic_turnlevel | `linguistic_turnlevel.csv` | turn CSV, 3 metadata columns | 39 observed; registry `original_feature_dim=41` **unresolved** |
| 003D linguistic | linguistic_videolevel | `linguistic_videolevel.csv` | video CSV, no metadata columns in sampled row | 39 |
| 003D linguistic | mpnet_sentence_embeds | `mpnet_sentence_embeds.npz` | turn embedding | `embed` float32 (turns,768), `turn_ids` int32 |

The 11 audio rows include emotion, XLS-R and TRILLsson. CARE has **no separate registry files** for arousal/valence/dominance, voice quality or pauses: these are columns/subfamilies of the registered audio files. Never invent new feature filenames instead of matching the 19 existing names.

## Scientific, runtime and availability caveats

- Audio standard: CARE reference producers explicitly load **16 kHz mono** and select participant turns from diarization/RTTM; the exact implementation and parameterization must be preserved. DiariZen GPU runtime is feasible on three WSM TRAIN files but all three produced one detected speaker. The human speaker/participant identification check (TASK-002C) is **deferred** because the owner connects by SSH and cannot play audio there. Downstream outputs using longest-speaker selection are **provisional**; do not assert verified respondent-only segments.
- Video and speech-derived feature availability are not guaranteed uniformly. At audit, only 1,776 WSM files had `transcript.txt`; missing transcripts cannot be replaced by invented or zero-filled linguistic features. For visual models/checkpoints, verify actual data, model source and licenses before execution. Record `blocked_missing_input` / `blocked_missing_dependency` / `blocked_missing_weights` rather than manufacturing success.
- Reference CARE producers: `audio_descriptors/extract_egemaps.py` (eGeMAPS), `extract_disvoice.py` (five DisVoice outputs), `extract_wav2vec.py`, `extract_trillsson.py`, `extract_audeering_emotionmsp.py`; `linguistic_descriptors/extract_linguistics.py`, `extract_mpnet.py`; `video_descriptors/extract_openface.sh`, `extract_emonet.py`, `extract_earblink.py`, `extract_bodypose.py`. Read source as needed. Existing CARE numerical schemas are anchors, not permission to copy values or fabricate unavailable outputs.
- Output acceptance means exact filenames and directories, correct per-family producer semantics, columns/keys/order, dtypes, feature widths, timestamp/turn alignment and sound missing-data handling. Keep output writes atomic, idempotent and resumable; preserve transcripts and preexisting valid features. Validation should use a small real TRAIN extraction before bulk.
- Owner authorizes **new WSM feature outputs under `extracted/`** for CARE parity, initially capped at three TRAIN recordings in TASK-003A until source/schema sanity is reviewed. Read CARE only; no changes to WSM original media, `splits/` or existing registered feature files without specific approval. Then prioritize full TRAIN/DEV/TEST extraction as separate approved batch tasks, with no use of TEST labels/metrics for development.
- **Owner's permanent limited-traffic policy:** Codex may not initiate large downloads or installation of missing dependencies or weights. It supplies owner-verified commands, official links, revisions, expected traffic size or unknown-size warning, cache locations and verification. Owner downloads manually; Codex reuses caches. Existing `.venv-diarizen` is a documented approved exception; don't spawn new environments automatically.

## Accelerated execution queue

1. **TASK-003A now:** implement, execute and validate seven eGeMAPS/DisVoice descriptor producers on at most three WSM TRAIN recordings using real extracted outputs; support resumable batch mode; preserve all CARE schema semantics.
2. **TASK-003B immediately after PASS:** XLS-R, TRILLsson and emotion/voice trajectories and embeddings; manual weight provisioning if not cached; independently verify 4 outputs.
3. **TASK-003C:** OpenFace, EmoNet, EAR and MediaPipe body pose; verify five outputs.
4. **TASK-003D:** linguistic turn/video features and MPNet embeddings; only for genuine available transcripts / validated segment alignment; verify three outputs.
5. **TASK-003E:** batch-extract the remaining WSM eligible train/dev/test rows with cost estimate, resumability, feature-by-feature coverage and schema/shape verification. At no stage claim 19/19 across all rows when underlying inputs, weights or turn identity are unverified.

Status progresses per family; a task's PASS never silently denotes all-19 parity.
