# TASK-001A — CAREWSM corpus, feature-contract, and legacy-repository audit

Audit date: 2026-10-16. Base: `0629f35970ad47bad2d05298bdf96c2540ef6b64`. Branch: `codex/task-001a-corpus-feature-repo-audit`.

This read-only audit records observed facts before cleanup. No training, extraction, diarization, model loading, Test evaluation, or dataset write was performed.

## 1. Executive result

Both corpora have fixed train/dev/test CSVs and five train-internal folds. Sample and profile partitions are disjoint across partitions and within every fold pair; fold validation unions equal canonical train exactly.

Deviations: CARE has no `splits/videos.csv`, and the requested `splits/care_features.py` is absent. WSM has explicit task-level missing labels. WSM `extracted/` contains only transcripts, not CARE-registered features.

## 2. CARE splits / folds / labels

Columns are `sample_id,label_id,profile_id,subgroup,video_id,video_dir,split,cv_fold`; the row unit is one video. `sample_id` is unique and `profile_id` is the grouping key. Labels are complete.

| file | rows | samples | profiles | label counts |
|---|---:|---:|---:|---|
| train | 1,437 | 1,437 | 197 | CONTROL 931; DEPRESSION 365; PARKINSON 141 |
| dev | 276 | 276 | 43 | CONTROL 176; DEPRESSION 71; PARKINSON 29 |
| test | 318 | 318 | 41 | CONTROL 187; DEPRESSION 97; PARKINSON 34 |

Train/dev/test sample and profile overlap is zero. Fold train/validation rows (profiles) are `0: 1132/305 (157/40)`, `1: 1149/288 (157/40)`, `2: 1122/315 (157/40)`, `3: 1172/265 (158/39)`, `4: 1173/264 (159/38)`. Every side is a canonical-train subset, pair overlap is zero, and validation covers all 1,437 train samples once. `subgroup` has 21 observed values and is not equivalent to `label_id`. **PASS.**

## 3. WSM splits / folds / labels

WSM has `videos.csv` and 36-column split files. In addition to common identifiers, fields include `dataset,task,diagnosis,label,y_depression,y_parkinson,participant_id,source_video_id,source_group_id,segment_id,segment_file,stratum,source_group,topic_group,storage_type,video_path,feature_dir,archive_relative_path,tar_video_prefix,n_feature_files,available_modalities,missing_modalities,source_csv_path,source_row,csv_video_id,csv_diagnosis,csv_segment_file`. The row unit is one listed segment.

| file | rows | samples/profiles | label counts |
|---|---:|---:|---|
| train | 6,133 | 6,133 / 538 | CONTROL 3,644; DEPRESSION 1,431; PARKINSON 1,058 |
| dev | 936 | 936 / 100 | CONTROL 514; DEPRESSION 317; PARKINSON 105 |
| test | 1,364 | 1,364 / 101 | CONTROL 896; DEPRESSION 335; PARKINSON 133 |

`y_depression` 1/0/empty: train 1,431/2,229/2,473; dev 317/307/312; test 335/492/537. `y_parkinson` 1/0/empty: train 1,058/1,415/3,660; dev 105/207/624; test 133/404/827. Empty values are unavailable and must remain masked.

Partition overlap is zero. Fold validation sizes are 1,227, 1,227, 1,227, 1,226, 1,226; all are disjoint, canonical-train subsets and collectively cover 6,133 samples once. Raw organization is `depression|parkinson/{train_labels,dev_labels,test_labels}/<source-video>/segments/*.mp4`; source-profile counts are depression 294/60/59 and parkinson 274/45/47. Existing summary metadata records 318 ignored duplicate-profile occurrences in train; assignments were not regenerated. **PASS.**

## 4. Cross-corpus supervision implications

CARE is complete mutually exclusive three-way source labeling. WSM contains explicit per-task unknowns. A unified manifest should derive two independent targets and carry a two-element observed mask plus corpus/source-task provenance. Cross-corpus subgroup equivalence is not established.

## 5. CARE feature registry

`features.json` is `CARE_FEATURE_REGISTRY_V3`, version 3; Human annotations, Discourse markers, `start_sec`, and `end_sec` are excluded.

| name | file | level/policy | dim; aggregation |
|---|---|---|---|
| egemaps_func | egemaps_func.csv | turn/pool | 88; mean_std |
| egemaps_lld | egemaps_lld.csv | frame_in_turn/sequence | 25; none |
| disvoice_prosody_static | disvoice_prosody_static.csv | turn/pool | 103; mean_std |
| disvoice_prosody_dynamic | disvoice_prosody_dynamic.csv | segment_in_turn/sequence | 13; none |
| disvoice_phonation_static | disvoice_phonation_static.csv | turn/pool | 28; mean_std |
| disvoice_phonation_dynamic | disvoice_phonation_dynamic.csv | frame_in_turn/sequence | 7; none |
| disvoice_articulation_static | disvoice_articulation_static.csv | turn/pool | 488; mean_std |
| wav2vec_xlsr300m | wav2vec_xlsr300m.npz | turn_embedding/pool | 2048; mean_std |
| trillsson4_1 | trillsson4_1.npz | turn_embedding/pool | 2048; mean_std |
| emotionmsp_trajectories | emotionmsp_trajectories.csv | turn/pool | 3; mean_std |
| emotionmsp_embeds | emotionmsp_embeds.npz | turn_embedding/pool | 1024; mean_std |
| openface | openface.csv | frame/sequence | 710; none |
| emonet_trajectories | emonet_trajectories.csv | frame/sequence | 10; none |
| emonet_embeds | emonet_embeds.npz | turn_embedding/pool | 512; mean_std |
| ear_blinking | ear_blinking.csv | frame/sequence | 4; none |
| mediapipe_bodypose | mediapipe_bodypose.csv | frame/sequence | 165; none |
| linguistic_turnlevel | linguistic_turnlevel.csv | turn/pool | 41; mean_std |
| linguistic_videolevel | linguistic_videolevel.csv | video/identity | 39; none |
| mpnet_sentence_embeds | mpnet_sentence_embeds.npz | turn_embedding/pool | 768; mean_std |

No loader exists at `splits/care_features.py`; return type, keys, normalization, padding, and masking are unverified.

## 6. CARE representative feature schemas

Representative directory: `CONTROL/ASTHMA/PROFILE_053/VIDEO_000/`. CARE contains 2,031 files for every registered name, matching 2,031 videos. CSVs are numeric tables with ID/timestamp metadata; static families are one row per turn and dynamic families preserve frame/segment rows. Emotion trajectories contain per-turn `arousal,dominance,valence`.

NPZ headers: XLS-R and TRILLsson contain `mean`/`std` float32 `(1,1024)` plus `turn_ids` int64 `(1,)`; the 2,048 registry dimension is mean/std concatenation. Emotion embeddings contain `embed` float32 `(1,1024)` plus int64 `turn_ids`. No normalization/padding is encoded. All requested audio families are present under exact registry names; there are no separate arousal/valence/dominance files.

## 7. CARE extractor inventory

`extract_diarizen.py` produces RTTM (torch/DiariZen). `extract_egemaps.py` produces eGeMAPS CSVs (librosa/opensmile/pandas); `extract_disvoice.py` produces DisVoice CSVs (librosa/soundfile/DisVoice/pandas); `extract_wav2vec.py` produces XLS-R NPZ (torch/torchaudio); `extract_trillsson.py` produces TRILLsson NPZ (TensorFlow Hub `google/trillsson4/1`); and `extract_audeering_emotionmsp.py` produces emotion outputs (transformers `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`). Linguistic/video families produce transcript/MPNet, OpenFace/EmoNet/EAR/body-pose outputs.

Scripts accept corpus/output roots, write `<group>/<subgroup>/PROFILE_<id>/VIDEO_<id>`, and assume RTTM plus participant-turn selection. They are outside Chimera and plausibly adaptable independently, but WSM paths, source task/subgroup, segment naming, and turn semantics must be parameterized. No heavyweight script was executed.

## 8. WSM feature coverage gap

WSM `extracted/` has 1,776 `transcript.txt` files at `extracted/<label>/<subgroup>/PROFILE_<id>/VIDEO_<segment>/`. No CARE-registered CSV/NPZ name occurs; manifest rows report zero feature files and list all 19 families as missing.

| feature | CARE reference | WSM | schema match | action |
|---|---|---|---|---|
| all 19 registered families | present, 2,031 each | absent | N/A | approved extraction/parity task |
| transcript.txt | source evidence | 1,776 | not a CARE feature | retain as source input |

## 9. Repository KEEP / ADAPT / LEGACY / DELETE inventory

**KEEP candidates:** path-independent masked-loss mechanics in `src/common/loss/`, generic optimizer/callback utilities, and tensor/mask helpers after contract tests. **ADAPT:** `src/common/data/wsm_manifest.py`, `src/common/utils/segment_index.py`, all audio/fusion/text/video DataModules and feature/cache helpers; they hard-code WSM roots, old layouts, or 768/512 dimensions. **LEGACY:** `src/audio/models`, `src/fusion/models`, `src/video/models`, `src/text/models`, and pseudo-label/reliability/MTL callbacks/losses; future destination `src/legacy/wsm/` and `configs/legacy_wsm/`. **DELETE candidates after review:** `configs/wsm_mm_pd_dep_v1/`, generated caches/egg-info, and historical evidence helpers. Nothing was moved/deleted.

`src/chimera_plugin.py` exposes WSM callbacks, losses, DataModules, models, features, and optimizers. Packaging remains `wsm` in `pyproject.toml`/egg-info; entry point is `wsm = chimera_plugin:register`. No meaningful CAREWSM data-layer test exists.

## 10. Proposed active CAREWSM layout

Use `src/carewsm/{data,features,models,evaluation}`, `src/legacy/wsm/`, `scripts/extract/`, `configs/carewsm/`, `tests/carewsm/`, and `docs/audits/`. Rename the active package/plugin only in an authorized cleanup; this is not a move instruction.

## 11. DiariZen benchmark design

Select 3–5 WSM recordings spanning source task, control/positive organization, duration, and multi-turn structure. Record audio duration, wall-clock, real-time factor, speaker/turn counts, total turn duration, and respondent-vs-interviewer sanity. Accept if usable and within budget; narrow-optimize if sanity passes but runtime is excessive; drop if selection is unreliable. Do not run now.

## 12. Owner-assumption deviations

CARE `videos.csv`: missing. CARE `care_features.py`: missing. CARE labels: complete three-way `label_id`, not explicit two-column masks. WSM labels: explicit empty per-task values. WSM parity: 0/19 families present. Grouping/fixed folds: PASS for both manifests.

## 13. Recommended next atomic task

Manager review, then separately authorize cleanup/data-contract implementation. First preserve fixed assignments and add schema/fingerprint validation; do not extract until the loader gap and WSM mask contract are accepted.

## 14. Reproducibility / commands

Read-only commands included `git rev-parse`, `git status`, `find`, `head`, `cat`, standard-library Python CSV/JSON cardinality and overlap checks, `sed`/`rg` source inventory, `unzip -l`, and a `zipfile`/NPY-header parser. The default interpreter lacked pandas/numpy, so no dependency installation was performed. No CARE/WSM path was written and no Test metric was computed.
