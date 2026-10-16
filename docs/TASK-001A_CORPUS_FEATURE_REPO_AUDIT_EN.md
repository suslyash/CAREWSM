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

## FAIL-01 corrective evidence supplement

This supplement is part of sections 2–14 and corrects/expands the first-pass report without changing its 14 numbered section titles.

### Sections 2–3: exact typing, nulls, duplicates, and fold evidence

Loading method: Python `csv.DictReader` with `encoding='utf-8-sig'`; every returned field is a Python `str`. The audit separately applied `int(x)` then `float(x)` to non-empty values. Thus `int-like`/`float-like` below are inferred semantics, not native pandas dtypes; empty strings are nullable values. CARE has eight columns in every split/fold: `sample_id,label_id,profile_id,subgroup,video_id,video_dir,split,cv_fold`. WSM has 34 columns, exactly: `sample_id,profile_id,label_id,subgroup,video_id,dataset,task,diagnosis,label,y_depression,y_parkinson,participant_id,source_video_id,source_group_id,segment_id,segment_file,split,stratum,cv_fold,source_group,topic_group,storage_type,video_path,feature_dir,archive_relative_path,tar_video_prefix,n_feature_files,available_modalities,missing_modalities,source_csv_path,source_row,csv_video_id,csv_diagnosis,csv_segment_file`.

CARE inferred types: all columns are string-like except populated `cv_fold` int-like; holdout `cv_fold` is empty. WSM inferred int-like columns are `diagnosis,label,cv_fold,n_feature_files,source_row,csv_diagnosis`; `y_depression` and `y_parkinson` are int-like plus empty; all other columns are string-like, with `topic_group,archive_relative_path,tar_video_prefix,available_modalities` empty in the observed manifests.

Exact duplicate-row counts by key `(sample_id, profile_id, sample_id+profile_id)` are CARE train `0/1240/0`, dev `0/233/0`, test `0/277/0`; WSM train `0/5595/0`, dev `0/836/0`, test `0/1263/0`. The duplicate profile rows are expected repeated videos/segments within participants; sample and combined keys are unique.

CARE exact split/cv values: train `split=train`, `cv_fold={0:305,1:288,2:315,3:265,4:264}`; dev `split=dev`, empty cv 276; test `split=test`, empty cv 318. WSM: train `split=train`, cv `{0:1227,1:1227,2:1227,3:1226,4:1226}`; dev/test have empty cv for 936/1364 rows.

Explicit split-pair overlap counts are CARE and WSM, respectively: train/dev sample/profile `0/0`, train/test `0/0`, dev/test `0/0`. Fold cardinalities are CARE fold 0 `(train rows/samples/profiles=1132/1132/157, val=305/305/40)`, fold 1 `1149/1149/157;288/288/40`, fold 2 `1122/1122/157;315/315/40`, fold 3 `1172/1172/158;265/265/39`, fold 4 `1173/1173/159;264/264/38`; WSM fold 0 `4906/4906/431;1227/1227/107`, fold 1 `4906/4906/430;1227/1227/108`, fold 2 `4906/4906/428;1227/1227/110`, fold 3 `4907/4907/429;1226/1226/109`, fold 4 `4907/4907/434;1226/1226/104`. Every fold train/val sample overlap and profile overlap is `0/0`, and both sides are subsets of canonical train.

Cross-fold validation counters were recomputed, not inferred: CARE union 1,437, missing 0, duplicate memberships 0, min/max membership 1/1; WSM union 6,133, missing 0, duplicate memberships 0, min/max 1/1.

CARE `test -e /media/maxim/Databases/CARE/splits/videos.csv` and `test -e /media/maxim/Databases/CARE/splits/care_features.py` both returned exit 1. `find /media/maxim/Databases/CARE/splits -maxdepth 2 -type f` found no equivalent loader or `videos.csv`; only split/fold CSVs, `features.json`, and summary metadata. The concatenated three split files have 2,031 unique samples and agree with `summary.json:n_videos=2031`; this is an observed row-inventory equivalence, not a fabricated canonical file.

WSM observed task/source values are `task={depression,parkinson}`, `subgroup={DEPRESSION,PARKINSON}`, `source_group={depression,parkinson}`, `storage_type={files}`, `stratum={CONTROL,DEPRESSION,PARKINSON}`. The current rows have only `(y_depression,y_parkinson)=(0,empty),(1,empty),(empty,0),(empty,1)`; this is current manifest availability, not a claim of intrinsic disease mutual exclusivity. Empty is unknown/unavailable, never negative. The 318 ignored duplicate-profile count is from the trusted existing artifact `/media/maxim/Databases/WSM_NEW/splits/summary.json` (`ignored_segments`, `ignored_by_split`, `ignored_by_reason`), not a new deduplication operation.

### Sections 5–6: exact registry keys and all 19 schemas

The registry source count is 19. Every entry has exact fields `modality,file,format,level,video_policy,families,interpretable,original_feature_dim,metadata`; pooled entries additionally have `aggregation`, and embedding entries additionally have `embedding_keys`/`embedding_aggregation`. `openface`, `emonet_trajectories`, `ear_blinking`, and `mediapipe_bodypose` additionally have `quality_column`; linguistic turn-level additionally has `aggregation_note`.

Bounded sampling policy: inspect exactly `CARE/extracted/CONTROL/ASTHMA/PROFILE_053/VIDEO_000/<registered file>` once per registry entry, read CSV header/rows or NPZ ZIP NPY headers, and separately count registered filenames under CARE `extracted/`. CARE has 2,031 occurrences of each of the 19 registered filenames. Variants outside this directory are not established.

| registry entry | sampled schema: storage, row/column count or keys, metadata, dtype/shape |
|---|---|
| `egemaps_func` | CSV 1x91; metadata `turn_id,turn_start_sec,turn_end_sec`; 88 numeric feature columns, float-like; turn-level |
| `egemaps_lld` | CSV 10,364x30; metadata `turn_id,turn_start_sec,turn_end_sec,frame_start_sec,frame_end_sec`; 25 numeric columns, float-like; frame-in-turn |
| `disvoice_prosody_static` | CSV 1x106; 3 turn metadata + 103 numeric float-like columns; turn-level |
| `disvoice_prosody_dynamic` | CSV 279x18; 5 turn/frame metadata + 13 numeric float-like columns; segment-in-turn |
| `disvoice_phonation_static` | CSV 1x31; 3 turn metadata + 28 numeric float-like columns; turn-level |
| `disvoice_phonation_dynamic` | CSV 2,438x12; 5 turn/frame metadata + 7 numeric float-like columns; frame-in-turn |
| `disvoice_articulation_static` | CSV 1x491; 3 turn metadata + 488 numeric float-like columns; turn-level |
| `wav2vec_xlsr300m` | NPZ `mean,std,turn_ids`; mean/std `<f4 (1,1024)`, ids `<i8 (1,)`; turn embedding, pooled mean/std |
| `trillsson4_1` | NPZ `mean,std,turn_ids`; mean/std `<f4 (1,1024)`, ids `<i8 (1,)`; turn embedding, pooled mean/std |
| `emotionmsp_trajectories` | CSV 1x6; 3 turn metadata + exact `arousal,dominance,valence`; float-like; turn-level |
| `emotionmsp_embeds` | NPZ `embed,turn_ids`; embed `<f4 (1,1024)`, ids `<i8 (1,)`; turn-indexed embedding |
| `openface` | CSV 2,910x714; 4 metadata + 710 numeric-like columns; frame/timestamp/confidence/success; frame sequence |
| `emonet_trajectories` | CSV 2,910x13; 3 metadata + 10 numeric-like columns; frame/timestamp/success; frame sequence |
| `emonet_embeds` | NPZ `mean,std,valid_ratio,turn_ids`; mean/std `<f4 (1,256)`, valid ratio `<f8 (1,)`, ids `<i8 (1,)`; pooled turn embedding |
| `ear_blinking` | CSV 2,910x7; frame/timestamp/success + `blinking,ear_left,ear_right,ear_avg`; sequence; success is string-like in sample |
| `mediapipe_bodypose` | CSV 2,910x168; frame/timestamp/success + 165 numeric-like columns; frame sequence |
| `linguistic_turnlevel` | CSV 1x42; turn metadata `turn_id,start_sec,end_sec` plus 39 numeric-like observed columns; turn summary. Registry width is 41, so this sampled file has an unresolved width discrepancy |
| `linguistic_videolevel` | CSV 1x39; no metadata; 39 numeric-like columns; video identity-level |
| `mpnet_sentence_embeds` | NPZ `embed,turn_ids`; embed `<f4 (1,768)`, ids `<i4 (1,)`; turn-indexed embedding |

NPZ types/shapes were read from NPY headers inside ZIP containers with `zipfile`/`struct`, not by loading arrays with NumPy. CSV types are explicit first-sample inference, not native pandas dtypes. No sampled file exposes normalization/padding behavior. CARE voice-quality/pausing are registered subfamilies/columns rather than standalone files: eGeMAPS includes `Voice quality` and `Voice timing/pauses`; DisVoice includes timing/pausing and perturbation-related fields. Emotion MSP stores arousal/dominance/valence trajectories and latent embeddings as named above.

### Sections 7–8: script and parity details

Relevant CARE producer scripts and exact source-defined interfaces are: `audio_descriptors/extract_egemaps.py` (`--audio-dir --output-dir --is-functional`, eGeMAPS CSV; librosa/opensmile/pandas; RTTM and `sample_id.split('__')`); `extract_disvoice.py` (`--audio-dir --output-dir --is-static`, three DisVoice CSVs; librosa/soundfile/DisVoice/pandas; same RTTM/path assumptions); `extract_wav2vec.py` (`--audio-dir --output-dir`, XLS-R NPZ; torch/torchaudio/numpy); `extract_trillsson.py` (`--audio-dir --output-dir`, TRILLsson NPZ; TensorFlow Hub URL `https://tfhub.dev/google/trillsson4/1`); `extract_audeering_emotionmsp.py` (`--audio-dir --output-dir`, emotion NPZ+CSV; transformers model `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`, hard-coded `cuda:1`); `extract_diarizen.py` (`--audio-dir --output-dir`, RTTM; diarizen/torch); `linguistic_descriptors/extract_linguistics.py` (`--transcript-dir --rttm-dir --output-dir`, two linguistic CSVs; TextGrid/NLTK/pandas); `extract_mpnet.py` (same three flags, MPNet NPZ; transformers/sentence-transformers); `video_descriptors/extract_openface.sh` (external OpenFace CSV); `extract_emonet.py` (`--video-dir --output-dir --checkpoint --mean-face-path`, EmoNet NPZ; explicit `cuda:1`, local checkpoints); `extract_earblink.py` (`--video-dir --output-dir`, EAR CSV; OpenCV/MediaPipe); and `extract_bodypose.py` (`--video-dir --output-dir --checkpoint`, body-pose CSV; OpenCV/MediaPipe). `whisperx_transcript_alignment.py` is a supporting alignment script, not one of the 19 producers.

No absolute `/media/.../CARE` path was found in producer source. Coupling is through CARE `group__subgroup__profile__video` naming, positional path components, RTTM/participant-turn assumptions, local checkpoint defaults, and hard-coded CUDA devices. All are potentially adaptable outside Chimera after those assumptions are parameterized; no `--input-root/--output-root` claim is made beyond the flags listed.

WSM parity matching used the exact predicate `(Path('/media/maxim/Databases/WSM_NEW') / row['feature_dir'] / registry_file).exists()` for all 8,433 rows (6,133/936/1,364). It found 0 for every registry filename in every split; every row had `n_feature_files=0` and a missing-modality list containing all 19. A separate global filename inventory under `WSM_NEW/extracted` found exactly 1,776 files, all named `transcript.txt`. Representative path: `extracted/CONTROL/PARKINSON/PROFILE_-4QFjLjZio4/VIDEO_-4QFjLjZio4_001/transcript.txt`; it is plain text without a feature schema. This verifies global filename absence, not merely sampled absence.

The 19-row matrix is therefore: each registry name (`egemaps_func`, `egemaps_lld`, all five DisVoice entries, XLS-R, TRILLsson, both emotion entries, OpenFace, both EmoNet entries, EAR, body pose, both linguistic entries, and MPNet) has CARE reference `yes/2,031`, WSM `0`, schema match `N/A`, and action “approved CARE-compatible extraction plus schema validation for that exact file/level/aggregation contract.” The action is intentionally per-family even though the result is uniform.

### Sections 9–10: actionable repository inventory

`src/common/data/{wsm_manifest.py,wsm_speaker_map.py}`, `src/common/utils/segment_index.py`, and all audio/fusion/text/video DataModules are **ADAPT**: they hard-code WSM root/segment/cache semantics and cannot be the active CAREWSM data layer. `src/common/loss/wsm_masked_sparse_loss.py` and tensor/mask helpers are **KEEP candidates** mathematically but require a CAREWSM mask contract; WSM callback/pseudo-label/balance files are **ADAPT or LEGACY** because they encode old evaluation/research semantics. `src/audio/models`, `src/fusion/models`, `src/video/models`, and `src/text/models` are **LEGACY** comparator families, not active methods. `src/audio/features`, `src/common/features`, video cache features, and text cache features are **ADAPT** only after schema review.

All 157 config files are WSM-era; `configs/audio_experiments/` and `configs/wsm_mm_pd_dep_v1/` are **LEGACY/DELETE candidates after manager approval**, not active configs. Retained `scripts/common` include historical WSM manifest, pseudo-label, evaluation, and ablation orchestration; only bounded audit utilities are potential **ADAPT**. The three tests are `test_task010a_gradient_mtl.py` (mathematical gradient method), `test_task012a_postclosure_evaluation_guards.py`, and `test_task012b_shared_progress_test_correctness_audit.py` (historical evaluation guards); none is a CAREWSM data-contract test.

`src/chimera_plugin.py` imports 32 modules: five WSM callbacks, common WSM loss/optimizer, audio DataModule/model/loss, eight fusion models, three fusion DataModules, four fusion losses, two video models plus DataModule, and text feature/DataModule/model. Exhaustive decorator registration names are reproducible with `rg -n "@(?:MODELS|DATAMODULES|LOSSES|CALLBACKS|OPTIMIZERS)\\.register" src --glob '*.py'`; they include 13 model names, 7 DataModules, 6 losses, 6 callbacks, and 1 optimizer, all listed in the first-pass report’s plugin subsection. `pyproject.toml` and `src/wsm.egg-info/*` retain package name/description/entry point `wsm`; those are later packaging cleanup targets.

### Section 11 design limitation

No candidate recording was hard-coded because the corrective pass did not decode raw media. A deterministic later query is: `find /media/maxim/Databases/WSM_NEW/{depression,parkinson}/{train_labels,dev_labels,test_labels} -type f -path '*/segments/*.mp4' | sort`, then choose 3–5 paths spanning source task, control/positive manifest strata, duration if available cheaply, and multi-turn structure. Record audio duration, wall-clock, `RTF=wall_clock/audio_duration`, speaker count, turn count, total turn duration, and respondent-selection sanity. Accept/narrow-optimize/drop thresholds are owner-approved proposals, not observed facts.

### Section 12 owner-expectation table

| OWNER EXPECTATION | OBSERVED FACT | DEVIATION/PASS | NEXT CONTRACT IMPLICATION |
|---|---|---|---|
| both corpora provide `splits/videos.csv` | WSM yes; CARE existence check exit 1 | DEVIATION | explicitly decide whether CARE split union is canonical |
| CARE provides `splits/care_features.py` | existence check exit 1; no bounded alternative found | DEVIATION/UNVERIFIED | no loader keys/return semantics may be assumed |
| fixed five folds and participant/profile disjointness | all sample/profile and fold checks pass | PASS | preserve assignments |
| two independent labels | CARE complete three-way; WSM task masks, no both-observed row | PARTIAL | preserve task provenance and masks; no semantic extrapolation |
| all expected CARE audio families exist | all requested families registered; voice/pausing are subfamilies | PASS WITH LIMIT | encode actual names/categories |
| WSM feature parity exists | 0/19 filenames, 0/8,433 manifest matches | DEVIATION | separate extraction task |
| copied WSM code is active-compatible | WSM paths/protocols/registrations remain | DEVIATION | review classification before cleanup |

### Section 14 exact command/evidence record and limitations

Literal commands used include:

```bash
test -e /media/maxim/Databases/CARE/splits/videos.csv; echo $?
test -e /media/maxim/Databases/CARE/splits/care_features.py; echo $?
find /media/maxim/Databases/CARE/splits -maxdepth 2 -type f -printf '%P\n' | sort
find /media/maxim/Databases/WSM_NEW/extracted -type f | awk -F/ '{print $NF}' | sort | uniq -c
find /media/maxim/Databases/CARE/scripts -type f -not -path '*/__pycache__/*' -printf '%P\n' | sort
sed -n '1,55p' src/chimera_plugin.py
rg -n "@(?:MODELS|DATAMODULES|LOSSES|CALLBACKS|OPTIMIZERS)\.register" src --glob '*.py'
```

The split/fold/mask command was the standard-library snippet: `csv.DictReader(open(path,encoding='utf-8-sig'))`, sets of `sample_id`/`profile_id`, `collections.Counter` for duplicate keys and validation memberships, and `Path(root)/row['feature_dir']/registry_file` existence checks. The registry command was `json.load(open('/media/maxim/Databases/CARE/splits/features.json'))`, iterating `sources.items()` and printing exact keys. Schema commands read CSV headers/rows and NPZ ZIP NPY headers only. No heavy dependency was installed; default Python lacked pandas/numpy, so native library dtypes remain unverified.

Unverified/blocked claims are explicit: CARE loader return type, keys, normalization, padding, and masking are unavailable; native pandas dtypes are unavailable; feature variants outside the bounded representative directory are not established; WSM source `participant_id=profile_id` equivalence is only recorded in the existing summary artifact; DiariZen runtime/speaker outputs were not run; no audio-duration budget was observed; and Git cannot prove external data roots were unchanged. The audit commands were read-only and no mutation command targeted CARE/WSM.
