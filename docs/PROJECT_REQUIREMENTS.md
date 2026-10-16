# CAREWSM Project Requirements

## 1. Goal

CAREWSM studies multitask, multilabel recognition of Depression and Parkinson's disease across CARE and WSM, starting with unimodal audio and progressing to multimodal modelling.

The intended output is two independent binary targets. Unknown/unavailable labels are masked rather than converted to negatives. TASK-001A audited the source label columns and masks; the precise cross-corpus two-head encoding must still be deliberately specified before implementation.

## 2. Data roots and expected partition contract

- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`

Owner expectations **verified by TASK-001A** (observational audit): both corpora have fixed `train/dev/test` CSVs, five CV folds within TRAIN, and no sample/profile overlap across partitions or fold train/validation. The fixed assignments must not be regenerated.

Observed differences from the initial owner-provided expected file list:

- CARE has `splits/train.csv`, `dev.csv`, `test.csv`, five `splits/folds/fold_{0..4}_{train,val}.csv` pairs and `splits/features.json`, but **no `splits/videos.csv` and no `splits/care_features.py`** in the audited location. The observed union of CARE splits contains 2,031 unique samples. Loader return types, normalization/padding and masking remain unknown.
- WSM has `splits/videos.csv`, train/dev/test, five TRAIN CV folds, and **34-column** split manifests. Task-level `y_depression`/`y_parkinson` entries may be blank; blank means unknown and must remain masked. In the audited manifest, no sample has both task labels observed.
- CARE provides complete three-way `label_id` (`CONTROL`, `DEPRESSION`, `PARKINSON`), **not** two explicit target columns. Mapping to the intended two-head targets requires an explicit audited design; source-task and subgroup semantics must not be extrapolated.

These are observations, not permission to change split, label or evaluation policy. Source: `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`.

## 3. Feature parity

CARE is the reference for feature naming, storage, file format, keys, shapes, dtypes, aggregation, and loader semantics.

Target WSM layout:

`extracted/<LABEL>/<SUBGROUP>/PROFILE_<profile id>/VIDEO_<video name>/`

Before Chimera ML modelling, WSM must obtain CARE-compatible versions of all required audio features that exist in CARE:

- 16 kHz mono audio preprocessing;
- respondent speech segmentation/turn alignment, subject to the DiariZen gate below;
- eGeMAPS v2.0 frame-level LLD trajectories;
- eGeMAPS turn-level functional summaries;
- DisVoice articulation;
- DisVoice prosody;
- DisVoice phonation;
- CARE-represented voice-quality and pausing-behaviour features;
- XLS-R wav2vec 300M representations;
- TRILLsson 4.1 representations;
- XLS-R and TRILLsson aggregation at the same turn/utterance level as CARE;
- arousal trajectories;
- valence trajectories;
- dominance trajectories;
- corresponding emotion latent representations;
- deep representations stored in the same aggregated form provided by CARE.

Registered CARE features and representative schemas are documented in `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md` (19 registered families, including non-audio families). WSM has **0/19 CARE-registered feature filenames** across 8,433 audited manifest rows; 1,776 `transcript.txt` files are not equivalent to these registered features. The CARE `linguistic_turnlevel` registry declares `original_feature_dim=41` while the sampled CSV exposes 39 feature names/columns plus three metadata columns. This discrepancy is unresolved and **must not be silently normalized**. Loader behavior and across-sample schema variants remain unverified. No feature extraction is authorized by this document alone.

## 4. DiariZen gate

DiariZen is a small engineering gate, not an independent research programme.

A later task may benchmark it on a small representative WSM subset and record:

- wall-clock time;
- audio duration;
- real-time factor;
- detected speaker/turn summary;
- basic sanity of respondent-vs-interviewer selection where inspectable.

The manager/owner then chooses retain / optimize narrowly / drop. Do not let diarization block the project for an extended period.

## 5. Extraction versus training

- One-time feature extraction/preprocessing is outside Chimera ML.
- CARE extraction scripts under `/media/maxim/Databases/CARE/scripts` are reference implementations and must first be audited.
- WSM adaptations should live in project-controlled scripts rather than silently mutating CARE reference code.
- Training/evaluation pipelines after feature extraction use Chimera ML.

## 6. Selection and evaluation

- Five TRAIN CV folds: development and hyperparameter selection.
- DEV: final model/configuration selection.
- TEST: descriptive/reporting output, stored so it need not be recomputed; never a selection criterion.

For a completed run/configuration, prefer to evaluate TEST once for the DEV-selected checkpoint and persist metrics plus evaluator/config/checkpoint identity.

TEST must not trigger:

- checkpoint/threshold changes;
- architecture/hyperparameter changes;
- extra seeds/sweeps;
- ablation continuation;
- reruns motivated only by the Test value.

## 7. Reproducibility

Every significant experiment must trace to:

- corpus split/fold fingerprint;
- feature schema/cache fingerprint;
- resolved config;
- seed;
- source commit/dirty state;
- checkpoint identity;
- CV/DEV metrics used for decisions;
- TEST metrics as report-only evidence;
- run/artifact identifier.

No split, label, feature file, or result may be silently rewritten.

## 8. Repository layout and legacy/archive contract — latest owner decision

Active ChimeraML implementation must live in `src/audio/`, `src/fusion/`, `src/common/`, with `src/chimera_plugin.py` as the plugin entrypoint. One-time preprocessing/extraction/research scripts belong under root `scripts/`, documentation in root or `docs/`; no README outside those allowed locations. `src/carewsm/` is not part of the active architecture.

**Passive source archive:** old models and historical callbacks may live in *shallow, explicitly named* folders `src/legacy/audio_models/`, `fusion_models/`, `video_models/`, `text_models/`, `callbacks/`. After TASK-001D (PR #12), the passive archive is flattened and preserves **14 verified historical model implementation files** (not 15; prior management count was incorrect) plus two historical callback sources. The archive is excluded from installed packages and auto-registration. The callable `wsm_summary_callback` is **required in future training configs**, but its archived source must be adapted/registered by a separately authorized task before use; `wsm_segment_metrics_callback` is optional pending label-mask, metric and TEST-policy verification.

**Obsolete configurations:** all 157 old copied WSM-era tracked `configs/` were removed in TASK-001D (PR #12). Do not replace them with training configs or activate callbacks until training is separately authorized. Future approved ChimeraML training configs must declare callbacks `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `wsm_summary_callback` and loggers `mlflow_logger`, `console_file_logger`; native-vs-project implementations must be verified before config authoring. The owner does not want a test suite or tests executed for this cleanup. After archiving old WSM models, `pip install -e .` must not attempt to install/build their stale training dependencies. In TASK-001D, minimize direct package requirements to current active ChimeraML code (`chimera-ml>=0.2.4`), removing obsolete `mamba-ssm` and old Torch/audio/video pins from direct project dependencies; future requirements belong to specifically authorized model/extraction tasks. Verify editable packaging with bounded pip checks, without disabling build isolation or altering the user's existing Torch stack as a workaround. No data modification, extraction, DiariZen, evaluation or training is authorized by this source-layout requirement.

## 9. Legacy reset

The repository began as a code copy of WSM. TASK-001A classified copied components as KEEP/ADAPT/LEGACY/DELETE candidates; classification is not automatic activation of old WSM methods. Subsequent changes require a separately reviewed atomic task.

Previous WSM task history, result evidence, metrics, claims, final-method roles, pseudo-label thresholds, and experimental decisions do not carry into CAREWSM automatically.
