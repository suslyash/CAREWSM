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

**Latest owner scope — audio extraction + audio/multimodal modelling, NO nonaudio extraction:** CARE's registry contains 19 families, but **our extraction responsibility is only the 11 audio families**: seven eGeMAPS/DisVoice files (TASK-003A) and four XLS-R/TRILLsson/emotion MSP files (TASK-003B). Other teams provide **five video and three linguistic** families; never write, implement, or schedule their extraction here. We will **use** those externally supplied nonaudio representations in later multimodal fusion and check input availability, compatibility and masking, but this is distinct from producing features. See `docs/FEATURE_PARITY_TARGET_EN.md`. Preserve the WSM's observed **canonical `extracted/<STRATUM>/<SUBGROUP>/PROFILE_<id>/VIDEO_<segment>/` output structure** and CARE audio producer/numeric/CSV/NPZ/turn semantics. TASK-003A later observed that WSM manifest `feature_dir` points instead to a **legacy `*/train_labels/*/features/*` source path**; it must not be treated as or silently changed into the canonical output path. Resolve output path using validated manifest stratum/subgroup/profile/video identifiers, corroborate with existing `extracted/` transcript paths, preserve original manifest values and refuse ambiguous/unsafe destinations. Owner authorizes only a three-TRAIN-sample WSM `extracted/` smoke for the seven descriptors now; any additional audio families or corpus-wide writes need separate task authorization. Do not overwrite existing valid features, transcripts, raw media or manifests. TASK-002C listening is deferred due to SSH and participant-speaker selection must remain provisional. Large dependencies and model weights are downloaded only by the owner, after Codex supplies verified commands and traffic estimates.

**Owner resolved CARE numerical replay:** The original CARE audio is unavailable; therefore exact re-extraction and elementwise CARE numerical comparison are **not possible**. Never reconstruct, fetch or substitute audio, and do not block other WSM audio feature work on this unfulfillable gate. TASK-003A established bounded WSM schema/file/numeric-integrity compatibility for seven descriptor families on three TRAIN samples (21 CSVs), but **does not prove numeric equivalence with CARE's old CSV values**. Mark numeric CARE parity permanently **UNVERIFIED** unless authentic original media becomes available through a separately approved future task. Owner authorizes TASK-003B four deep-audio families on the same three TRAIN samples with verified offline cached models. See `docs/NEXT_TASK_EN.md`.

## 3. Feature parity

CARE is the reference for feature naming, storage, file format, keys, shapes, dtypes, aggregation, and loader semantics.

Target WSM layout:

`extracted/<LABEL>/<SUBGROUP>/PROFILE_<profile id>/VIDEO_<video name>/`

**Clarification after TASK-003A:** WSM `feature_dir` is not guaranteed to point to this target: observed TRAIN values start with `depression/train_labels/` or `parkinson/train_labels/` and end in `/features/<segment_id>`. Treat it as legacy metadata; derive and independently verify the canonical output path from manifest columns and existing `WSM_NEW/extracted/` layout before writing. Do not edit `feature_dir` or the split CSVs.

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

## 4. User-controlled downloads and traffic

For all subsequent stages, potentially large installs/downloads (notably CUDA/PyTorch wheels, feature extractors, model weights, datasets, caches and their transitive dependencies) are **owner-executed only**. Codex must present verified exact commands, official sources, package/model versions and revisions, expected size/uncertainty, cache paths and post-install verification **before** any such download. If size is unknown or the command can download recursively, explicitly flag the risk. No implicit `from_pretrained` or package-manager network fetch without specific owner approval. Existing cached weights should be reused. The owner-approved existing `.venv-diarizen` exception is permitted where the DiariZen stack genuinely conflicts with CAREWSM's primary `.venv`; Codex may not proliferate new environments by default. Routine local work and Git metadata fetch are unaffected.

## 4. DiariZen gate

DiariZen is a small engineering gate, not an independent research programme.

A later task may benchmark it on a small representative WSM subset and record:

- wall-clock time;
- audio duration;
- real-time factor;
- detected speaker/turn summary;
- basic sanity of respondent-vs-interviewer selection where inspectable.

The manager/owner then chooses retain / optimize narrowly / drop. Do not let diarization block the project for an extended period.

**TASK-002B owner's latest execution split:** Codex writes a verified numbered installation/weight-caching command guide and an unexecuted reusable `scripts/audio/cache_diarizen_weights.py` downloader. The owner personally runs all package installations and weight downloads in the existing project `.venv`; Codex must not provision or invoke inference during this preparatory phase. Only after the owner's setup handoff and a later manager approval may the bounded DiariZen GPU TRAIN gate run. The earlier general authorization to provision packages is OWNER-only at this stage.

**TASK-002B GPU requirement:** After a system reboot corrected the NVIDIA NVML driver/library mismatch, the owner requires GPU/CUDA PyTorch and rejects CPU-only wheels. Verify functioning `nvidia-smi` and `torch.cuda.is_available()` in the existing `.venv`, repair any accidentally introduced CPU-only `torch`/`torchaudio` by selecting verified official compatible CUDA wheels, and do not use a CPU fallback. If still incompatible, document the blocker; no new environments or dataset writes.

**TASK-002B owner provisioning approval:** Install all needed upstream DiariZen inference libraries (including `tqdm`, `torchaudio`, a compatible `torch` stack, and other required dependencies) and download/cache authentic pretrained DiariZen weights once to unblock a bounded WSM TRAIN feasibility run, **using the existing CAREWSM `.venv` only**, with no additional virtual environments; model/dependency caches stay outside the Git repository and both dataset roots. Compatible package adjustments inside the existing `.venv` are authorized for this task, while `pyproject.toml` stays unchanged. If incompatibility is irreconcilable, report a blocker rather than creating another environment. Model weights and cache files must not be committed; provenance/revisions, install versions, model license/access terms and timings must be documented. This exception never authorizes TEST access, large-corpus processing, training, feature extraction, dataset writes or changing CAREWSM's declared project dependencies.

## 5. Extraction versus training

- One-time feature extraction/preprocessing is outside Chimera ML.
- CARE extraction scripts under `/media/maxim/Databases/CARE/scripts` are reference implementations and must first be audited.
- WSM audio adaptations, preprocessing/DiariZen gates and all one-off audio feature extraction/aggregation scripts must live in **`scripts/audio/`**, not in `src/` and not directly in `scripts/`. They must be independent of ChimeraML (no framework imports, plugin/registry/trainer/DataModule coupling or training configs); do not mutate CARE reference code. Training/evaluation pipelines subsequently consume extracted features through ChimeraML `src/audio/` components.
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
