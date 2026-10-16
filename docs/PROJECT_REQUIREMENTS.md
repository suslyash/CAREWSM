# CAREWSM Project Requirements

## 1. Goal

CAREWSM studies multitask, multilabel recognition of Depression and Parkinson's disease across CARE and WSM, starting with unimodal audio and progressing to multimodal modelling.

The intended output is two independent binary targets. Unknown/unavailable labels are masked rather than converted to negatives. The exact corpus label columns and availability must be verified before coding against this intent.

## 2. Data roots and expected partition contract

- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`

Owner-provided expectations, to be verified in the first audit:

- each corpus has fixed `train`, `dev`, `test`;
- each has five CV folds inside TRAIN;
- grouping is participant/profile based and prevents overlap;
- split/fold assignments are not to be regenerated.

Expected files include:

- `splits/videos.csv`
- `splits/train.csv`, `dev.csv`, `test.csv`
- `splits/folds/fold_{0..4}_{train,val}.csv`

CARE additionally exposes `splits/features.json` and `splits/care_features.py`, which are part of the reference feature contract.

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

Exact filenames, model revisions, dependencies, shapes, dtypes, and aggregation rules are **audit outputs**, not assumptions.

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

## 8. Legacy reset

The repository began as a code copy of WSM. Copied implementation is reference material only until audited.

Previous WSM task history, result evidence, metrics, claims, final-method roles, pseudo-label thresholds, and experimental decisions do not carry into CAREWSM automatically.
