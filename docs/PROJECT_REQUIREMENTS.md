# CAREWSM Project Requirements

## Goal

Build an audio-only system for Depression and Parkinson recognition using CARE and WSM, with strong emphasis on temporal modelling and controlled cross-corpus generalization.

Multimodality is explicitly deferred.

## Corpora and structure

### CARE

Root:

`/media/maxim/Databases/CARE/`

Expected structure:

- `splits/videos.csv` — complete video registry with `sample_id`, `profile_id`, `video_dir`, `split`, `cv_fold`;
- `splits/train.csv`, `dev.csv`, `test.csv`;
- `splits/folds/fold_{0..4}_{train,val}.csv`;
- `splits/features.json`;
- `splits/care_features.py`;
- `extracted/<LABEL>/<SUBGROUP>/PROFILE_<profile_id>/VIDEO_<video_name>/`.

### WSM

Root:

`/media/maxim/Databases/WSM_NEW/`

Expected structure:

- `splits/videos.csv`;
- `splits/train.csv`, `dev.csv`, `test.csv`;
- `splits/folds/fold_{0..4}_{train,val}.csv`;
- raw segmented video trees under `depression/` and `parkinson/`;
- `extracted/<LABEL>/<SUBGROUP>/PROFILE_<profile_id>/VIDEO_<video_name>/`.

Both corpora use fixed train/dev/test splits and five CV folds inside TRAIN, grouped by participant/profile to prevent leakage. Codex must verify the live files rather than assume them.

## Audio feature inputs

CARE extracted video folders may contain acoustic CSV/NPZ artifacts including:

- eGeMAPS v2.0 frame-level LLD trajectories;
- eGeMAPS turn-level functionals;
- DisVoice articulation;
- DisVoice prosody;
- DisVoice phonation;
- voice-quality/pausing-related descriptors;
- XLS-R wav2vec 300M aggregated turn representations;
- TRILLsson 4.1 aggregated turn representations;
- arousal trajectories;
- valence trajectories;
- dominance trajectories;
- latent emotion representations.

The owner reports **11 intended audio feature families have been extracted for the current programme**. The exact operational family registry, file names, keys, shapes, timestamps and coverage must be discovered from the live extraction outputs/code during TASK-001.

All source audio preprocessing used for extraction is 16 kHz mono.

Deep representations in the published CARE corpus are available as aggregated embeddings rather than reconstructable raw hidden-state sequences.

## Feature failure policy

Some feature extraction failed for some samples.

Do not globally delete such samples.

Create deterministic sample × feature-family validity metadata. A run requiring set S uses samples for which all features in S are valid, unless a later experiment explicitly studies missing-feature handling.

Every experiment must report retained sample counts/coverage.

## Classifier structure

The intended structure is:

`shared/common feature representation`

followed by separate branches:

`Depression-specific feature layer -> Depression binary head`

`Parkinson-specific feature layer -> Parkinson binary head`.

The shared trunk may be one of the selected temporal architectures. The task-specific feature layers are mandatory experiment components unless a specific ablation removes them.

## Labels

Two independent binary targets:

- Depression;
- Parkinson.

Missing labels are masked, never interpreted as negative.

## Loss programme

Compare:

1. standard `BCEWithLogitsLoss`;
2. weighted BCE with `pos_weight = N_negative / N_positive`, computed independently per task using known labels from TRAIN only.

Each task loss is averaged only over examples where that task label is observed.

Base equal combination:

`L = 0.5 * L_depression + 0.5 * L_parkinson`.

Compare multitask optimization/balancing methods:

- Equal;
- PCGrad;
- CAGrad;
- STCH;
- PAGB.

Also compare separate single-task models.

## Required metrics

Base metrics:

- F1;
- weighted F1;
- UAR;
- weighted average recall.

Metrics must be reported per task and according to the experiment's corpus/transfer protocol.

## Mantis temporalization

For each video and each usable feature block, derive a sequence of frozen Mantis embeddings using:

- 1.0-second windows;
- 0.5-second hop;
- no global averaging of the full video to one vector.

Persist:

- embedding array `[num_windows, embedding_dim]`;
- window start/end times and original order;
- validity mask;
- valid-observation fraction/confidence per window;
- corpus/profile/video/block IDs;
- split;
- labels and missing-label masks;
- Mantis checkpoint;
- selected layer;
- pooling method;
- source-feature list;
- preprocessing parameters.

Scaler and PCA, if used, are stored separately and fitted only from TRAIN data of the current split/fold.

Do not duplicate the frozen Mantis model for every video.

Exact checkpoint/layer/pooling and family-specific window construction are frozen by a dedicated post-audit task.

## Cross-corpus programme

Required experiment families:

1. **Within corpus**
   - CARE alone;
   - WSM alone.

2. **Direct transfer**
   - CARE -> WSM;
   - WSM -> CARE;
   - no fitting/tuning on the target corpus.

3. **External test**
   - Depression models on D-Vlog;
   - Parkinson models on YouTubePD.

4. **Joint training**
   - CARE + WSM;
   - evaluate separately on each corpus and on external sets.

5. **Adaptation**
   - fine-tune/adapt using a small labeled subset of target TRAIN;
   - adaptation fractions/protocol must be frozen before running the study.

External dataset locations and their fixed evaluation protocol must be supplied/verified before those tasks begin.

## Engineering

All active Chimera-ML code belongs under `src/audio/`, organized by responsibility.

Keep implementation simple, explicit, testable and independent of obsolete WSM experiment logic.
