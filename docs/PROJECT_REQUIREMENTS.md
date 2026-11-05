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

## Primary joint training protocol

The main CAREWSM programme trains on CARE and WSM together from the first modelling experiment.

For CV fold `k`:

- training data = CARE `fold_k_train` ∪ WSM `fold_k_train`;
- validation data = CARE `fold_k_val` ∪ WSM `fold_k_val`;
- preprocessing, PCA/scalers and task class weights are fitted only on this joint training portion;
- checkpoint/model/feature/loss selection uses validation evidence only;
- the selected fold checkpoint is evaluated on CARE TEST and WSM TEST for statistics.

CARE-only and WSM-only training are not the primary baseline path. They may be added later only as explicit ablations.

The fixed DEV split remains a development resource, but its exact role is deferred until the fold-based joint pipeline is implemented; it must never be silently mixed into TRAIN.

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

## Required validation and fold-level TEST tables

For each task, fold, and configuration report the same six metrics on **validation** and on **TEST**:

- Macro F1;
- Weighted F1;
- Macro Recall;
- Weighted Recall;
- Patient Recall;
- Healthy Recall.

For validation, report the CARE portion, WSM portion and metric-wise Average whenever both corpora have evaluable labels.

Every CV fold must also be evaluated on both fixed TEST sets:

- CARE TEST;
- WSM TEST.

For every metric also report:

`Average = (CARE metric + WSM metric) / 2`.

This Average is a simple corpus-level arithmetic mean, not a metric recomputed from pooled CARE+WSM predictions.

For binary classification:

- Patient Recall = recall of the positive/patient class;
- Healthy Recall = recall of the negative/healthy class;
- Macro Recall = arithmetic mean of the two class recalls (UAR).

All metrics are computed only on rows where the corresponding disease label is observed.

The intended reporting format has one row per Configuration × Task × Fold and three six-metric blocks: CARE, WSM, Average.

The **validation table is the selection table**.

The **TEST table is statistics only** and must never drive model/feature/loss/hyperparameter decisions.

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

The **primary** programme is joint CARE + WSM training from the outset.

Deferred secondary studies, to be run only after the main joint model is established:

1. CARE-only / WSM-only ablations;
2. direct transfer CARE -> WSM and WSM -> CARE;
3. external D-Vlog / YouTubePD evaluation;
4. small-target adaptation.

These secondary protocols must not complicate or delay the primary joint feature/model search.

## Engineering

All active Chimera-ML code belongs under `src/audio/`, organized by responsibility.

Keep implementation simple, explicit, testable and independent of obsolete WSM experiment logic.


## Simplified active data layer

The active training data layer has exactly two files: `src/audio/data/carewsm_audio_dataset.py` and `src/audio/data/carewsm_audio_datamodule.py`. The Dataset directly loads native artifacts and supports ordered concatenation only for static `T=1` features. The DataModule constructs canonical CARE/WSM fold populations and separate evaluation populations.

The availability manifest is a validity/filtering source, not a complete path registry. It may not contain `label_id`, `subgroup`, or every source-path field. The DataModule must obtain those fields by joining each sample with authoritative CARE/WSM split metadata. It must preserve fixed split membership, profile isolation, labels, and `VALID` feature filtering. No `JointDataModule`, `NativeDataset`, `Sample`, `View`, `NativeFeature`, or replacement adapter hierarchy belongs in the active pipeline.

Required evaluation names are `train`, `val_care`, `val_wsm`, `test_care`, and `test_wsm`; joint training does not justify collapsing corpus-specific validation or TEST outputs.

## Logging and selection

Epoch INFO output in console and `train.log` includes native Chimera sample counts, loss, Macro Recall, Patient Recall, and Healthy Recall. All six approved metrics remain calculated: Macro F1, Weighted F1, Macro Recall, Weighted Recall, Patient Recall, and Healthy Recall. Label masks and support counts remain internal and are not emitted in new epoch logs or summaries. The `[carewsm_combined epoch N]` record reports equal-weight VAL and TEST averages; additional metrics also belong in `summary.txt` and optionally MLflow.

The canonical selection monitor is `val_mean_corpora_tasks/macro_recall`, the unweighted arithmetic mean of CARE Depression Macro Recall, CARE Parkinson Macro Recall, WSM Depression Macro Recall, and WSM Parkinson Macro Recall. The deprecated `dev_mean_macro_recall` alias is numerically identical. `val_mean_corpora/<task>/<metric>` and `test_mean_corpora/<task>/<metric>` are unweighted CARE/WSM means for one task; `val_mean_corpora_tasks/<metric>` and `test_mean_corpora_tasks/<metric>` are unweighted means across the four corpus/task combinations. The combined monitor calculates these canonical averages for all six metrics. Missing components are reported as unavailable; they are not zero-filled or renormalized. Checkpoint and early stopping use only `val_mean_corpora_tasks/macro_recall`; TEST remains descriptive only.

The historical 110 canonical fold-runs used `joint/average_macro_f1`. They are preserved historical results and must not be reinterpreted as selected by `dev_mean_macro_recall`.

## Active refactor sequence

REFACTOR-000 through REFACTOR-007 are complete and merged. LAUNCH-001 completed one successful real CUDA training epoch. FINAL-INTEGRATION-001 is complete and merged through PR #40 with an approved PASS result and two non-blocking cleanup findings. The active audio-only Chimera pipeline and canonical 110-config matrix are integration-ready. No real training run after REFACTOR-007 has been performed. TASK-011 remains PAUSED. No new experiments, cleanup implementation, or research tasks are authorized by the completed audit.


## Deferred repository cleanup

REFACTOR-007 removed the obsolete `configs/audio_feature_selection/joint_native_fold0.yaml` and `configs/audio_feature_selection/task004_models.yaml`; the remaining obsolete smoke configuration, `scripts/runs/`, and `smoke_*` scripts may be removed only with reference updates.
