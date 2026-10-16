# Initial WSM Project Structure

## 1. Audit Snapshot

- Workspace: /media/maxim/Programs/Projects/WSM.
- Branch/commit at initial audit: main, c34c2ba (Initial commit).
- Installed chimera-ml: 0.3.0; pyproject requirement: chimera-ml >=0.2.4.
- Python: >=3.12.
- The worktree was clean before project documentation was created.
- No tests were present; unit tests are not mandatory.

This is a historical initial-state audit. Current execution status belongs in [PROGRESS_EN.md](PROGRESS_EN.md). The audit used source, configs under logs, summaries, MLflow artifacts, and the best audio run's reproducibility snapshot.

## 2. Initial Layout

    .
    ├── configs/audio_experiments/
    ├── docs/
    │   ├── 2402.19078v3.pdf
    │   ├── 2501.10945v3.pdf
    │   ├── BDCC-10-00089.pdf
    │   ├── Ryumin_EMNLP.pdf
    │   └── SOTA_REVIEW.md (archival original; active compact English version: SOTA_REVIEW_EN.md)
    ├── scripts/
    │   ├── collect_summary.py
    │   ├── extract_segment_wavs.py
    │   └── run_audio_experiments.sh
    ├── src/
    │   ├── chimera_plugin.py
    │   ├── audio/{data,features,loss,models}/
    │   └── common/{callbacks,features,utils}/
    ├── logs/
    ├── mlruns/
    └── pyproject.toml

At audit time there were no working src/video, src/text, src/description, or src/fusion packages.

## 3. Plugin and Registry

pyproject.toml declares:

    [project.entry-points."chimera_ml.plugins"]
    wsm = "chimera_plugin:register"

Initial project keys:

| Registry | Key | Implementation | Initial state |
|---|---|---|---|
| DATAMODULES | wsm_audio_segment_datamodule | WSMAudioSegmentDataModule | Registered |
| MODELS | audio_mamba_segment_model | AudioMambaSegmentModel | Provider import failed |
| LOSSES | wsm_audio_loss | WSMAudioLoss | Registered |
| CALLBACKS | wsm_segment_metrics_callback | WSMSegmentMetricsCallback | Registered |
| CALLBACKS | wsm_audio_metrics_callback | Common callback alias | Registered |
| CALLBACKS | wsm_summary_callback | WSMSummaryCallback | Registered |

Chimera ML 0.3.0 provides MODELS, LOSSES, METRICS, OPTIMIZERS, SCHEDULERS, CALLBACKS, DATAMODULES, COLLATES, LOGGERS, and INFERENCE_STEPS registries. It has no standard FEATURE_EXTRACTORS registry.

## 4. Data Flow

### 4.1. Segment index

src/common/utils/segment_index.py:

- defines depression/Parkinson tasks and IDs;
- reads train/dev from split_labels_segments_min_filtered.csv-like files;
- reads semicolon-separated test metadata;
- excludes three listed damaged depression/dev segments;
- stores label, video_id, segment path, and soft/hard filters;
- prevents video_id cross-split leakage with priority test > dev > train.

This still needs a formal data audit. The initial code has neither a global two-label target matrix nor an observed-label mask.

### 4.2. Audio datamodule

WSMAudioSegmentDataModule combines both tasks for training, creates separate per-task DEV loaders, creates TEST_NONE/SOFT/HARD loaders, computes inverse-frequency task weights, and publishes dimensions/names/weights through Chimera context.

Important issue: it merges test_dataset into val_dataset, so Test runs every fit epoch. Checkpointing monitors only dev/mean_score, but repeated Test visibility risks human-in-the-loop selection. New methods must evaluate Test only after freeze.

### 4.3. Dataset and cache

WSMAudioSegmentDataset caches each segment and includes modality, extractor/model, layer, pooling, and split in the key. Missing cache triggers extraction in the dataset constructor. It returns temporal audio, pooled audio_cls, task_id, scalar target, metadata, and a padded audio_mask.

Strength: frozen features are reusable. Risks:

- datamodule construction may trigger expensive GPU/network work;
- cache metadata lacks a full preprocessing/code fingerprint;
- an empty dataset accesses feature_paths[0];
- a scalar target cannot represent an unknown second disease label.

### 4.4. Audio features

Supported extractors include WavLM, HuBERT, wav2vec2, data2vec, Transformers audio classifiers, Kintsugi DAM, SpeechBrain emotion, and Vox-Profile speech-flow/voice-quality. The frozen SSL path uses 30-second windows, waveform normalization, a selected hidden layer, and temporal average pooling.

src/common/features/wsm_feature_extractor.py also handles generic video embeddings through Transformers AutoModel. It is a historical helper, not a finished video module.

### 4.5. Audio model and loss

AudioMambaSegmentModel:

1. projects frozen audio features;
2. applies a temporal Mamba or Transformer;
3. resamples to fixed sequence_steps;
4. adds a task-conditioned CLS token;
5. applies a segment-level temporal encoder;
6. uses two task heads and two auxiliary heads;
7. selects the active head by task_id.

WSMAudioLoss computes per-task cross-entropy, averages task losses, and supports label smoothing, focal modulation, class weights, and auxiliary loss.

This is shared-representation MTL, not direct cross-task supervision. A depression sample does not train the Parkinson head. The scalar target and selected output are not multilabel masked BCE.

### 4.6. Metrics and callbacks

WSMSegmentMetricsCallback caches outputs, creates confusion matrices, logs MLflow artifacts, and computes:

\[
Score_D=(UAR_D+MF1_D)/2,
\qquad
Score_P=(UAR_P+MF1_P)/2,
\]

\[
Mean\_Score=(Score_D+Score_P)/2.
\]

Saved configs include checkpoint/early stopping on dev/mean_score in max mode, snapshot, summary, console-file, and MLflow components.

## 5. Experiment Infrastructure

### 5.1. Initial configs

Initial sweep configs covered SSL extractor/layer/pooling, domain-specific wrappers, Mamba versus Transformer, and coarse/refined Optuna searches. scripts/run_audio_experiments.sh referenced a base config that existed only under logs. The initial layout did not satisfy the shared-experiment-name requirement.

### 5.2. Best saved audio run

- run: wsm_audio_models-e0ce-006;
- selected epoch: 4;
- DEV/Mean_Score: 0.787827;
- DEV/Score_D: 0.747918;
- DEV/Score_P: 0.827735;
- historical TEST_NONE/Mean_Score: 0.809486;
- historical TEST_SOFT/Mean_Score: 0.815156;
- historical TEST_HARD/Mean_Score: 0.828135.

Configuration: microsoft/wavlm-base-plus layer 9, temporal_pool 4; Transformer hidden 192, 3 layers, 4 heads, 128 steps, dropout 0.25; label smoothing 0.02, focal gamma 1.0, auxiliary weight 0.1; AdamW lr 1e-4, weight decay 0.01; seed 42.

This is one best saved seed, not a statistically established final result. It is the frozen audio starting point.

## 6. Initial Critical Defect

src/audio/models/audio_mamba_segment.py imported TemporalEncoder, _autocast_enabled, _head, _projection, masked_mean, and resample_valid_sequence from fusion.models.av_sync_mamba_segment. src/fusion was absent, so plugin registration warned that fusion.models could not be found and audio_mamba_segment_model was missing.

The best run's code.zip contained the historical provider plus AV datamodule/dataset/loss. The historical model used modality encoders, lagged synchrony, interleaved tokens, task-conditioned CLS, task heads, auxiliary heads, and contrastive AV loss.

Because src/audio is frozen, the repair must restore a compatible provider outside src/audio. The historical AV implementation is not the new final fusion method.

## 7. Initial Technical Debt

| Priority | Problem | Risk |
|---|---|---|
| P0 | Missing provider required by audio | Main model not registered |
| P0 | Root .gitignore ignored everything except src | Docs/configs/scripts invisible to Git |
| P0 | No canonical base audio config | Checkout not reproducible |
| P1 | Test loaders in every validation epoch | Test-selection leakage |
| P1 | No global multilabel target/mask | Disjoint labels represented incorrectly |
| P1 | Generic primitives located in fusion | Wrong dependency direction |
| P1 | No video/text/description/new fusion code | Target method absent |
| P1 | Extraction in dataset constructor | Heavy uncontrolled initialization |
| P2 | pyproject references missing README.md | Packaging risk |
| P2 | Tracked pycache/egg-info | Platform-specific noise |
| P2 | Flat scripts | Violates scripts/modality layout |
| P2 | No manifest/cache schema version | Stale-feature risk |

## 8. Existing Strengths

- Registry-first Chimera style.
- video_id split-leakage protection.
- Cached frozen features.
- Task heads and task-balanced loss.
- Exact required DEV selection formula.
- All three Test filter protocols.
- Required callbacks/loggers in saved configs.
- Snapshots and MLflow enabled provider recovery.

## 9. Initial Conclusion

The repository had a useful audio baseline and solid tracking, but the initial checkout was non-executable because its provider and canonical config were missing. Scientifically, it remains masked multitask classification: it neither represents both diseases as independent simultaneous labels nor directly trains the unknown head.

The first development gate is reproducibility without changing src/audio. See PROGRESS_EN.md for current verification.
