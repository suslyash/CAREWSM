# WSM Project Requirements

## 1. Authority

This file converts the owner's requirements into testable engineering and experimental rules. **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

Conflict order: latest explicit user instruction; this document; PLAN.md; BASELINES.md and SOTA_REVIEW_EN.md; existing source style.

## 2. Target Task

The model MUST predict two independent binary variables:

\[
\mathbf y_i=[y_i^D,y_i^P],
\qquad y_i^D,y_i^P\in\{0,1,?\},
\]

where D is depression, P is Parkinson's disease, and ? is unknown. For the depression corpus, \(\mathbf m_i=[1,0]\); for the Parkinson corpus, \(\mathbf m_i=[0,1]\). This mask MUST be used by all supervised losses and metrics.

- Unknown MUST NOT be treated as negative.
- Healthy in one corpus MUST NOT become a negative label for the other disease.
- The final model MUST use two sigmoid logits or equivalent independent binary heads.
- It MUST NOT reduce the task to a healthy/depression/PD softmax.
- Simultaneous positive predictions MUST be possible even without co-annotated comorbidity.

Here, multi-task multilabel means two disease tasks, one two-component output, and dataset-level missing task annotations.

Target streams are audio, video, transcript, LLM/VLM-derived observable description or fixed semantic features, and their fusion. Descriptions MUST cover observable evidence, MUST NOT receive the diagnosis/ground truth, and SHOULD explicitly forbid diagnosis.

## 3. Chimera ML

- All training/inference code MUST use chimera-ml.
- src/audio is the style reference.
- Every config-selectable component MUST have a stable registry key.
- Use Chimera MODELS, LOSSES, METRICS, CALLBACKS, DATAMODULES, COLLATES, and INFERENCE_STEPS where applicable.
- If Chimera lacks an appropriate registry, an offline extractor MAY use a typed local registry under src/common/registry; config selection by name remains mandatory.
- src/chimera_plugin.py MUST import every registration module.
- Required project modules MUST import cleanly and MUST NOT be hidden as optional-dependency warnings.

## 4. Required Layout

    src/
    ├── audio/             # frozen existing method
    ├── video/{data,features,models,loss}/
    ├── text/{data,features,models,loss}/
    ├── description/{data,features,models,loss}/
    ├── fusion/{data,models,loss}/
    └── common/{callbacks,data,loss,metrics,models,registry,utils}/

- Unimodal code MUST live only in its modality package.
- Multimodal model/data/loss MUST live under src/fusion.
- Universal components MUST live under src/common.
- A modality package MUST NOT import fusion; fusion MAY import public modality components; common MUST NOT import modalities.
- Executable helpers MUST live in scripts/<modality> or scripts/common, never in src.
- Scripts contain orchestration/preprocessing/reporting; reusable logic remains in src.

## 5. Configs and Experiment Naming

The first programme uses experiment_name:

    wsm_mm_pd_dep_v1

Required layout:

    configs/wsm_mm_pd_dep_v1/
    ├── audio/
    ├── video/
    ├── text/
    ├── description/
    ├── fusion/
    └── ablations/

Every YAML in this programme MUST set:

    experiment_info.params.experiment_name: wsm_mm_pd_dep_v1

Use run_name for specific variants. A fundamentally different programme MAY use another root/name. Configs MUST be self-contained; referenced bases MUST exist under configs, not only logs.

## 6. Frozen Audio Baseline

- src/audio MUST NOT change.
- Use WavLM-base-plus layer 9, pool 4, and the current temporal Transformer.
- New models MAY load its checkpoint or cached outputs through an external adapter.
- A missing provider MAY be restored outside src/audio.
- Any shared-module change MUST leave git diff -- src/audio empty.
- No new audio sweep; only minimal reproduction/smoke verification.

## 7. Unimodal Experiment Budget

Execution-order override accepted by the dataset owner/manager:

- Text/description remains required for the final programme but is deferred until after the audio+video fusion, RAMPS, and core ablation research cycle.
- After Stage 2 video is complete, work MUST proceed directly to audio+video fusion rather than starting text/description.
- The deferred text/description stage MUST be completed before final paper-ready freeze and before claims that require all planned modalities.

- Audio: one frozen baseline; no new architecture family.
- Video: at most two families: DEPART-like CLIP+Transformer, then its prototype-aware variant.
- Text: at most two families: a pretrained text encoder, then selected text plus observable description/semantic features.
- Description MAY have a sanity check but no broad search.
- Most compute MUST go to fusion, missing-label recovery, reliability, and negative-transfer ablations.

## 8. Required Instrumentation

Every training config MUST include checkpoint_callback, snapshot_callback, early_stopping_callback, wsm_summary_callback, wsm_segment_metrics_callback (or a documented successor), console_file_logger, and mlflow_logger.

Checkpoint and early stopping MUST monitor dev/mean_score in max mode. Snapshots MUST save config and relevant source; final runs SHOULD also save manifest/cache versions and commit SHA.

Diagnostics MAY add per-task/aggregate scores, segment/video metrics, pseudo-label coverage/precision, ECE/Brier, task gradient norms/cosines, negative-transfer deltas, and gate statistics.

## 9. Selection and Test Protocol

\[
Score_D=(UAR_D+MF1_D)/2,\quad
Score_P=(UAR_P+MF1_P)/2,
\]

\[
Mean\_Score=(Score_D+Score_P)/2.
\]

Model/config selection MUST maximize DEV/Mean_Score. The formula and monitor MUST remain fixed.

Every training epoch/validation cycle MUST report DEV, TEST_NONE, TEST_SOFT, and TEST_HARD using the same per-task UAR/MF1/Score and Mean_Score definitions.

The metric implementation and reporting stack MUST therefore support the same masked two-task contract for all four protocol prefixes:

    dev
    test_none
    test_soft
    test_hard

For each protocol, report per-task UAR, MF1, Score=(UAR+MF1)/2, and Mean_Score as the arithmetic mean of the depression and Parkinson Scores.

Model/checkpoint selection, early stopping, and any automatic best-epoch logic MUST use only:

    dev/mean_score

The epoch-level TEST_NONE/SOFT/HARD metrics are mandatory comparative-monitoring outputs. They MUST be logged and surfaced alongside DEV metrics so runs can be compared continuously with baselines/other systems.

Test metrics:

- MUST be computed every epoch/validation cycle for TEST_NONE, TEST_SOFT, and TEST_HARD;
- are comparative-monitoring outputs, not automatic selection signals;
- MUST NOT be consumed by checkpoint_callback, early_stopping_callback, threshold search, or automatic hyperparameter/model-selection logic;
- MUST explicitly identify uncleaned/soft/hard protocols;
- MUST use the same task-wise masked-label semantics as DEV metrics.

New datamodules MUST separate DEV from Test at the dataset/protocol level, but the training/evaluation stack MUST expose DEV, TEST_NONE, TEST_SOFT, and TEST_HARD loaders/evaluation streams in every epoch-level evaluation cycle. The three Test protocols MUST never be merged into val_dataset; they remain separately named evaluation streams.

Segment-level metrics remain primary; video/person aggregation SHOULD be secondary. Ablations SHOULD use at least 3 seeds. Final methods MUST use at least 5 seeds and report mean, standard deviation, and confidence interval or a pre-justified alternative. Use paired comparison where samples match.

## 10. Data Integrity

- All modalities MUST share one canonical segment manifest.
- Splits MUST be video independent and identical across streams. For the current WSM_NEW train/dev/test partition, the dataset owner explicitly accepts the provided split assignment as speaker-independent even though no authoritative speaker_id is available. This is an owner-provided dataset assumption, not an empirically verified speaker-identity audit. Do not infer speaker_id or block Stage 1 on a missing speaker map.
- Save a pre-training audit of counts, ID intersections, missing files, duration, and modality availability.
- Feature-cache fingerprints MUST include model revision, layer, preprocessing, sampling, and manifest hash.
- Labels/task identity MUST NOT enter prompts/extraction, except an explicitly studied trainable task token.
- Extraction failure MUST produce an availability mask, not a fake feature.
- For DEPART-comparable video preprocessing, a YOLO body-detection miss is NOT an extraction failure. Every readable sampled frame MUST use the detected body ROI when a valid box exists and MUST fall back to the full RGB frame when no valid body box exists. This fallback is part of the reproducible preprocessing contract, not a fabricated feature.
- The DEPART-comparable video cache used for training/evaluation MUST cover all canonical readable rows in TRAIN, DEV, and TEST. Rows MUST NOT be excluded solely because YOLO failed to detect a body.
- TEST_NONE/SOFT/HARD dataset membership counts are defined before video preprocessing availability filtering. For DEPART-comparable comparisons, the evaluation DataModule MUST preserve full protocol membership unless the underlying video itself is unreadable or another non-YOLO fatal extraction error is explicitly documented.
- Distinguish modality dropout from real extraction failure.

## 11. Supervision

Required sparse baseline:

\[
\mathcal L_{\mathrm{obs}}
=
\frac{\sum_{i,t}m_{it}\,\ell(\hat y_{it},y_{it})}
{\sum_{i,t}m_{it}+\epsilon}.
\]

Implement it before pseudo-labeling.

The final method SHOULD create a soft pseudo-target only when \(m_{it}=0\). Acceptance MUST use calibrated confidence, uncertainty, independent modality/teacher agreement, separate positive/negative handling, warm-up, and a detached reliability weight. Observed truth always overrides pseudo-labels.

LLM/VLM output is auxiliary evidence, not a sole medical annotator. Cache/audit descriptions. Never use a freely generated diagnosis as truth. Semantic descriptions/label embeddings MAY enter a separately ablated acceptance or consistency term.

## 12. Publication Claims

Without a dual-annotated audit subset, do not claim verified recovery of true comorbidity or the missing disease label. Limit claims to improved observed-task generalization and controlled cross-corpus transfer.

For stronger claims, create a small independently dual-annotated audit subset under a predeclared evaluation/calibration protocol; do not use it for ordinary training.

Required novelty ablations:

1. sparse masked MTL;
2. task-aware directed fusion;
3. direct pseudo-supervision;
4. semantic evidence;
5. uncertainty/reliability filter;
6. reliability-aware objective balancing.

## 13. Verification

Unit tests are optional, but each stage MUST verify:

- plugin imports without project-module warnings;
- expected registry keys;
- Chimera config validation;
- one forward/loss/backward smoke batch when relevant;
- output shapes/masks and finite values;
- required callbacks/loggers;
- no src/audio diff;
- no Test-based selection.

## 14. Reproducibility

Every significant run MUST save resolved config, seed, commit SHA/dirty flag, package versions, feature/cache manifest, selected checkpoint, DEV metrics, separate final Test report, summary, and MLflow run ID. Every paper number MUST trace to a config, checkpoint, and run artifact.

## 15. Completion

The project is paper-ready only when all streams/fusion are registered; frozen audio is reproducible; video/text limits are respected; the baseline ladder is complete; the final model predicts two independent labels; direct reliability-filtered cross-corpus supervision exists; DEV/Mean_Score is the sole selector; Test runs only after freeze; multi-seed/calibration/negative-transfer/modality ablations exist; and claims match available evidence.
