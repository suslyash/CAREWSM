# CAREWSM Active Progress

## Current state

TASK-005 native Chimera-ML training pipeline is implemented on `codex/task-005-training-pipeline`. A real bounded fold-0 MPTSNet train/VAL smoke completed on joint CARE+WSM eGeMAPS LLD with standard console logger and callbacks; evidence is in `docs/evidence/TASK-005_TRAINING_PIPELINE_EN.md`. MLflow is restored with pinned SQLAlchemy 2.0.x compatibility; a real FINISHED MLflow run is recorded.


TASK-004 model infrastructure is implemented on `codex/task-004-model-infrastructure`: registered TSCMamba-inspired, MPTSNet-inspired and MD-Former-inspired native sequence classifiers, masked BCE and six binary metrics. Focused YAML/forward-backward smoke evidence is recorded; TSCMamba remains dependency-blocked if `mamba_ssm` is unavailable.

TASK-001 audit implementation and evidence are complete on `codex/task-001-feature-audit`; manager review is pending. The review correction now resolves WSM through the real `extracted/` tree, validates CARE/WSM NPZ headers without numpy, and records split/fold/leakage and coverage summaries. Live audit found 11 audio registry sources, missing CARE `splits/videos.csv`, a WSM train/dev split inconsistency, CARE artifact failures, and no valid WSM feature mapping under current paths.

The project has been restarted as an audio-only research programme.

Owner-reported state:

- 11 intended audio feature families have been extracted;
- some sample/feature extractions failed;
- coverage has not yet been validated;
- Mantis temporal window embeddings have not yet been frozen as a derived representation contract;
- no current CAREWSM model result is accepted as project evidence.

## Current methodology

Targets:

- Depression binary head;
- Parkinson binary head;
- missing labels masked.

Architecture:

- common/shared representation;
- task-specific feature layer per disease;
- disease-specific binary head.

Metrics per task/fold:

- Macro F1;
- Weighted F1;
- Macro Recall (UAR);
- Weighted Recall;
- Patient Recall;
- Healthy Recall.

Every fold/configuration is evaluated on CARE TEST and WSM TEST; the same six metrics are also logged as a simple metric-wise CARE/WSM Average. TEST metrics are evidence only and cannot drive experiment selection.

Loss study:

- BCE;
- weighted BCE;
- Equal;
- PCGrad;
- CAGrad;
- STCH;
- PAGB;
- single-task reference models.

Temporal representation:

- frozen Mantis embeddings;
- 1.0 s windows;
- 0.5 s hop;
- sequence retained per video/block.

Primary research protocol:

- CARE + WSM are mixed from the first training experiment;
- fold-k TRAIN = CARE fold-k TRAIN + WSM fold-k TRAIN;
- fold-k VAL = CARE fold-k VAL + WSM fold-k VAL;
- model/feature/loss selection uses validation only;
- every selected fold checkpoint logs CARE TEST and WSM TEST statistics.

Deferred until the primary joint model is complete:

- CARE-only / WSM-only ablations;
- direct transfer;
- D-Vlog / YouTubePD;
- small-target adaptation.

## Current stage

TASK-003 dual representation adapters are implemented on `codex/task-003-dual-representation`. Native sequences and defensible timestamped 1.0 s / 0.5 s windows are smoke-tested on real CARE and WSM `egemaps_lld` samples. Frozen Mantis inference is explicitly BLOCKED because the official `mantis-tsfm` package is not installed; no checkpoint or cache was invented. Manager review is required before installing/pinning Mantis and generating embeddings.

## Next task

TASK-001 — fast audit of real extracted feature artifacts and deterministic filtering metadata.

No training or Mantis cache generation is authorized until manager review of TASK-001.

## TASK-006 preparation

The TASK-006 experiment organization is prepared but no real training was run in this revision. Native eGeMAPS LLD fold-specific YAMLs and one explicit universal launcher with five-fold execution are available for MPTSNet and MD-Former. Each fold uses full canonical TRAIN/VAL rows, a fixed seed, contiguous full-trajectory bin pooling to 256 steps (no first-N truncation), VAL-only aggregate Macro F1 checkpoint/early-stopping monitor, and standard Chimera logging/callbacks. TEST is intentionally deferred to a separate fixed-checkpoint evaluation mode.

## TASK-007 audio feature matrix

Static native schema audit confirmed all 11 registry audio families and dimensions. Explicit MPTSNet and MD-Former fold YAMLs are prepared under `configs/audio/<feature>/` for every confirmed family. The single Bash launcher discovers folders and existing fold files; no experiments were run.
