# CAREWSM Active Progress

## Current state

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

Stage 2 — TASK-002 joint CARE+WSM native data layer is implemented on `codex/task-002-joint-datamodule`; manager review is pending. It consumes canonical fold files directly, keeps DEV metadata-only, and does not train or derive Mantis representations.

## Next task

TASK-001 — fast audit of real extracted feature artifacts and deterministic filtering metadata.

No training or Mantis cache generation is authorized until manager review of TASK-001.
