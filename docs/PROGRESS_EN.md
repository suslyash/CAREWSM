# CAREWSM Active Progress

## Current state

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

Metrics:

- F1;
- weighted F1;
- UAR;
- weighted average recall.

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

Research protocols:

- CARE within-corpus;
- WSM within-corpus;
- CARE -> WSM;
- WSM -> CARE;
- CARE + WSM;
- D-Vlog external Depression;
- YouTubePD external Parkinson;
- small labeled target adaptation.

## Current stage

Stage 1 — feature integrity and coverage audit.

## Next task

TASK-001 — fast audit of real extracted feature artifacts and deterministic filtering metadata.

No training or Mantis cache generation is authorized until manager review of TASK-001.
