# CAREWSM Active Progress

## Current state

The project has restarted around an audio-only research programme.

Current repository `main` at manager bootstrap start:

`08e6cb5fc3d5e6bb737e796c7c13feff6e165c76`

The owner reports that all intended 11 audio feature families have been extracted, but some samples/features failed and coverage has not yet been validated.

No feature file is accepted as usable until TASK-001 validates it.

No CAREWSM audio model result is currently accepted as project evidence.

## Current stage

Stage 0 — documentation and operating contract.

## Next stage

TASK-001 — fast feature extraction integrity/coverage audit and deterministic filtering metadata.

## Scientific direction

Main emphasis: temporal modelling over extracted acoustic features.

Three research model families:

- TSCMamba-inspired;
- MPTSNet-inspired;
- MD-Former-inspired.

A pooled MLP is a sanity baseline.

Feature search uses a staged funnel rather than exhaustive subset enumeration.

Optuna is reserved for the final winning model/feature combination.

Multimodality is not active.
