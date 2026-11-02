# CAREWSM Project Requirements

## Research goal

Build an audio-only joint system for Depression and Parkinson recognition from CARE and WSM using the already extracted acoustic feature families.

The owner wants a broad but disciplined feature-combination study with strong emphasis on temporal modelling.

Multimodality is explicitly deferred.

## Data

CARE:

`/media/maxim/Databases/CARE/`

WSM:

`/media/maxim/Databases/WSM_NEW/`

Expected fixed partition contract, to be verified by Codex:

- train/dev/test;
- five CV folds inside TRAIN;
- participant/profile-level grouping without leakage.

Never regenerate splits unless the owner explicitly requests it.

## Audio feature contract

The owner reports that **11 audio feature families have now been extracted**.

The exact names, files, schemas, shapes, temporal/static status, and successful coverage are not to be guessed by the manager. TASK-001 must discover them from the actual data and extraction code.

The audit must identify, per sample and per feature family:

- expected artifact;
- file presence;
- readability;
- schema/keys;
- dtype;
- shape;
- temporal length where applicable;
- finite-value validity;
- failure reason.

Do not delete failed samples or failed feature files.

Create deterministic availability metadata that later experiments can use for filtering/masking.

## Missing feature policy

Default research policy until the audit is reviewed:

- experiments requiring feature set S use only samples where every feature in S is valid;
- unavailable disease labels remain label-masked;
- no unavailable feature is silently replaced by zeros as if observed;
- coverage/retention must be logged for every run.

For fair feature-combination comparison, the manager may define a fixed common-coverage cohort after TASK-001 reports how much data would be lost. This decision is intentionally deferred until coverage is known.

## Model objective

Use three strong recent temporal classification architectures as the main model families:

1. TSCMamba-inspired;
2. MPTSNet-inspired;
3. MD-Former-inspired.

Implementation must be adapted cleanly to CAREWSM feature tensors and Chimera-ML, not copied blindly.

A small pooled MLP baseline is allowed as a sanity/reference model and is not one of the three research models.

## Heterogeneous feature handling

Feature families may be temporal or static and may use different temporal resolutions.

Do not naively concatenate incompatible raw timelines.

The implementation should expose family-specific adapters:

- temporal family -> masked sequence adapter;
- static/aggregated family -> compact feature token/embedding;
- model/fusion layer consumes standardized hidden representations.

The exact adapter contract is frozen only after TASK-001 records actual schemas.

## Code quality

Active Chimera-ML Python code belongs under `src/audio/` and must be split by responsibility.

Prefer small modules, clear dataclasses/config parameters, deterministic behavior, explicit masks, and unit tests for data/model contracts.

Do not revive old WSM architecture coupling merely because legacy files remain in the repository.
