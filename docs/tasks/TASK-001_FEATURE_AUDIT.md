# TASK-001 — Fast audio feature integrity, coverage and temporal-schema audit

## Goal

Audit the **actual extracted CARE and WSM audio features** on the local machine as quickly as possible and produce deterministic availability metadata that unblocks the Chimera-ML data layer and Mantis temporalization.

The owner reports 11 intended audio feature families have been extracted, but extraction failed for some samples.

Do not guess the exact operational family list. Discover it from live extraction outputs/code, CARE's registry/loader and the actual files.

## Base / branch

Repository:

`suslyash/CAREWSM`

Start from current `origin/main`.

Create:

`codex/task-001-feature-audit`

Normal non-force push is authorized.

Do not merge. The manager owns review/PR/merge.

## Required reading

Read:

1. `README.md`
2. `AGENTS.md`
3. all active `docs/*.md`
4. this task
5. extraction scripts/registries/loaders needed to determine feature outputs

Inspect `git status --short` before editing.

## Data roots

CARE:

`/media/maxim/Databases/CARE/`

WSM:

`/media/maxim/Databases/WSM_NEW/`

Expected CARE metadata:

- `splits/videos.csv`
- `splits/train.csv`, `dev.csv`, `test.csv`
- `splits/folds/fold_{0..4}_{train,val}.csv`
- `splits/features.json`
- `splits/care_features.py`

Expected WSM metadata:

- `splits/videos.csv`
- `splits/train.csv`, `dev.csv`, `test.csv`
- `splits/folds/fold_{0..4}_{train,val}.csv`

Expected extracted layout:

`extracted/<LABEL>/<SUBGROUP>/PROFILE_<profile_id>/VIDEO_<video_name>/`

The task may read both corpora but must not delete, re-extract, rename, repair or overwrite source feature artifacts.

## Priority

Finish this gate quickly.

Prefer:

- directory/metadata enumeration;
- vectorized CSV operations;
- minimal container opening;
- mmap/header inspection where appropriate;
- one deterministic audit pass;
- parallelism only where safe.

Do not run models, feature extraction, Mantis or Chimera training.

## A. Verify split/fold/sample contract

For CARE and WSM independently verify:

- actual `videos.csv` columns;
- `sample_id`, `profile_id`, `video_dir`, `split`, `cv_fold` when present;
- train/dev/test membership agreement;
- five TRAIN CV folds and fold membership;
- no obvious profile/participant overlap across train/dev/test;
- duplicate sample identities;
- actual Depression/Parkinson label columns;
- missing-label representation.

Report observed deviations explicitly.

Do not regenerate any split.

## B. Discover the exact audio feature registry

Use, in order where useful:

1. actual extracted video directories;
2. CARE `splits/features.json`;
3. CARE `splits/care_features.py`;
4. current CAREWSM extraction scripts;
5. representative valid artifacts.

Resolve the owner's reported count of 11.

For every operational audio feature family record:

- canonical family name;
- CARE path/file pattern;
- WSM path/file pattern;
- file extension/container;
- top-level keys/columns;
- dtype;
- rank and shape semantics;
- frame-level / turn-level / static-aggregated classification;
- temporal axis and units if present;
- timestamps/start/end fields if present;
- turn identifiers if present;
- valid-observation/confidence information if present;
- extractor/source script;
- CARE/WSM schema compatibility.

Do not merge distinct families merely to force the count to 11. If operational reality differs, report it.

## C. Validate every expected sample × feature family

Use explicit status classes:

- `VALID`
- `MISSING`
- `ZERO_BYTE`
- `UNREADABLE`
- `SCHEMA_MISMATCH`
- `EMPTY`
- `NONFINITE`
- `INVALID_DTYPE`
- `INVALID_SHAPE`
- `IDENTITY_MISMATCH`
- `OTHER_ERROR`

A file is VALID only if the checks appropriate for that family pass.

Validate at least:

- existence;
- nonzero size;
- opens successfully;
- required keys/columns;
- numeric dtype where expected;
- expected rank/shape;
- nonempty time/turn/static axis;
- finite numeric values;
- identity metadata consistency where available;
- monotonic/order-valid timestamps where available.

Do not silently repair invalid artifacts.

## D. Temporal-schema evidence for later Mantis windows

The next methodology stage intends 1.0-second windows with 0.5-second hop.

TASK-001 must therefore answer, per feature family:

- is there a native time axis?
- what is its resolution?
- are timestamps absolute or relative?
- can 1.0 s windows be created without inventing timing?
- if turn-level: are turn start/end times available?
- if static/aggregated: is the family inherently not windowable?
- are validity/confidence values available for computing valid-observation fraction?

Classify each family provisionally as:

- `MANTIS_WINDOWABLE_NATIVE`
- `MANTIS_WINDOWABLE_FROM_TIMESTAMPS`
- `NATIVE_ONLY_NO_DEFENSIBLE_1S_AXIS`
- `UNRESOLVED`

This is an audit classification only. Do not generate Mantis embeddings in this task.

## E. Coverage outputs

Produce:

1. coverage by corpus / split / feature family;
2. failure counts by status/reason;
3. valid count/percentage for each single family;
4. valid count/percentage for all families;
5. pairwise joint-valid coverage matrix;
6. Depression-observed and Parkinson-observed coverage;
7. coverage by corpus and label task;
8. per-family Mantis-windowability classification.

Also compute retention estimates for:

- global complete-case across all families;
- per-experiment filtering for arbitrary selected feature sets.

## F. Machine-readable availability manifest

Build a deterministic manifest keyed by canonical sample identity.

Include:

- corpus;
- sample/video/profile IDs;
- split;
- fold information;
- Depression label + observed mask;
- Parkinson label + observed mask;
- one status column per family;
- one boolean valid column per family.

Preferred tracked output:

`reports/feature_audit/sample_feature_availability.csv`

If too large for Git:

- store the full manifest in a deterministic local report location;
- record absolute path, SHA256, row count and byte size in tracked evidence;
- track only a small representative sample if useful.

Do not modify canonical split CSVs.

## G. Reusable audit code

Place reusable Python code under:

`src/audio/data/`

Suggested:

`src/audio/data/feature_audit.py`

Requirements:

- deterministic;
- no Chimera training dependency;
- pure validation functions where practical;
- clear family registry representation;
- CLI or thin launch wrapper;
- reusable by later DataModule/tests.

A small wrapper under `scripts/` is allowed if necessary.

## H. Tracked evidence

Create:

`docs/evidence/TASK-001_FEATURE_AUDIT_EN.md`

Include:

1. executive result;
2. exact observed family registry;
3. CARE split/fold/label sanity;
4. WSM split/fold/label sanity;
5. schema table;
6. temporal/timestamp/Mantis-windowability table;
7. coverage tables;
8. failure reasons;
9. complete-case retention;
10. pairwise coverage;
11. manifest path/hash/rows;
12. runtime;
13. exact commands;
14. limitations;
15. recommendation for TASK-002 data-layer/windowization contract.

Update only:

`docs/PROGRESS_EN.md`

Do not rewrite project methodology docs from Codex. If observed facts contradict them, report the contradiction and let the manager change the contract.

## I. Tests

Add focused synthetic tests for:

- valid artifact;
- missing artifact;
- corrupt/unreadable artifact;
- empty temporal/turn axis;
- wrong shape;
- invalid dtype;
- NaN/Inf;
- invalid timestamp order where applicable;
- manifest status aggregation.

Unit tests must not require real CARE/WSM data.

## J. Prohibitions

Do not:

- train a model;
- run Chimera experiments;
- run Optuna;
- derive Mantis embeddings;
- select Mantis checkpoint/layer/pooling;
- re-extract audio features;
- delete/repair failed features;
- rewrite train/dev/test/folds;
- implement final DataModule;
- implement classifier models;
- start transfer/adaptation/external experiments;
- start TASK-002.

## Verification

Before commit:

- `git diff --check`;
- focused unit tests pass;
- no source feature artifact changed;
- no canonical split/fold file changed;
- record audit runtime/row counts;
- repository diff is task-scoped;
- worktree clean after commit.

Commit message:

`TASK-001: audit audio feature integrity and temporal schemas`

## Required handoff

Respond in English with exactly:

1. Outcome
2. Changed files
3. Verification
4. Data/Test access
5. Experimental evidence
6. Git handoff
7. Blockers and risks
8. Next atomic step

Section 8 must say exactly:

`Manager review of TASK-001; do not implement the DataModule, derive Mantis embeddings, train models, run loss-balancing experiments, Optuna, transfer, adaptation, or external evaluation yet.`
