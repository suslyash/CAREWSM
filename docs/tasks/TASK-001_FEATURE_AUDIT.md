# TASK-001 — Fast audio feature integrity, coverage, and filtering audit

## Goal

Audit the **actual extracted CARE/WSM audio features** on the local machine as quickly as possible and produce deterministic availability metadata that unblocks Chimera-ML experiments.

The owner reports 11 feature families were extracted, but some extractions failed.

Do not guess their names from documentation. Discover the exact families and schemas from the live extraction code and data.

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
3. all active files under `docs/` except old evidence if any
4. extraction/run scripts needed to determine feature outputs

Inspect `git status --short` before edits.

## Data roots

CARE:

`/media/maxim/Databases/CARE/`

WSM:

`/media/maxim/Databases/WSM_NEW/`

Expected WSM extracted root:

`/media/maxim/Databases/WSM_NEW/extracted/`

The task may read both corpora but must not delete, re-extract, rename, repair, or overwrite any feature artifact.

## Priority

This task should be fast.

Prefer:

- directory/metadata enumeration;
- vectorized CSV operations;
- lightweight file header/container inspection;
- memory-mapped or minimal loading where possible;
- one deterministic audit pass.

Do not load huge temporal arrays fully when shape/dtype/finite checks can be obtained safely with a cheaper method. If full finite checking is required for a format, stream/sample intelligently and clearly state what was exhaustive versus sampled.

## A. Discover the exact feature registry

Inspect the live extraction code, CARE feature registry/loader if available, and actual extracted directories.

Produce one canonical table with exactly the feature families that are intended for the audio programme.

For each family record:

- canonical name;
- corpus availability;
- path pattern;
- file extension/container;
- top-level keys;
- temporal vs static/aggregated;
- expected rank/shape semantics;
- dtype;
- timestamp/turn identifiers if any;
- extractor/source script.

Resolve the owner's reported count of 11. If the live data/code implies a different count, report the discrepancy explicitly rather than forcing 11.

## B. Canonical sample identity and split sanity

Inspect the actual split metadata necessary to map features to samples.

At minimum record:

- sample/video/profile identity columns;
- train/dev/test membership;
- fold columns/files;
- Depression/Parkinson label columns and missing-label representation;
- obvious duplicate identities;
- obvious participant/profile overlap across canonical splits.

This is a safety check, not a full new corpus study.

Do not regenerate splits.

## C. Artifact validation

For every expected sample × feature family, classify status using explicit reason codes.

Required status classes:

- VALID
- MISSING
- ZERO_BYTE
- UNREADABLE
- SCHEMA_MISMATCH
- EMPTY
- NONFINITE
- INVALID_SHAPE
- OTHER_ERROR

A file is VALID only if the checks appropriate for that family pass.

Validate at least:

- existence;
- nonzero size;
- container opens;
- required keys;
- dtype is numerically usable;
- rank/shape contract;
- temporal axis > 0 where temporal;
- static vector is nonempty where static;
- finite values;
- identity/turn metadata consistency where stored.

Do not silently repair files.

## D. Coverage outputs

Produce:

1. coverage by corpus / split / feature family;
2. failure counts by reason;
3. number and percentage of samples valid for each single family;
4. number and percentage valid for all audited families;
5. pairwise feature-family joint coverage matrix;
6. label-observed coverage for Depression/Parkinson;
7. coverage by task/corpus where relevant.

Also compute retention estimates for:

- global complete-case across all valid families;
- per-experiment filtering, where a run only requires its selected features.

The manager will choose the final comparison policy after seeing these numbers.

## E. Machine-readable availability manifest

Create a deterministic manifest keyed by canonical sample identity with:

- corpus;
- split;
- fold information where applicable;
- profile/participant id where available;
- Depression label + observed mask;
- Parkinson label + observed mask;
- one status column per feature family;
- one boolean valid column per feature family.

Do not edit canonical dataset split CSVs.

Preferred tracked output:

`reports/feature_audit/sample_feature_availability.csv`

If this is unreasonably large for Git, keep the full manifest under a deterministic local report path, record its SHA256 + row count in the tracked summary, and track a small representative sample instead.

## F. Implement reusable audit code

Put reusable project Python code under:

`src/audio/data/`

Suggested module:

`src/audio/data/feature_audit.py`

The implementation must:

- avoid Chimera training dependencies;
- expose pure functions where practical;
- have a CLI entry or a thin wrapper;
- be deterministic;
- be reusable later by the DataModule/tests.

A small repository wrapper under `scripts/` is allowed only if needed to launch it.

## G. Tracked evidence

Create:

`docs/evidence/TASK-001_FEATURE_AUDIT_EN.md`

Include:

- exact discovered feature list;
- schemas;
- coverage tables;
- failure reasons;
- complete-case retention;
- per-experiment retention findings;
- split/label sanity result;
- runtime;
- exact commands;
- any limitations of the finite-value scan;
- recommendation for Stage 2 filtering/data-layer design.

Update:

`docs/PROGRESS_EN.md`

Do not rewrite PLAN/requirements/model shortlist unless the observed facts make them materially wrong; in that case report the mismatch and leave contract correction to the manager.

## H. Tests

Add focused tests for the audit logic using temporary synthetic feature files.

Test at least:

- valid artifact;
- missing;
- unreadable/corrupt;
- empty temporal;
- wrong shape;
- NaN/Inf;
- manifest aggregation/status mapping.

Do not require the real datasets for unit tests.

## I. Prohibitions

Do not:

- train any model;
- run Optuna;
- re-extract features;
- delete failed samples/features;
- rewrite train/dev/test/folds;
- implement the final DataModule;
- add model code;
- touch TEST for metrics;
- start TASK-002.

## Verification

Before commit:

- `git diff --check`;
- unit tests pass;
- repository diff is limited to audit code/tests/reports/docs explicitly required here;
- no extracted feature file changed;
- no canonical split file changed;
- record audit runtime and row counts;
- worktree clean after commit.

Commit message:

`TASK-001: audit audio feature integrity and coverage`

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Data/Test access
5. Experimental evidence
6. Git handoff
7. Blockers and risks
8. Next atomic step

Section 8 must say exactly:

`Manager review of TASK-001; do not implement the DataModule, models, feature combinations, Optuna, or modify extracted CARE/WSM features yet.`
