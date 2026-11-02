# TASK-002 — Joint CARE+WSM audio DataModule (native features only)

## Goal
Implement a small, reusable, **data-only** Chimera-ML-compatible layer for joint CARE+WSM five-fold CV, consuming TASK-001's audited feature registry and availability manifest. Do not train, derive Mantis embeddings, choose models, or run experiments.

## Base / branch
Start from latest `origin/main`; create `codex/task-002-joint-datamodule`. Normal push, **no PR or merge**. Read `AGENTS.md`, current docs, `docs/evidence/TASK-001_FEATURE_AUDIT_EN.md`, `reports/feature_audit/audit_summary.json`, and TASK-001 implementation before editing.

## Authoritative inputs
- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`
- Source splits: `splits/train.csv`, `dev.csv`, `test.csv`, `folds/fold_{0..4}_{train,val}.csv`.
- Audit manifest: `reports/feature_audit/sample_feature_availability.csv` (10,464 unique `(corpus,sample_id)` at TASK-001); 11 audio families. Follow actual `extracted/<LABEL>/<SUBGROUP>/PROFILE_<id>/VIDEO_<id>/` mapping.
- CARE lacks `splits/videos.csv`; do not require it. WSM has one. Read observed label conventions rather than guessing.

## Split and identity contract — critical
1. Canonical identity is `(corpus, sample_id)`; retain `corpus, profile_id, video_id` and source provenance. Do not collapse same-named IDs across corpora.
2. **Fold files are authoritative**: for each k=0..4, TRAIN = CARE fold_k_train ∪ WSM fold_k_train; VAL = CARE fold_k_val ∪ WSM fold_k_val. Build them directly from the fold CSVs, joined to the manifest by canonical identity. Reject missing, conflicting or duplicate keys rather than silently dropping them.
3. TASK-001 found `dev.csv` is represented within `train.csv` (CARE: 276, WSM: 936 overlapping sample IDs). The source `train.csv` is NOT automatically a model-training set. **Do not add DEV independently to TRAIN or VAL.** Keep `dev.csv` as metadata only pending a separate manager policy. Preserve and report provenance and conflicts; do not alter canonical CSVs.
4. Make strict assertions for fold TRAIN/VAL disjointness by sample and by `(corpus, profile_id)`; assert no TEST overlap with fold TRAIN/VAL by sample/profile, and investigate/report any mismatch. Validate fold membership against the actual source metadata, not embedded `split` flags alone.
5. Construct separate fixed CARE TEST and WSM TEST evaluation views (never mixed into training). Provide CARE-only and WSM-only VAL views **for metrics**, not separate training pipelines. Require explicit corpus namespace in every split/profile comparison.
6. Return deterministic row order and split/fold fingerprints. Report row/profile counts and exclusion reasons for every view.

## Labels / availability
- Two independent targets: Depression and Parkinson. Provide binary target values and explicit `observed` booleans; never treat missing as healthy/negative. Handle CARE and WSM's observed semantics explicitly and document assumptions; raise on conflicting labels for duplicate source rows.
- For user-selected feature set S (subset of audited family registry), keep only rows where every family in S is `VALID`. Do **not** globally complete-case filter; report before/after sample counts, per-corpus/per-view/task-observed coverage, and failures by family.
- Ensure no dropped sample appears silently. Keep filtering identical in meaning across TRAIN/VAL/TEST. Expose individual family availability even when not selected.

## Feature data interface
- Implement native, lazy loading for verified CSV and NPZ family schemas using TASK-001 registry/evidence. Preserve temporal rows, turn ordering, timestamps and validity metadata when present. Static/aggregated vectors stay static. Do **not** fabricate a time axis or expand static vectors to 1-second frames.
- Return typed/explicit per-sample structures with family-specific arrays, lengths, masks/metadata; handle variable-length blocks in a minimal collate adapter, without assuming equal sampling rates or sequence lengths. Ensure numerical shape/dtype validation and actionable errors.
- Do not fit scaler/PCA or calculate class weights in this task. Define a clear future TRAIN-only fitting hook/interface, but leave implementation for a separate task. No global normalization.
- Keep active code under `src/audio/data/` and, only if necessary, `src/audio/features/`; tiny configs/wrappers allowed. Follow existing Chimera-ML conventions without implementing model, loss, trainer, or callbacks.

## Deliverables / checks
- Modular implementation; concise example CLI or runnable smoke command that prints five joint fold counts, feature-set retention, observed-label coverage and CARE/WSM test-view counts **without training**.
- Small focused synthetic tests for identity, duplicate DEV provenance, fold sample/profile isolation, label masks, feature-set intersection, and native temporal/static loading. No expanded testing programme.
- Run available checks and a **lightweight real-data smoke** for all five folds and a representative high-coverage feature, plus one 2-family intersection. Avoid full feature rescans, extraction or large jobs.
- Add `docs/evidence/TASK-002_JOINT_DATAMODULE_EN.md` with commands, real counts, limitations, split integrity, fingerprints, and representative shapes. Update only `docs/PROGRESS_EN.md` among management docs. Do not rewrite methodology.
- `git diff --check`, focused tests and clean worktree; normal commit + push. Handoff with the eight headings prescribed by `AGENTS.md`.

## Explicitly prohibited
Changing canonical splits/artifacts, Mantis derivation, feature re-extraction, classifier implementation, training, Optuna, loss studies, transfer/adaptation, external evaluations, TEST-driven choices, beginning TASK-003, or creating/merging PRs.

**Stop after handoff for independent manager PASS/FAIL.**
