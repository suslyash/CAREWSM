# NEXT TASK

## TASK-001C — clean WSM source/scripts, preserve legacy models separately

**Status: READY — final owner-refined replacement for all earlier TASK-001C variants.** This is an authorized *repository-only* cleanup, not an extraction/training/data task.

### Owner's exact layout decision

- `src/` is exclusively for **ChimeraML code**. Keep the original active `src/audio/`, `src/fusion/`, `src/common/` plus the required `src/chimera_plugin.py`.
- **Exception:** archived, **model implementations only**, may live in `src/legacy/`. This is a passive archive; no automatic import, registration, training, or active reuse. Preserve the legacy model source verbatim where practicable.
- Do not add `src/carewsm/`, README/docs or one-off scripts inside `src/`. Put project documentation under `docs/` and any future one-time extraction/preprocessing under top-level `scripts/`.
- **Clean the copied obsolete WSM content out of both `src/` (except archived model code) and top-level `scripts/`.**
- The owner **does not want tests**. Do not create or run tests; remove copied obsolete `tests/` and TASK-001B scaffold test files. Verification is source/diff inspection and explicit evidence, not a new test suite.

### Git startup
1. Run `git fetch origin`, `git switch main`, `git pull --ff-only origin main`; create NEW branch `codex/task-001c-src-scripts-cleanup` from current `origin/main`. Do not reuse `codex/task-001c-src-cleanup`, `codex/task-001c-plugin-packaging-isolation` or TASK-001B branch.
2. Independently read current `AGENTS.md`, all active `docs/` contracts and accepted `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`; inspect `git status --short`.
3. Inventory `git ls-files src scripts tests`, `src/chimera_plugin.py`, and `pyproject.toml`. Compare all candidate deletions with TASK-001A KEEP / ADAPT / LEGACY / DELETE classification. **If any file is not conclusively copied/historical WSM or is new owner work, do not delete it: report the ambiguity to the manager.**

### Work scope — one atomic cleanup
1. **Preserve ONLY actual old model implementation files** from `src/audio/models/`, `src/fusion/models/`, `src/video/models/`, and `src/text/models/` under `src/legacy/<original domain>/models/` (use `git mv` where possible; preserving original names is preferred). Include only minimal Python `__init__.py` needed for archive packaging; no README in `src/`. Do not move old WSM DataModules, feature/cache code, losses, callbacks, utilities, evaluator code, entrypoints, or experimental orchestration into `src/legacy/`. Archived models may have unresolved imports; do not repair or execute them in this task; document the limitation in handoff.
2. Remove all other audited copied WSM implementation from `src/common/`, `src/audio/`, `src/fusion/`, `src/video/`, `src/text/`; leave only empty active `audio/`, `fusion/`, `common/` namespace initializers and a minimal import-safe no-op `src/chimera_plugin.py` that registers **nothing**. Remove the mistaken `src/carewsm/` scaffold, the obsolete `src/legacy/README.md`, and tracked `src/wsm.egg-info/` or other identified generated metadata.
3. Inspect ALL tracked files under top-level `scripts/` by filename, purpose, imports and TASK-001A classification. Remove stale WSM-specific manifests, pseudo-label/metric/evaluation orchestrators, old run/ablation tooling, generated helpers, and obsolete supporting scripts; retain **only** scripts unambiguously relevant to the new CAREWSM work (if none, leave `scripts/` empty/absent). Do not move old scripts into `src/` or run them. In handoff enumerate exactly what was removed, what was retained and why.
4. Remove copied historical WSM tests from `tests/` and obsolete TASK-001B `tests/carewsm/test_package_scaffold.py`. **Create no replacement tests or test infrastructure.** If any test is clearly new owner-authored CAREWSM work, preserve and flag it instead of deleting.
5. In `pyproject.toml`, set project name/description to CAREWSM, keep one plugin entrypoint `carewsm = "chimera_plugin:register"`; keep the necessary `py-modules = ["chimera_plugin"]`. Ensure archived `legacy*` packages are **excluded from install-time package discovery** without deleting their tracked source. Leave dependency pins untouched; dependency pruning is a separate task.
6. Update ONLY `docs/PROGRESS_EN.md` to record TASK-001C changes pending manager review. Preserve TASK-001A findings/limitations unchanged.

### Authorized tracked paths
`src/**`, `scripts/**`, `tests/**` (removal-only), `pyproject.toml`, and `docs/PROGRESS_EN.md`. Do not change AGENTS, PLAN, NEXT_TASK, REQUIREMENTS, manager contract, audit report or unrelated configs. No PR or merge by Codex.

### Safety and verification
- Absolutely no changes to CARE/WSM datasets, features, splits, folds, labels or raw media. No feature extraction, DiariZen, data-layer creation, model loading, training, evaluation, TEST metrics, dependency installation or executing archived code.
- **Do not write or run tests.** Use read-only verification: `git diff --check`, full `git diff --name-status origin/main...HEAD`, `git ls-files src scripts tests`, static inspection of `pyproject.toml` and `src/chimera_plugin.py`, and comparison to the pre-cleanup inventory.
- Handoff must provide before/after counts for `src/`, `scripts/`, `tests/`; an exact retained-model list with destination paths; deletion/retention rationale; proof the only active plugin is no-op; and any unresolved imports/ambiguous owner files.
- Clean worktree, exact current-main base SHA, task branch, full commit SHA, normal non-force push. No PR — manager independently reviews and merges after PASS.
- End: `Manager review of TASK-001C; do not start further configs cleanup, extraction, DiariZen, data-layer implementation, evaluation or training.`
