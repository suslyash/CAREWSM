# NEXT TASK

## TASK-001C — CAREWSM packaging and Chimera plugin isolation

**Status: READY.** TASK-001A and TASK-001B are manager-accepted and merged (PR #3 and PR #5). TASK-001C is a separate, narrow **non-destructive** repository-cleanup task. It must not trigger corpus/data processing or activate copied WSM models.

### Startup / branch
1. Run `git fetch origin`, `git switch main`, `git pull --ff-only origin main`.
2. Create the fresh branch `codex/task-001c-plugin-packaging-isolation` from the now-current `origin/main`. Never reuse TASK-001B.
3. Read `AGENTS.md`, active `docs/` in the prescribed order, accepted `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`, and inspect `pyproject.toml`, `src/chimera_plugin.py` and the new `src/carewsm/` scaffold. Run `git status --short`; preserve unrelated changes.

### Goal
Make CAREWSM the active distribution/plugin identity while leaving all previous WSM implementation and legacy configurations **untouched** and **unregistered**. The project does not yet have an approved active model or DataModule.

### Allowed tracked changes
- `pyproject.toml`: change distribution metadata `name`/`description` to CAREWSM, replace the existing `chimera_ml.plugins` entry `wsm = chimera_plugin:register` with exactly one `carewsm = carewsm.chimera_plugin:register`. Remove the old `py-modules = ["chimera_plugin"]` stanza if needed to prevent packaging the old top-level entrypoint. Preserve all pinned dependencies and all unrelated build metadata.
- Create `src/carewsm/chimera_plugin.py` with an explicit minimal, import-safe, callable `register()` that **registers no models, DataModules, losses, callbacks, or optimizers** and imports no old WSM modules. Explain in a docstring that active registrations require later reviewed tasks.
- Create `tests/carewsm/test_plugin_contract.py`: standard-library-only tests for distribution metadata / entrypoint, import and call of the new no-op plugin, and no old WSM registration triggered. Use `tomllib` and `importlib` if helpful. Ensure tests can be run directly via a bounded Python invocation when pytest is unavailable.
- Update only `docs/PROGRESS_EN.md` to describe TASK-001C outcome pending manager review.

### Explicit prohibitions
- Do not modify `src/chimera_plugin.py`, `src/common/`, `src/audio/`, `src/fusion/`, `src/text/`, `src/video/`, or historical `configs/`, `scripts/`, `tests/`. No moving/deleting legacy code or stale `src/wsm.egg-info/` artifacts. Do not alter active PLAN/NEXT_TASK/REQUIREMENTS/MANAGER_CONTRACT/AGENTS or TASK-001A report.
- No dependency installation, packaging/build step that resolves heavy dependencies, model import/load, Chimera training/evaluation, feature extraction, DiariZen, Test metrics, data writes, or new data layer. Do not repair unknown CARE loader or registry dimensions.

### Verification / handoff
- Provide `git diff --check`, changed-path list against `origin/main`, standalone standard-library test invocation and output, evidence that the only active plugin entrypoint is `carewsm`, and evidence the old `src/chimera_plugin.py` remains unmodified.
- Worktree clean after commit. Normal non-force push of `codex/task-001c-plugin-packaging-isolation`; no PR (manager handles review and merge).
- Handoff: Outcome; Changed files; Verification; Data/Test access; Git handoff (full SHA, base SHA, branch, push); Blockers and risks; Next atomic step.
- Final sentence: `Manager review of TASK-001C; do not start legacy relocation, extraction, DiariZen, data-layer implementation, or training.`
