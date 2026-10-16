# NEXT TASK

## TASK-001C — remove obsolete WSM source and restore original ChimeraML layout

**READY. This replaces/cancels the prior TASK-001C plugin-only plan.** The owner requires `src/` to contain **only ChimeraML code**, with the original `audio/`, `fusion/`, `common/` division. No `src/carewsm/`, `src/legacy/`, README or one-off scripts inside `src/`.

### Start
Run `git fetch origin`, `git switch main`, `git pull --ff-only origin main`, then create NEW branch `codex/task-001c-src-cleanup` from the current main. Never reuse old TASK-001B or the canceled `codex/task-001c-plugin-packaging-isolation` branch. Read `AGENTS.md`, active docs and the accepted TASK-001A audit. Check `git status --short` and inventory all `git ls-files src` before touching anything. If unknown/unclassified owner code appears, stop and flag it, not delete blindly.

### Goal — this task must actually CLEAN src, not merely relabel code legacy
Only tracked source files remaining under `src/` on completion:
- `src/audio/__init__.py`
- `src/fusion/__init__.py`
- `src/common/__init__.py`
- `src/chimera_plugin.py`
The three namespaces are empty placeholders for future reviewed ChimeraML components. `src/chimera_plugin.py` must be a minimal no-op callable `register() -> None` that imports/registers NO old WSM models, callbacks, losses, DataModules or optimizers. No README/docs or one-off code inside src.

### Authorized cleanup scope
1. Confirm against TASK-001A that tracked pre-CAREWSM files beneath `src/common/`, `src/audio/`, `src/fusion/`, `src/video/`, `src/text/` are copied obsolete WSM components. Remove those legacy implementations from the current Git tree, then create only the three clean namespace initializers. Git history preserves the deleted implementation (bootstrap PR #1). Do not move it to `src/legacy/` or create a copy elsewhere.
2. Remove the mistaken TASK-001B `src/carewsm/` scaffold and `src/legacy/README.md`. Their accepted audit rationale remains in `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`. Remove tracked generated `src/wsm.egg-info/` and other previously identified WSM package metadata. Enumerate any additional tracked src path before deleting it; stop for manager if it is not conclusively old WSM/generated.
3. Update `pyproject.toml`: project name and description CAREWSM; exactly one `[project.entry-points."chimera_ml.plugins"]` mapping `carewsm = "chimera_plugin:register"`; keep `[tool.setuptools] py-modules = ["chimera_plugin"]` for the simple source-level shim. Preserve dependency list and other metadata unchanged; dependency pruning is another task.
4. Remove the now-obsolete `tests/carewsm/test_package_scaffold.py`. Create a minimal standard-library-only `tests/test_src_boundary.py` verifying the exact tracked or filesystem source layout, no WSM plugin registration, the new project entrypoint and import-safe no-op plugin. Do not edit unrelated historical tests.
5. Update only `docs/PROGRESS_EN.md` with TASK-001C changes, evidence and pending manager review.

### Allowed tracked paths
`src/**` (removals/minimal files as above), `pyproject.toml`, deletion of `tests/carewsm/test_package_scaffold.py`, creation of `tests/test_src_boundary.py`, and `docs/PROGRESS_EN.md`. No other tracked modifications. Do not modify `AGENTS.md`, NEXT_TASK, PLAN, PROJECT_REQUIREMENTS, MANAGER_CONTRACT, or TASK-001A audit evidence yourself.

### Prohibitions
No data writes to CARE or WSM; no changes to splits, folds, labels or extracted features. No extraction, DiariZen, audio processing, model loading, training, evaluation, TEST metrics or new DataModule/model implementation. Do not move/remove old `configs/`, `scripts/`, or other historical `tests/` yet. No heavyweight pip install, no force push, no Codex PR/merge.

### Verification / handoff
- Report before/after tracked src inventory, counts, reasoned deletion summary, and any possible historic-script import fallout to handle later.
- `git diff --check`; `git ls-files src` shows only the four specified files; Git diff limited to allowed paths.
- Run `python -m unittest discover -s tests -p 'test_src_boundary.py'` and `PYTHONPATH=src python -c 'import chimera_plugin; chimera_plugin.register()'`. State exact outputs or dependency blockers; don't run unrelated old tests or claim they pass.
- Commit and normally push branch `codex/task-001c-src-cleanup`, leave clean worktree, provide full SHA, exact main base, and detailed evidence. No PR; manager reviews then creates/merges on PASS or sends a brief correction on FAIL.
- End handoff: `Manager review TASK-001C. Do not proceed with configs/tests/scripts cleanup, extraction, DiariZen, training or data-layer implementation.`
