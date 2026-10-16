# NEXT TASK

## TASK-001B — minimal CAREWSM namespace and legacy-isolation scaffold

**Status: READY — issue only after TASK-001A manager PASS (PR #3 merged).**

Base: current `origin/main` after the accepted audit and management-contract update. Create a **new** branch `codex/task-001b-repository-scaffold` only after `git fetch origin`, switching to local `main`, and `git pull --ff-only origin main`. Never reuse the TASK-001A branch. Follow `AGENTS.md`, read active docs in order, then the accepted `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`. Check `git status --short` before changes.

### Goal
Create a minimal, **non-destructive** CAREWSM active namespace and documented legacy boundaries. This is preparation for later approved cleanup; do not implement a new DataModule, move old files, or activate old methods.

### Allowed tracked changes
- Create `src/carewsm/__init__.py` plus empty package initializers under `src/carewsm/{data,features,models,evaluation}/`.
- Create `src/legacy/README.md` identifying old WSM modules/configs as **untouched legacy candidates** and linking the accepted TASK-001A KEEP/ADAPT/LEGACY/DELETE inventory.
- Create `configs/carewsm/README.md` describing the future config root, explicitly without active model/train configs.
- Create `tests/carewsm/test_package_scaffold.py` with lightweight import/discovery checks only; run only bounded tests that cannot train/load models.
- Update `docs/PROGRESS_EN.md` to record TASK-001B outcome pending manager review.

### Prohibited
Do not edit `pyproject.toml`, `src/chimera_plugin.py`, existing `src/common/`, `src/audio/`, `src/fusion/`, `src/video/`, `src/text/`, old configs/scripts/tests, active requirements/PLAN/NEXT_TASK/MANAGER_CONTRACT/AGENTS, or the TASK-001A audit. Do not relocate/delete legacy code, register new Chimera components, install heavy dependencies, implement features/DataModules, run DiariZen, evaluate TEST, extract features, train, or write to CARE/WSM. Preserve the missing CARE loader and `linguistic_turnlevel` discrepancy as unverified—do not repair them.

### Acceptance evidence
Report added/changed paths, test commands and outputs, import behavior, `git diff --check`, clean worktree, exact `origin/main` base, new branch/commit SHA, and normal non-force push. Diff must stay within allowed paths. No PR: manager reviews and merges on PASS, or sends a brief corrective message on FAIL.

### Final handoff
Use `Outcome`, `Changed files`, `Verification`, `Data/Test access`, `Git handoff`, `Blockers and risks`, `Next atomic step`. For the last section write: `Manager review of TASK-001B; do not begin cleanup relocation, extraction, DiariZen, or training.`
