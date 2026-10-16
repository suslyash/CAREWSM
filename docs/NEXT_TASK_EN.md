# NEXT TASK

## TASK-001D — delete obsolete WSM configs and flatten legacy models/callbacks

**Status: READY; issued after TASK-001C manager PASS and PR #9 merge.**

### Owner-mandated end state

- Preserve active ChimeraML source structure `src/audio/`, `src/fusion/`, `src/common/` and the no-op `src/chimera_plugin.py`.
- Restructure the **passive** archive to exactly these direct category folders (plus optional minimal Python package initializers):

```text
src/legacy/
├── audio_models/
├── fusion_models/
├── video_models/
├── text_models/
└── callbacks/
```

- Move the **15** previously archived model implementation files, byte-for-byte, from `src/legacy/{audio,fusion,video,text}/models/` to their respective `src/legacy/{audio_models,fusion_models,video_models,text_models}/` directories. Use `git mv` when possible. Remove now-obsolete intermediate `audio/…/models/` etc package initializers and directories; `src/legacy/__init__.py` may remain.
- Move both existing historical callback source files **without deleting or rewriting their functionality** from `src/common/callbacks/` to `src/legacy/callbacks/`: `wsm_summary_callback.py` and `wsm_segment_callback.py`. Remove now-unused `src/common/callbacks/__init__.py` if the directory is empty of tracked files. Preserve keys `wsm_summary_callback` and `wsm_segment_metrics_callback` in the archived source, **but do not automatically import/register/activate them**.
- Remove **all tracked old WSM-era configuration files** from `configs/` (TASK-001A classified the 157 config files as WSM-era). No stale training, run, ablation, pseudo-label or logging configs should remain. If any tracked config is demonstrably owner-authored new CAREWSM work rather than copied WSM-era, preserve it and flag for manager review. If none are retained, `git ls-files configs` must be empty; empty local directories can persist because Git doesn't track directories.
- Preserve README files only at the repository root or in `docs/`; do **not** create a README anywhere under `src/` or `configs/`. Do not recreate `src/carewsm/` or add new tests.

### Scientific and runtime boundary

This is an **organizational cleanup only**. Archived model/callback source is for reference, may retain unresolved old imports, and is excluded from installed active packages by the existing `pyproject.toml` `exclude = ["legacy", "legacy.*"]`. `src/chimera_plugin.py` remains a no-op and must not import or register archive modules.

Future manager-approved ChimeraML training configs **must** use callbacks `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `wsm_summary_callback` and loggers `mlflow_logger`, `console_file_logger`. The `wsm_summary_callback` implementation is retained in `legacy/callbacks/`, **not active** until a separate task explicitly restores/adapts and registers it under `src/common/`. `wsm_segment_metrics_callback` is optional and needs CAREWSM mask/metrics/TEST-policy validation before any use. Do not create training configs or promise these callbacks currently resolve.

### Startup and branch

1. Run `git fetch origin`, `git switch main`, `git pull --ff-only origin main`. Check `git status --short`. Create **new** `codex/task-001d-configs-legacy-flatten` branch from updated `origin/main`; do not reuse the TASK-001C branch.
2. Independently read `AGENTS.md`, `docs/README.md`, `docs/PROJECT_REQUIREMENTS.md`, `docs/PLAN.md`, `docs/PROGRESS_EN.md`, `docs/NEXT_TASK_EN.md` and `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`. Inspect exact tracked config and legacy/callback inventories before edits.
3. Report unusual owner-created/untracked content instead of force-removing it. Never use global `git clean -fdx` or delete local ignored files.

### Allowed tracked scope

- `configs/**` — delete only obsolete WSM-era files, no replacements.
- `src/legacy/**` — moves/minimal `__init__.py` only, no content/behavior changes to model or callback implementations.
- `src/common/callbacks/**` — move the two designated callback sources out, remove their now-unused package init only if appropriate. Do not change other active `src/` code.
- `docs/PROGRESS_EN.md` — record TASK-001D result **pending manager review**.

Do not edit `pyproject.toml`, `src/chimera_plugin.py`, other `src/`, `scripts/`, `tests/`, `AGENTS.md`, `docs/PLAN.md`, `docs/NEXT_TASK_EN.md`, `docs/PROJECT_REQUIREMENTS.md`, `docs/MANAGER_CONTRACT.md`, or TASK-001A evidence.

### Strict prohibitions

No CARE/WSM dataset access/writes, feature extraction, DiariZen, model or archived code import, package build, training, evaluation, TEST metrics, configuration activation or dependency installations. The owner does not want tests: **do not write or run tests**.

### Acceptance evidence and handoff

- Give full before/after `git ls-files configs src/legacy src/common/callbacks` inventories, counts, a complete 15-model mapping and both callback move mappings. Confirm archive contents are unchanged (git rename similarity or byte-wise SHA-256 before/after).
- Verify `git diff --check`, `git diff --name-status origin/main...HEAD`, only approved tracked paths changed, no historical configs left, `src/chimera_plugin.py` still unchanged/no-op, and `pyproject.toml` still excludes `legacy*`. No nested README or new tests.
- Full commit SHA, exact `origin/main` base SHA, branch, normal non-force push and clean worktree. Do not create a PR or merge: manager handles review/merge on PASS.
- Handoff sections: Outcome; Changed files; Verification; Data/Test access; Git handoff; Blockers and risks; Next atomic step. Stop at manager review: no additional cleanup, feature extraction, diarization, training or data-layer work.
