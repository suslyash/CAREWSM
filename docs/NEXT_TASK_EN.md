# NEXT TASK

## TASK-001D — delete obsolete WSM configs and flatten legacy models/callbacks

**Status: READY — amended at owner's direction after a reproducible editable-install failure.** TASK-001C manager PASS/PR #9 merged. TASK-001D also repairs packaging dependencies so `pip install -e .` does not attempt to build obsolete WSM `mamba-ssm`.

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
- **Repair editable installation:** the existing `pyproject.toml` still declares copied WSM requirements (`torch==2.10.0`, torchaudio/vision/codec, `mamba-ssm` etc.) even though active CAREWSM source now only exposes inactive namespaces and a no-op ChimeraML plugin. User's unmodified `pip install -e .` failed during the `mamba-ssm` build-requirements hook: its isolated build environment raised `ModuleNotFoundError: No module named 'torch'`. Remove unnecessary legacy-only direct dependencies, especially `mamba-ssm`; do not work around by disabling build isolation or installing another PyTorch just for this. Keep `chimera-ml>=0.2.4` as the required active framework dependency. With no other currently required active imports, `[project].dependencies` should be only `chimera-ml>=0.2.4`; restore further dependencies only in independently authorized feature/model tasks with evidence. Do not add archived model extras or pin a legacy Torch stack.

### Scientific and runtime boundary

This is an **organizational cleanup only**. Archived model/callback source is for reference, may retain unresolved old imports, and is excluded from installed active packages by the existing `pyproject.toml` `exclude = ["legacy", "legacy.*"]`. `src/chimera_plugin.py` remains a no-op and must not import or register archive modules.

Future manager-approved ChimeraML training configs **must** use callbacks `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `wsm_summary_callback` and loggers `mlflow_logger`, `console_file_logger`. The `wsm_summary_callback` implementation is retained in `legacy/callbacks/`, **not active** until a separate task explicitly restores/adapts and registers it under `src/common/`. `wsm_segment_metrics_callback` is optional and needs CAREWSM mask/metrics/TEST-policy validation before any use. Do not create training configs or promise these callbacks currently resolve.

### Startup and branch

1. Run `git fetch origin`, `git switch main`, `git pull --ff-only origin main`. Check `git status --short`. Create **new** `codex/task-001d-configs-legacy-flatten` branch from updated `origin/main`; do not reuse the TASK-001C branch.
2. Independently read `AGENTS.md`, `docs/README.md`, `docs/PROJECT_REQUIREMENTS.md`, `docs/PLAN.md`, `docs/PROGRESS_EN.md`, `docs/NEXT_TASK_EN.md` and `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`. Inspect exact tracked config and legacy/callback inventories before edits.
3. Report unusual owner-created/untracked content instead of force-removing it. Never use global `git clean -fdx` or delete local ignored files.

### Allowed tracked scope

- `pyproject.toml` — **authorized exception to previous TASK-001D:** change only `[project].dependencies` to reflect the minimal active ChimeraML plugin (keep `chimera-ml>=0.2.4`); do not alter project identity, entrypoint, package discovery, or build backend. Confirm `exclude = ["legacy", "legacy.*"]` remains.
- `configs/**` — delete only obsolete WSM-era files, no replacements.
- `src/legacy/**` — moves/minimal `__init__.py` only, no content/behavior changes to model or callback implementations.
- `src/common/callbacks/**` — move the two designated callback sources out, remove their now-unused package init only if appropriate. Do not change other active `src/` code.
- `docs/PROGRESS_EN.md` — record TASK-001D result **pending manager review**.

Do not edit `src/chimera_plugin.py`, other `src/`, `scripts/`, `tests/`, `AGENTS.md`, `docs/PLAN.md`, `docs/NEXT_TASK_EN.md`, `docs/PROJECT_REQUIREMENTS.md`, `docs/MANAGER_CONTRACT.md`, or TASK-001A evidence.

### Strict prohibitions

No CARE/WSM dataset access/writes, feature extraction, DiariZen, model or archived code import, training, evaluation, TEST metrics or configuration activation. The owner does not want tests: **do not write or run tests**. **Only bounded Python packaging/editable-install verification is authorized**, not a model build, heavyweight dependency installation or CUDA/Torch version migration; prefer `python -m pip install -e . --dry-run` in the existing virtualenv and, when package resolution is already satisfied, the exact user command `python -m pip install -e .`. Do not use `--no-build-isolation`, install `mamba-ssm`, or change the owner's virtualenv Torch versions as a workaround. If index/network/other existing dependency issues prevent full installation, report verbatim and separate from the original mamba error; do not claim installation PASS based only on `--no-deps`.

### Acceptance evidence and handoff

- Give full before/after `git ls-files configs src/legacy src/common/callbacks` inventories, counts, a complete 15-model mapping and both callback move mappings. Confirm archive contents are unchanged (git rename similarity or byte-wise SHA-256 before/after).
- Verify `git diff --check`, `git diff --name-status origin/main...HEAD`, only approved tracked paths changed, no historical configs left, `src/chimera_plugin.py` still unchanged/no-op, and `pyproject.toml` still excludes `legacy*`. No nested README or new tests.
- Specifically document **before/after** active `[project].dependencies`, confirm `mamba-ssm` and unused pinned old WSM GPU/audio/video libraries no longer appear as direct dependencies, and provide the exact output/status of `python -m pip install -e . --dry-run` (and, if safe and dependencies already present, `python -m pip install -e .`). This is a packaging smoke check, not a research test. Do not present a workaround such as `pip install --no-build-isolation` as the fix.
- Full commit SHA, exact `origin/main` base SHA, branch, normal non-force push and clean worktree. Do not create a PR or merge: manager handles review/merge on PASS.
- Handoff sections: Outcome; Changed files; Verification; Data/Test access; Git handoff; Blockers and risks; Next atomic step. Stop at manager review: no additional cleanup, feature extraction, diarization, training or data-layer work.
