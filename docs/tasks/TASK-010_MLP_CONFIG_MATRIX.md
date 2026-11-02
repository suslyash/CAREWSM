# TASK-010 — Explicit MLP baseline configs for all audio families

## Goal
Add the existing registered `carewsm_pooled_mlp` baseline to the owner-controlled audio experiment matrix. Exactly **55 new literal YAML files**: `configs/audio/<feature>/mlp_fold0.yaml` through `mlp_fold4.yaml` for each of the 11 established feature folders. No training.

## Required implementation
- Inspect `AGENTS.md`, the TASK-007/008 matrix and existing `configs/audio/<feature>/mptsnet_fold*.yaml`; verify the current registered `carewsm_pooled_mlp` model factory/constructor and `input_dim`. Do not invent feature families or dimensions.
- Add exactly five explicit MLP YAML files for each existing feature folder. Each must state numeric fold 0–4, correct registry feature family and verified input dimension, model `carewsm_pooled_mlp`, fixed seed 5, existing CARE+WSM joint loaders, masking, real full train/VAL folds, normal `logs/audio/<feature>/` base path, MLflow, summary, checkpoint, snapshot, early stopping with **VAL-only** `joint/average_macro_f1`, `mode: max`. Keep established optimizer, epochs and sequence handling unless scientifically justified otherwise.
- **ONE Bash launcher** `scripts/run_audio_experiments.sh`: retain `--list`, `--check`, `--feature`, `--model`, `--all`, and no-argument help. Ensure `--feature NAME --model mlp` selects exactly five static YAML files. Update hardcoded preflight model registry mapping/expected model counts, and verify all **165** configs. `--all` must continue to select all existing actual YAML files without editing/generating them. Do not introduce a Python launcher, extra Bash launchers, dynamic fold substitution, or seed sweeps.
- For documentation and science, MLP is a **pooled baseline** on both T=1 and temporal features, not a substitute for temporal architectures. Make the interpretation explicit; maintain static/sequence labels and never claim MLP evaluates temporal dynamics.
- Use existing project code; make no MLP architecture changes unless a genuine runtime incompatibility is evident and can be corrected minimally. Keep PEP8/readable formatting. Never silently ignore YAML parameters.

## Verification
- Static YAML parse and config validation; matrix check for all 11 families × 3 models × folds 0–4; `bash -n`; `scripts/run_audio_experiments.sh --check --all`; `--check --feature egemaps_func --model mlp`; `git diff --check`.
- **Never launch training, smoke, MLflow runs, TEST inference, or cleanup**. No generated numerical results. Do not alter canonical splits.
- Update `scripts/runs/README.md`, `docs/evidence/TASK-010_MLP_CONFIG_MATRIX_EN.md`, and `docs/PROGRESS_EN.md`.

## Handoff
Work on `codex/task-010-mlp-configs`; commit + normal push, no PR/merge; report eight-heading handoff.
