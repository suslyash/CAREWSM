# TASK-009 — Fixed-checkpoint TEST evaluation preparation

## Goal
Prepare an **owner-invoked**, read-only evaluation workflow for a completed CARE+WSM audio fold, using the VAL-selected best checkpoint. Implement and statically verify the workflow, but **do not run training, smoke, TEST, MLflow, or cleanup**. Nothing should start automatically.

## Scientific protocol
- Train remains joint CARE+WSM, VAL is CARE/WSM-separated and determines model selection exclusively. TEST has fixed CARE and WSM partitions, never pooled for the final numbers.
- Load exactly the best checkpoint selected by VAL `joint/average_macro_f1`, not last checkpoint. Verify checkpoint-to-config/fold/feature/model consistency; refuse ambiguity or missing artifacts.
- A selected checkpoint is evaluated once on CARE TEST and WSM TEST independently when the owner explicitly invokes it. TEST results must never influence early stopping, architecture/feature selection or further tuning. Note in docs that repeated TEST peeking harms independence.
- Both independent binary heads: Depression, Parkinson; masked unobserved labels. Per corpus and per task report Macro F1, Weighted F1, Macro Recall/UAR, Weighted Recall, Patient Recall, Healthy Recall, observed support; arithmetic CARE/WSM average per task, only where both values defined. Show undefined metrics as N/A, not zero and not silently dropped.
- Output machine-readable results plus human-readable `test_summary.txt` in a clearly designated evaluation subdirectory of the original run. Preserve all training logs and checkpoints; do not overwrite an existing evaluation unless explicitly requested with safe, documented semantics. Optional MLflow logging only if explicitly owner-invoked and existing tracking data preserved.

## Implementation
- Add a **single separate Bash entrypoint** `scripts/evaluate_audio_checkpoint.sh` for TEST evaluation, whose arguments identify one completed run directory and/or checkpoint plus its frozen YAML. It must not alter `scripts/run_audio_experiments.sh` semantics. A small Python evaluation implementation may be needed to load PyTorch checkpoint and use existing CARE+WSM DataModule; use project registration and existing metrics, never duplicate split logic or reimplement training.
- Provide `--help` and a non-executing `--check` mode which checks paths, provenance and dependency readiness without loading TEST arrays, running inference or modifying files. No args: show help.
- First inspect Chimera-ML checkpoint saving, model load, data transforms and metrics; ensure reproducible loading of saved weights, correct sequence masks, and no optimizer/gradient updates. Use `eval()` + `inference_mode()`. A checkpoint from another fold/family must be rejected.
- Configs and feature matrix unchanged. No new seed sweeps, experiments or launchers for training. Keep readable PEP8 code, named Chimera registry factories and project conventions. Do not touch existing run/test data.

## Verification
Static-only syntax, import/config checks where safe, dry-run on nonexistent and simulated artifact paths (no TEST loading), `bash -n`, `git diff --check`. Clearly document what cannot be verified until an actual owner-run checkpoint exists.

Update `scripts/runs/README.md`, `docs/evidence/TASK-009_FIXED_CHECKPOINT_TEST_EN.md`, and `docs/PROGRESS_EN.md`. Commit to `codex/task-009-fixed-checkpoint-test` and normal push, no PR/merge. Provide eight-heading handoff and stop.
