# TASK-005 — First reproducible native-audio Chimera-ML training pipeline

## Goal
Connect TASK-002/003 DataModule and TASK-004 registered classifiers/loss/metrics into a **real, config-driven Chimera-ML training run** for joint CARE+WSM, starting with a small, bounded **fold-0 native eGeMAPS LLD** experiment. Prepare an extensible run configuration for the three architecture families, but do not start a broad sweep. Keep the programme **audio-only**.

## Work procedure
Read `AGENTS.md`, TASK-002–004 evidence, WSM's existing Chimera logger/callback configs and the installed **chimera-ml 0.3.0** training APIs. Branch `codex/task-005-training-pipeline` from latest `origin/main`. Explicit decorators for all newly introduced registry objects; full readable style; reuse built-in components instead of rewriting them.

## Scientific and split contract
- Joint TRAIN: CARE fold-0 train + WSM fold-0 train; joint VAL: matching two fold-0 validation subsets; CARE/WSM separated VAL reporting. Do **not** add DEV independently. Keep fixed CARE/WSM TEST isolated and never use TEST for selection, early stopping, checkpoint monitoring or tuning.
- Two independent heads, Depression and Parkinson; absent labels masked for both loss and metrics; only TRAIN known labels may inform pos_weight (if used). Start with **unweighted masked BCE** as baseline. Make task-specific results explicit rather than silently reporting task-0 only.
- All six project metrics **per task** (Macro F1, Weighted F1, Macro Recall/UAR, Weighted Recall, Patient Recall, Healthy Recall) on joint and per-corpus VAL. Keep sample/observed counts. CARE+WSM arithmetic Average of corresponding per-corpus metrics, never pooled Average. Any eventual TEST evaluation must use an already VAL-selected checkpoint and separately report CARE TEST and WSM TEST.
- Ensure input lengths, masks, feature dimensions, fitted preprocessing, and any sequence resampling are correctly handled. **Do not silently replace long real trajectories with first 32 frames**: the TASK-004 smoke-only truncation is not an acceptable training representation. For transformer compute limits, choose and document a defensible *mask-safe, time-preserving* bounded-length policy or fail clearly; do not silently discard the rest of a video. Benchmark resource needs briefly.
- Native representation first; no Mantis installation, no Mantis-derived training in this task.

## Runtime and configuration
- Construct actual Chimera-ML training run with YAML-driven data/model/loss/metrics, optimizer, seed, and VAL checkpoint selection. Reuse Chimera's `console_file_logger`, `mlflow_logger`, `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback` where supported in 0.3.0. WSM local `wsm_summary_callback` may be adapted only if truly needed; no duplicate logging framework.
- Make one minimal reproducible fold-0 configuration for **MPTSNet** or pooled reference baseline that **actually completes 1 small epoch / bounded training steps and VAL evaluation** on real data. If compute/access limits prevent completion, provide a precise blocker rather than a simulated run. Include a separate MD-Former config only if simple; no broad or repeated training.
- Expose an optional TSCMamba config but leave it **BLOCKED** without `mamba_ssm`; do not substitute architecture or auto-install dependencies.
- Existing metric objects are registered but check their config instantiation and per-task `task=0/1` wiring through the *real* Chimera trainer; correct narrowly if broken. No metrics from synthetic batches presented as research results.
- Keep the first run small and identify it as **pipeline smoke**, not a scientifically valid model ranking. No feature screening, Optuna, full 5-fold campaign, or TEST-driven selection.

## Deliverables / acceptance
- YAML config(s) under `configs/audio/`, a small run wrapper under `scripts/` only if needed, minimal fixes to `src/audio/` to make trainer/model/loss/metrics/callback contract real.
- `docs/evidence/TASK-005_TRAINING_PIPELINE_EN.md`: exact command and environment, model, feature, split counts/fingerprint, batch/sequence policy, training/VAL observations, per-task metric output and observed-label counts, logger/checkpoint evidence, blockers, run artifacts and limitations. Clearly mark bounded smoke outputs as non-comparative.
- One bounded real-data Chimera training/VAL execution; `py_compile`, focused lint/format and `git diff --check`. No extra test suite. Update only `docs/PROGRESS_EN.md` among management docs.
- Normal commit/push, no PR/merge. Eight-heading handoff, stop for review.

**Manager will review and, on PASS, immediately prepare the next atomic phase.**
